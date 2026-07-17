"""Error analysis and local explanations for the Extra Trees models.

This script is intentionally separate from training: explainability artifacts are
report outputs and are not needed by the timed prediction pipeline. It creates
the Tree SHAP feature-group plots used in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent / "solution"
sys.path.insert(0, str(_ROOT))
_CACHE_ROOT = _ROOT / "artifacts" / "cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clean import DEFAULT_IMAGE_SIZE, clean_image_bytes
from prepare import FEATURE_NAMES, FEATURE_VERSION


GROUPS = (
    ("False Positives", 0, 1, False),
    ("False Negatives", 1, 0, True),
    ("True Positives", 1, 1, False),
    ("True Negatives", 0, 0, True),
)


@dataclass(frozen=True)
class Example:
    index: int
    row_id: int
    source_class: int
    true_label: int
    predicted_label: int
    score: float
    threshold: float
    group: str
    image: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the report's Tree SHAP explanations.")
    parser.add_argument("--split", default="validation_augmented")
    parser.add_argument("--task", choices=("task02", "task03"), default="task03")
    parser.add_argument("--examples_per_group", type=int, default=5)
    return parser.parse_args()


def paths() -> tuple[Path, Path]:
    data_dir = _ROOT / "data"
    if not data_dir.exists():
        data_dir = _ROOT / "data-readonly"
    artifacts_dir = _ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return data_dir, artifacts_dir


def load_model(artifacts_dir: Path, task: str, split: str):
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit("explainability.py needs scikit-learn from requirements.txt") from exc

    model_dir = artifacts_dir / f"{task}_features"
    bundle = joblib.load(model_dir / "model.joblib")
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(f"The {task} model does not match the current feature extractor.")
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError(f"The {task} model and feature names do not match.")
    model = bundle["model"]
    model.set_params(n_jobs=1)

    threshold_payload = json.loads((model_dir / "threshold.json").read_text())
    threshold = float(threshold_payload["threshold"])
    prepared_path = artifacts_dir / "prepared" / "task02_features" / f"{split}.npz"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Missing {prepared_path}; run prepare.py first.")
    with np.load(prepared_path) as prepared:
        features = prepared["features"].astype(np.float32)
        labels = prepared["labels"].astype(np.int8)
        source_classes = prepared["source_class"].astype(np.int8)
    scores = np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    if "quality_thresholds" in threshold_payload:
        image_path = artifacts_dir / "prepared" / "task02" / f"{split}.npz"
        with np.load(image_path) as prepared_images:
            images = prepared_images["images"].astype(np.float32)
        luminance = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
        contrasts = luminance.std(axis=(1, 2))
        edges = np.asarray(threshold_payload["quality_edges"], dtype=np.float64)
        quality_thresholds = np.asarray(threshold_payload["quality_thresholds"], dtype=np.float64)
        thresholds = quality_thresholds[np.digitize(contrasts, edges[1:-1])]
    else:
        thresholds = np.full(len(scores), threshold, dtype=np.float64)
    predictions = (scores >= thresholds).astype(np.int8)
    return model, features, labels, source_classes, scores, predictions, thresholds


def select_examples(
    labels: np.ndarray,
    predictions: np.ndarray,
    margins: np.ndarray,
    per_group: int,
) -> dict[str, list[int]]:
    """Select confident errors and confident correct predictions deterministically."""

    selected: dict[str, list[int]] = {}
    for name, true_label, predicted_label, ascending in GROUPS:
        candidates = np.flatnonzero(
            (labels == true_label) & (predictions == predicted_label)
        )
        order = np.argsort(margins[candidates], kind="stable")
        if not ascending:
            order = order[::-1]
        selected[name] = candidates[order[:per_group]].astype(int).tolist()
    return selected


def load_selected_images(
    split_dir: Path,
    selected: dict[str, list[int]],
    labels: np.ndarray,
    predictions: np.ndarray,
    source_classes: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, list[Example]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("explainability.py needs pyarrow from requirements.txt") from exc

    index_to_group = {
        index: group for group, indices in selected.items() for index in indices
    }
    examples: dict[str, list[Example]] = {name: [] for name, *_ in GROUPS}
    global_index = 0
    parquet_paths = sorted(split_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        has_row_ids = "row_id" in parquet.schema.names
        columns = ["image", "source_class"]
        if has_row_ids:
            columns.insert(0, "row_id")
        for batch in parquet.iter_batches(
            batch_size=256, columns=columns
        ):
            rows = batch.to_pydict()
            row_ids = rows["row_id"] if has_row_ids else range(
                global_index, global_index + len(rows["image"])
            )
            for row_id, image_bytes, source_class in zip(
                row_ids, rows["image"], rows["source_class"]
            ):
                if global_index in index_to_group:
                    expected_source = int(source_classes[global_index])
                    if int(source_class) != expected_source:
                        raise RuntimeError("Raw and prepared split row orders do not match.")
                    group = index_to_group[global_index]
                    examples[group].append(
                        Example(
                            index=global_index,
                            row_id=int(row_id),
                            source_class=expected_source,
                            true_label=int(labels[global_index]),
                            predicted_label=int(predictions[global_index]),
                            score=float(scores[global_index]),
                            threshold=float(thresholds[global_index]),
                            group=group,
                            image=clean_image_bytes(image_bytes, DEFAULT_IMAGE_SIZE),
                        )
                    )
                global_index += 1
    if global_index != len(labels):
        raise RuntimeError(
            f"Raw split has {global_index} rows but prepared features have {len(labels)}."
        )
    for group, indices in selected.items():
        by_index = {example.index: example for example in examples[group]}
        examples[group] = [by_index[index] for index in indices]
    return examples


def positive_class_shap(model, features: np.ndarray) -> tuple[np.ndarray, float]:
    """Return exact Tree SHAP contributions for the AI-class probability."""

    try:
        import shap
    except ImportError as exc:
        raise SystemExit(
            "Tree SHAP explanations need shap; install solution/requirements.txt."
        ) from exc

    def explain_tree_ensemble(tree_model) -> tuple[np.ndarray, float]:
        positive_index = int(np.flatnonzero(np.asarray(tree_model.classes_) == 1)[0])
        explainer = shap.TreeExplainer(
            tree_model, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        raw_values = explainer.shap_values(features, check_additivity=True)
        if isinstance(raw_values, list):
            values = np.asarray(raw_values[positive_index], dtype=np.float64)
        else:
            values_array = np.asarray(raw_values, dtype=np.float64)
            values = (
                values_array[:, :, positive_index]
                if values_array.ndim == 3
                else values_array
            )
        expected = np.asarray(explainer.expected_value, dtype=np.float64)
        base_value = float(
            expected.reshape(-1)[positive_index]
            if expected.size > 1
            else expected.item()
        )
        return values, base_value

    tree_models = getattr(model, "models", [model])
    explanations = [explain_tree_ensemble(tree_model) for tree_model in tree_models]
    return (
        np.mean([values for values, _ in explanations], axis=0),
        float(np.mean([base for _, base in explanations])),
    )


def explanation_group(name: str) -> str:
    """Map a feature name to a small, plain-language group."""

    if name.startswith("patch_all_rgb_") and "_mean_" in name:
        return "Patch color level"
    if name.startswith("patch_all_rgb_") and "_std_" in name:
        return "Patch color spread"
    if name.startswith("patch_all_luminance_mean_"):
        return "Patch brightness"
    if name.startswith("patch_all_luminance_std_"):
        return "Patch brightness spread"
    if name.startswith("patch_all_diversity_"):
        return "Patch edge strength"
    if name.startswith(("patch_all_cross_residual_", "patch_all_laplacian_")):
        return "Patch residuals"
    if name.startswith("patch_all_lowbit_") and "_hist_" in name:
        return "Patch low-bit counts"
    if name.startswith("patch_all_lowbit_") and "gradient" in name:
        return "Patch low-bit changes"
    if name.startswith("patch_all_lowbit_"):
        return "Other patch low-bit stats"
    if name.startswith("texture_contrast_"):
        return "Smooth vs. detailed patches"
    if name.startswith("lowbit_max_gradient_patch_"):
        return "Noisiest low-bit patch"
    return "Compression blocks"


def plot_shap_subgroups(
    path: Path,
    examples: dict[str, list[Example]],
    features: np.ndarray,
    model,
) -> tuple[dict, float, float]:
    """Show which understandable feature groups drive each selected example."""

    selected = [example for name, *_ in GROUPS for example in examples[name]]
    shap_values, base_value = positive_class_shap(
        model, features[[example.index for example in selected]]
    )
    reconstructed = base_value + shap_values.sum(axis=1)
    maximum_error = float(
        np.max(np.abs(reconstructed - np.asarray([example.score for example in selected])))
    )
    if maximum_error > 1e-5:
        raise RuntimeError(
            f"Tree SHAP additivity check failed (maximum error {maximum_error:g})."
        )
    group_names = list(dict.fromkeys(explanation_group(name) for name in FEATURE_NAMES))
    group_indices = {
        group: np.array(
            [i for i, name in enumerate(FEATURE_NAMES) if explanation_group(name) == group]
        )
        for group in group_names
    }
    individual_values = np.vstack(
        [shap_values[:, indices].sum(axis=1) for indices in group_indices.values()]
    )
    order = np.argsort(np.mean(np.abs(individual_values), axis=1))[::-1]
    individual_values = individual_values[order]
    labels = [group_names[i] for i in order]
    short_names = {
        "False Positives": "FP: real called AI",
        "False Negatives": "FN: AI called real",
        "True Positives": "TP: AI called AI",
        "True Negatives": "TN: real called real",
    }
    columns = []
    column_labels = []
    outcome_means = {}
    start = 0
    block_centres = []
    for name, *_ in GROUPS:
        group_examples = examples[name]
        stop = start + len(group_examples)
        block = individual_values[:, start:stop]
        mean = block.mean(axis=1, keepdims=True)
        columns.extend([block, mean])
        column_labels.extend([f"row {example.row_id}" for example in group_examples] + ["Mean"])
        block_centres.append((len(column_labels) - (len(group_examples) + 2) / 2, short_names[name]))
        outcome_means[name] = {
            label: float(value) for label, value in zip(labels, mean[:, 0])
        }
        start = stop
    values = np.hstack(columns)
    limit = float(np.max(np.abs(values)))
    fig, ax = plt.subplots(figsize=(15.5, 6.8))
    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
    ax.set_xticks(np.arange(len(column_labels)), column_labels, rotation=55, ha="right", fontsize=7)
    for centre, group_label in block_centres:
        ax.text(centre, 1.02, group_label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for boundary in np.cumsum([len(examples[name]) + 1 for name, *_ in GROUPS])[:-1]:
        ax.axvline(boundary - 0.5, color="black", linewidth=1.2)
    ax.set_title("Tree SHAP split into feature groups (red pushes AI, blue pushes real)", pad=35)
    fig.colorbar(image, ax=ax, label="SHAP contribution to AI score", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return (
        {
            "plot": path.name,
            "examples": len(selected),
            "groups": labels,
            "mean_contributions_by_outcome": outcome_means,
        },
        base_value,
        maximum_error,
    )


def plot_lowbit_diagnostic(
    path: Path,
    examples: dict[str, list[Example]],
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Make the strongest low-bit feature of the worst false negative visible."""

    example = examples["False Negatives"][0]
    feature_name = "patch_all_lowbit_red_hist_7_max"
    feature_index = FEATURE_NAMES.index(feature_name)
    red_low = example.image[..., 0] & np.uint8(7)
    patch_fraction = (
        (red_low == 7)
        .reshape(8, 32, 8, 32)
        .transpose(0, 2, 1, 3)
        .mean(axis=(2, 3))
    )

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].imshow(example.image)
    axes[0].set_title(f"Worst false negative\nrow {example.row_id}, score {example.score:.3f}")
    low_image = axes[1].imshow(red_low, cmap="magma", vmin=0, vmax=7)
    axes[1].set_title("Red channel: lowest 3 bits\n(brighter means a larger value)")
    fig.colorbar(low_image, ax=axes[1], ticks=range(8), shrink=0.75)
    patch_image = axes[2].imshow(patch_fraction, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Share of low-bit value 7\nin each 32x32 patch")
    fig.colorbar(patch_image, ax=axes[2], shrink=0.75)
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    real_values = features[labels == 0, feature_index]
    ai_values = features[labels == 1, feature_index]
    bins = np.linspace(0, 1, 21)
    axes[3].hist(real_values, bins=bins, density=True, alpha=0.55, label="real")
    axes[3].hist(ai_values, bins=bins, density=True, alpha=0.55, label="AI")
    marker = float(features[example.index, feature_index])
    axes[3].axvline(marker, color="black", linestyle="--", label=f"row {example.row_id}: {marker:.2f}")
    axes[3].set_xlabel("Largest patch share of red low-bit value 7")
    axes[3].set_ylabel("density")
    axes[3].set_title("Feature distribution")
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot": path.name,
        "row_id": example.row_id,
        "feature": feature_name,
        "feature_value": marker,
    }


def main() -> int:
    args = parse_args()
    if args.examples_per_group <= 0:
        raise SystemExit("examples per group must be positive")

    data_dir, artifacts_dir = paths()
    model, features, labels, sources, scores, predictions, thresholds = load_model(
        artifacts_dir, args.task, args.split
    )
    selected = select_examples(
        labels, predictions, scores - thresholds, args.examples_per_group
    )
    examples = load_selected_images(
        data_dir / args.split,
        selected,
        labels,
        predictions,
        sources,
        scores,
        thresholds,
    )

    output_dir = artifacts_dir / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    subgroup_summary, base_value, maximum_error = plot_shap_subgroups(
        output_dir / f"{args.split}_tree_shap_subgroups.png",
        examples,
        features,
        model,
    )

    summary = {
        "split": args.split,
        "task": args.task,
        "threshold": (
            "quality-aware per image"
            if not np.allclose(thresholds, thresholds[0])
            else float(thresholds[0])
        ),
        "feature_version": FEATURE_VERSION,
        "groups": {name: len(examples[name]) for name, *_ in GROUPS},
        "selection": "most confident examples in each confusion-matrix outcome",
        "tree_shap": {
            "base_ai_probability": base_value,
            "maximum_additivity_error": maximum_error,
            "feature_perturbation": "tree_path_dependent",
            "interpretation": "positive values push the model toward AI; negative values push toward real",
            "limitations": [
                "SHAP explains this fitted model, not the true cause of an image being real or AI-generated.",
                "Correlated engineered features can share or redistribute attribution.",
                "Tree-path-dependent results depend on the fitted trees and their training-path counts.",
            ],
        },
        "tree_shap_subgroups": subgroup_summary,
    }
    if args.task == "task03":
        summary["lowbit_diagnostic"] = plot_lowbit_diagnostic(
            output_dir / f"{args.split}_lowbit_false_negative.png",
            examples,
            features,
            labels,
        )
    (output_dir / "explainability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
