"""Fully refit Extra Trees on clean and augmented engineered features."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import prepare_features
from clean import DEFAULT_IMAGE_SIZE, clean_image_bytes
from prepare_features import atomic_save_npz, extract_many, prepare_labeled_split
from train_features import (
    atomic_dump,
    atomic_write_json,
    balanced_sample_weights,
    load_prepared,
    metrics_at_threshold,
    model_bundle,
    predict_scores,
)


AUGMENTATION_VERSION = "single_operation_v1"
OPERATIONS = ("flip", "rotate", "grayscale", "contrast", "blur", "resize", "noise", "jpeg")
OPERATION_PROBABILITIES = (0.10, 0.10, 0.15, 0.20, 0.20, 0.10, 0.075, 0.075)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--max_features", type=float, default=0.35)
    parser.add_argument("--augmented_views", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def augment_image(
    image: np.ndarray, seed: int, row_index: int, view_index: int, view_count: int
) -> tuple[np.ndarray, str]:
    """Apply one deterministic operation; breadth comes from the whole dataset."""

    chooser = np.random.default_rng(np.random.SeedSequence([seed, row_index]))
    chosen = chooser.choice(
        OPERATIONS, size=view_count, replace=False, p=OPERATION_PROBABILITIES
    )
    operation = str(chosen[view_index])
    rng = np.random.default_rng(np.random.SeedSequence([seed, row_index, view_index, 1]))
    output = Image.fromarray(image, mode="RGB")
    if operation == "flip":
        output = ImageOps.mirror(output) if rng.random() < 0.75 else ImageOps.flip(output)
    elif operation == "rotate":
        output = output.rotate(int(rng.choice([90, 180, 270])))
    elif operation == "grayscale":
        output = ImageOps.grayscale(output).convert("RGB")
    elif operation == "contrast":
        output = ImageEnhance.Contrast(output).enhance(float(rng.uniform(0.45, 1.65)))
    elif operation == "blur":
        output = output.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 4.0))))
    elif operation == "resize":
        side = int(256 * float(rng.uniform(0.4, 0.9)))
        output = output.resize((side, side), Image.Resampling.BILINEAR).resize(
            (256, 256), Image.Resampling.BILINEAR
        )
    elif operation == "noise":
        array = np.asarray(output, dtype=np.float32)
        array += rng.normal(0, float(rng.uniform(2, 12)), array.shape)
        output = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    else:
        buffer = io.BytesIO()
        output.save(buffer, "JPEG", quality=int(rng.integers(20, 91)))
        output = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        output.load()
    return np.asarray(output, dtype=np.uint8), operation


def run_feature_preparation(timeout_seconds: int) -> None:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "--timeout_seconds", str(timeout_seconds)]
        prepare_features.main()
    finally:
        sys.argv = original_argv


def prepare_augmented_features(
    root: Path, args: argparse.Namespace, deadline: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    artifacts = root / "artifacts"
    output_dir = artifacts / "prepared" / "task03_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if not (
        summary.get("complete")
        and summary.get("version") == AUGMENTATION_VERSION
        and summary.get("seed") == args.seed
        and summary.get("views") == args.augmented_views
    ):
        counts: Counter[str] = Counter()
        outputs, row_offset = [], 0
        shards = sorted((artifacts / "cleaned_train_npz").glob("train_cleaned_*.npz"))
        if not shards:
            raise FileNotFoundError("Run clean.py before train_augmented.py.")
        for shard_index, path in enumerate(shards):
            with np.load(path) as data:
                images = data["images"]
                labels = data["labels"].astype(np.int8)
                sources = data["source_class"].astype(np.int8)
                for view_index in range(args.augmented_views):
                    augmented = np.empty_like(images)
                    for local_index, image in enumerate(images):
                        if time.time() >= deadline:
                            raise TimeoutError("Timed out while preparing augmented features.")
                        augmented[local_index], operation = augment_image(
                            image,
                            args.seed,
                            row_offset + local_index,
                            view_index,
                            args.augmented_views,
                        )
                        counts[operation] += 1
                    features = extract_many(
                        augmented, deadline, f"augmented view {view_index + 1} {path.name}"
                    )
                    name = f"train_augmented_v{view_index}_{shard_index:03d}.npz"
                    atomic_save_npz(
                        output_dir / name,
                        features=features,
                        labels=labels,
                        source_class=sources,
                    )
                    outputs.append(name)
            row_offset += len(labels)

        data_dir, _ = prepare_features.paths()
        calibration = prepare_labeled_split(
            data_dir, output_dir, "calibration_augmented", deadline
        )
        summary = {
            "complete": True,
            "version": AUGMENTATION_VERSION,
            "seed": args.seed,
            "views": args.augmented_views,
            "rows": row_offset * args.augmented_views,
            "outputs": outputs,
            "operation_counts": dict(sorted(counts.items())),
            "calibration_augmented": calibration,
        }
        atomic_write_json(summary_path, summary)

    features, labels, sources = [], [], []
    for name in summary["outputs"]:
        with np.load(output_dir / name) as data:
            features.append(data["features"].astype(np.float32))
            labels.append(data["labels"].astype(np.int8))
            sources.append(data["source_class"].astype(np.int8))
    with np.load(output_dir / summary["calibration_augmented"]["output"]) as data:
        calibration = (
            data["features"].astype(np.float32),
            data["labels"].astype(np.int8),
            data["source_class"].astype(np.int8),
        )
    return np.concatenate(features), np.concatenate(labels), np.concatenate(sources), calibration, summary


def empirical_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.19) -> dict:
    """Spend most of the FPR allowance while leaving a one-percentage-point margin."""

    real_scores = np.sort(scores[labels == 0])
    allowed = max(1, int(np.floor(target_fpr * len(real_scores))))
    threshold = float(np.nextafter(real_scores[-allowed], np.inf))
    false_positives = int(np.count_nonzero(real_scores >= threshold))
    return {
        "method": "tie_safe_empirical_fpr",
        "threshold": threshold,
        "target_fpr": target_fpr,
        "real_rows": int(len(real_scores)),
        "observed_false_positives": false_positives,
        "observed_fpr": false_positives / len(real_scores),
    }


def image_contrasts(data_dir: Path, split: str) -> np.ndarray:
    """Measure luminance contrast after the same cleaning used at inference."""

    values = []
    for row in prepare_features.parquet_rows(data_dir / split, ["image"]):
        image = clean_image_bytes(row["image"], DEFAULT_IMAGE_SIZE).astype(np.float32)
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        values.append(float(luminance.std()))
    return np.asarray(values, dtype=np.float32)


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0 or args.n_estimators <= 0 or args.min_samples_leaf <= 0:
        raise SystemExit("timeout, tree count, and leaf size must be positive")
    if not 0.0 < args.max_features <= 1.0:
        raise SystemExit("max features must lie in (0, 1]")
    if not 1 <= args.augmented_views <= len(OPERATIONS):
        raise SystemExit(f"augmented views must be between 1 and {len(OPERATIONS)}")
    started = time.time()
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    prepared_task2 = artifacts / "prepared" / "task02_features"
    summary_path = prepared_task2 / "summary.json"
    prepared = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if not prepared.get("complete"):
        run_feature_preparation(args.timeout_seconds)

    clean_x, clean_y, clean_source, splits, _ = load_prepared(prepared_task2)
    deadline = started + args.timeout_seconds * 0.72
    augmented_x, augmented_y, augmented_source, calibration_augmented, augmentation = (
        prepare_augmented_features(root, args, deadline)
    )
    train_x = np.concatenate((clean_x, augmented_x))
    train_y = np.concatenate((clean_y, augmented_y))
    train_source = np.concatenate((clean_source, augmented_source))

    from sklearn.ensemble import ExtraTreesClassifier

    _, class_weights = balanced_sample_weights(train_y)
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        class_weight={int(label): weight for label, weight in class_weights.items()},
        n_jobs=8,
        random_state=args.seed,
    )
    model.fit(train_x, train_y)
    model.set_params(n_jobs=1)

    task_dir = artifacts / "task03_features"
    task_dir.mkdir(parents=True, exist_ok=True)
    atomic_dump(task_dir / "model.joblib", model_bundle(model, args, len(model.estimators_)))
    schema = json.loads((prepared_task2 / "schema.json").read_text())
    atomic_write_json(task_dir / "schema.json", schema)

    calibration_domains = {
        "calibration": splits["calibration"],
        "calibration_augmented": calibration_augmented,
    }
    data_dir, _ = prepare_features.paths()
    scores = {
        name: predict_scores(model, features)
        for name, (features, _, _) in calibration_domains.items()
    }
    contrasts = {name: image_contrasts(data_dir, name) for name in calibration_domains}
    real_contrasts = np.concatenate(
        [contrasts[name][calibration_domains[name][1] == 0] for name in calibration_domains]
    )
    quality_edges = np.quantile(real_contrasts, np.linspace(0, 1, 3))
    quality_edges[0], quality_edges[-1] = -np.inf, np.inf
    quality_thresholds = []
    for lower, upper in zip(quality_edges[:-1], quality_edges[1:]):
        bin_scores, bin_labels = [], []
        for name, (_, labels, _) in calibration_domains.items():
            selected = (contrasts[name] >= lower) & (contrasts[name] < upper)
            bin_scores.append(scores[name][selected])
            bin_labels.append(labels[selected])
        quality_thresholds.append(
            empirical_threshold(np.concatenate(bin_scores), np.concatenate(bin_labels))["threshold"]
        )
    threshold_info = {
        "complete": True,
        "method": "two_bin_quality_aware_empirical_threshold",
        "threshold": 0.0,
        "quality_metric": "cleaned_luminance_std",
        "quality_edges": quality_edges.tolist(),
        "quality_thresholds": quality_thresholds,
        "calibration_target_fpr": 0.19,
        "feature_version": prepare_features.FEATURE_VERSION,
        "model_estimators": len(model.estimators_),
    }
    atomic_write_json(task_dir / "threshold.json", threshold_info)

    evaluation = {}
    for name, (features, labels, sources) in {**calibration_domains, **splits}.items():
        split_scores = scores[name] if name in scores else predict_scores(model, features)
        split_contrasts = contrasts[name] if name in contrasts else image_contrasts(data_dir, name)
        bins = np.digitize(split_contrasts, quality_edges[1:-1])
        evaluation[name] = metrics_at_threshold(
            split_scores - np.asarray(quality_thresholds)[bins], labels, sources, 0.0
        )
    metrics = {
        "complete": True,
        "seconds": round(time.time() - started, 2),
        "args": vars(args),
        "training_rows": int(len(train_y)),
        "clean_training_rows": int(len(clean_y)),
        "augmented_training_rows": int(len(augmented_y)),
        "augmentation": augmentation,
        "threshold_calibration": threshold_info,
        **evaluation,
    }
    atomic_write_json(task_dir / "metrics.json", metrics)
    print(json.dumps({name: evaluation[name] for name in ("validation", "validation_augmented")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
