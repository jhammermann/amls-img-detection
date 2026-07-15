"""Train and calibrate the engineered-feature Task 2 alternative."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Set thread limits before importing NumPy/scikit-learn native extensions.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "8"

import numpy as np

from prepare_features import FEATURE_NAMES, FEATURE_VERSION


MAX_FPR = 0.20
THRESHOLD_FAILURE_PROBABILITY = 0.05
ONE_SIDED_NORMAL_95 = 1.6448536269514722


def artifacts_path() -> Path:
    path = Path(__file__).resolve().parent / "artifacts"
    path.mkdir(exist_ok=True)
    return path


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def atomic_dump(path: Path, payload: dict) -> None:
    import joblib

    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def load_prepared(
    prepared_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict,
]:
    summary_path = prepared_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}; run prepare_features.py first.")
    summary = json.loads(summary_path.read_text())
    if not summary.get("complete"):
        raise RuntimeError("Feature preparation did not complete; rerun prepare_features.py.")
    if summary.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"Prepared feature version {summary.get('feature_version')!r} does not match "
            f"{FEATURE_VERSION!r}."
        )
    if summary.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("Prepared feature names do not match the current extractor.")

    training_features, training_labels, training_sources = [], [], []
    for output in summary["training"]:
        with np.load(prepared_dir / output["output"]) as data:
            training_features.append(data["features"].astype(np.float32))
            training_labels.append(data["labels"].astype(np.int8))
            training_sources.append(data["source_class"].astype(np.int8))

    splits = {}
    for split, output in summary["splits"].items():
        with np.load(prepared_dir / output["output"]) as data:
            splits[split] = (
                data["features"].astype(np.float32),
                data["labels"].astype(np.int8),
                data["source_class"].astype(np.int8),
            )

    train_x = np.concatenate(training_features)
    train_y = np.concatenate(training_labels)
    train_source = np.concatenate(training_sources)
    expected_columns = len(FEATURE_NAMES)
    named_feature_arrays = [
        ("training", train_x),
        *((name, values[0]) for name, values in splits.items()),
    ]
    for name, features in named_feature_arrays:
        if features.ndim != 2 or features.shape[1] != expected_columns:
            raise RuntimeError(
                f"{name} features have shape {features.shape}; expected (*, {expected_columns})."
            )
        if not np.isfinite(features).all():
            raise RuntimeError(f"{name} features contain non-finite values.")
    return train_x, train_y, train_source, splits, summary


def wilson_upper_bound(
    false_positives: int,
    real_count: int,
    z: float = ONE_SIDED_NORMAL_95,
) -> float:
    """Return a one-sided Wilson upper confidence bound for a binomial rate."""

    if real_count <= 0:
        raise ValueError("At least one real calibration image is required.")
    if not 0 <= false_positives <= real_count:
        raise ValueError("false_positives must lie between zero and real_count")
    proportion = false_positives / real_count
    z_squared = z * z
    numerator = proportion + z_squared / (2 * real_count)
    numerator += z * math.sqrt(
        proportion * (1.0 - proportion) / real_count
        + z_squared / (4 * real_count * real_count)
    )
    return float(numerator / (1.0 + z_squared / real_count))


def clopper_pearson_upper_bound(
    false_positives: int,
    real_count: int,
    confidence: float = 0.95,
) -> float:
    """Return the exact one-sided Clopper--Pearson upper confidence bound."""

    if real_count <= 0:
        raise ValueError("At least one real calibration image is required.")
    if not 0 <= false_positives <= real_count:
        raise ValueError("false_positives must lie between zero and real_count")
    if false_positives == real_count:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(confidence, false_positives + 1, real_count - false_positives))


def threshold_at_bounded_fpr(
    scores: np.ndarray,
    labels: np.ndarray,
    max_fpr: float = MAX_FPR,
    delta: float = THRESHOLD_FAILURE_PROBABILITY,
) -> dict:
    """Select a tie-safe Neyman--Pearson order-statistic threshold.

    The order statistic is fixed only by the real calibration sample count,
    ``max_fpr`` and ``delta``.  AI calibration scores do not affect it.  With
    probability at least ``1 - delta`` over the real calibration sample, the
    population false-positive rate of the selected threshold is at most
    ``max_fpr`` (under the usual exchangeable-sample assumption).
    """

    from scipy.stats import binom

    if not 0.0 < max_fpr < 1.0:
        raise ValueError("max_fpr must lie strictly between zero and one")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    real_scores = scores[labels == 0]
    if len(real_scores) == 0:
        raise ValueError("Calibration data contains no real images.")
    if not np.isfinite(real_scores).all():
        raise ValueError("Calibration scores contain non-finite values.")

    real_count = len(real_scores)
    order_k = None
    violation_probability = None
    for candidate_k in range(1, real_count + 1):
        candidate_probability = float(
            binom.sf(candidate_k - 1, real_count, 1.0 - max_fpr)
        )
        if candidate_probability <= delta:
            order_k = candidate_k
            violation_probability = candidate_probability
            break
    if order_k is None or violation_probability is None:
        raise ValueError(
            f"{real_count} real calibration rows are insufficient for a distribution-free "
            f"FPR={max_fpr:g}, delta={delta:g} threshold."
        )

    sorted_real_scores = np.sort(real_scores)
    boundary_score = float(sorted_real_scores[order_k - 1])
    # Predictions use score >= threshold.  Advancing by one representable float
    # excludes every score tied at the selected order statistic.
    threshold = float(np.nextafter(boundary_score, np.inf))
    false_positives = int(np.count_nonzero(real_scores >= threshold))
    allowed_false_positives = int(real_count - order_k)
    if false_positives > allowed_false_positives:
        raise AssertionError("Tie-safe threshold exceeded its order-statistic FP limit.")

    confidence = 1.0 - delta
    wilson_upper = wilson_upper_bound(false_positives, real_count)
    clopper_pearson_upper = clopper_pearson_upper_bound(
        false_positives, real_count, confidence
    )
    if wilson_upper > max_fpr + 1e-12 or clopper_pearson_upper > max_fpr + 1e-12:
        raise AssertionError(
            "Order-statistic threshold did not satisfy its one-sided FPR bounds."
        )
    return {
        "method": "neyman_pearson_order_statistic",
        "threshold": threshold,
        "max_fpr": float(max_fpr),
        "alpha": float(max_fpr),
        "delta": float(delta),
        "confidence": float(confidence),
        "real_rows": int(real_count),
        "order_statistic_k": int(order_k),
        "order_statistic_score": boundary_score,
        "allowed_false_positives": allowed_false_positives,
        "observed_false_positives": false_positives,
        "calibration_fpr": float(false_positives / real_count),
        "violation_probability_bound": float(violation_probability),
        "wilson_upper_fpr": wilson_upper,
        "clopper_pearson_upper_fpr": clopper_pearson_upper,
        "tie_safe": True,
    }


def metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    source_classes: np.ndarray,
    threshold: float,
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    predictions = (scores >= threshold).astype(np.int8)
    real = labels == 0
    ai = labels == 1
    true_negatives = int(np.sum((predictions == 0) & real))
    false_positives = int(np.sum((predictions == 1) & real))
    false_negatives = int(np.sum((predictions == 0) & ai))
    true_positives = int(np.sum((predictions == 1) & ai))

    per_source_class = {}
    for source_class in sorted(int(value) for value in np.unique(source_classes)):
        selected = source_classes == source_class
        binary_target = 0 if source_class == 0 else 1
        per_source_class[str(source_class)] = {
            "rows": int(np.sum(selected)),
            "positive_prediction_rate": float(np.mean(predictions[selected] == 1)),
            "recall_for_binary_target": float(
                np.mean(predictions[selected] == binary_target)
            ),
        }

    return {
        "threshold": float(threshold),
        "rows": int(len(labels)),
        "real_rows": int(np.sum(real)),
        "ai_rows": int(np.sum(ai)),
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "fpr_real": false_positives / max(1, false_positives + true_negatives),
        "recall_ai": true_positives / max(1, true_positives + false_negatives),
        "precision_ai": true_positives / max(1, true_positives + false_positives),
        "specificity_real": true_negatives / max(1, true_negatives + false_positives),
        "accuracy": float(np.mean(predictions == labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "per_source_class": per_source_class,
    }


def predict_scores(model, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)


def model_bundle(model, args: argparse.Namespace, iteration: int) -> dict:
    return {
        "model_type": "sklearn_extra_trees",
        "model": model,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "pixel_only": True,
        "iteration": int(iteration),
        "args": vars(args),
    }


def balanced_sample_weights(labels: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    labels = np.asarray(labels)
    real_count = int(np.count_nonzero(labels == 0))
    ai_count = int(np.count_nonzero(labels == 1))
    if real_count == 0 or ai_count == 0 or real_count + ai_count != len(labels):
        raise ValueError("Training labels must contain both binary classes 0 and 1.")
    row_count = len(labels)
    weights_by_class = {
        "0": float(row_count / (2.0 * real_count)),
        "1": float(row_count / (2.0 * ai_count)),
    }
    weights = np.where(
        labels == 0,
        weights_by_class["0"],
        weights_by_class["1"],
    ).astype(np.float64)
    return weights, weights_by_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--stage_estimators", type=int, default=100)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0 or args.n_estimators <= 0 or args.stage_estimators <= 0:
        raise SystemExit("timeout and estimator arguments must be positive")
    if args.min_samples_leaf <= 0:
        raise SystemExit("--min_samples_leaf must be positive")

    try:
        import joblib
        import sklearn
        from sklearn.ensemble import ExtraTreesClassifier
    except ImportError as exc:
        raise SystemExit("train_features.py needs scikit-learn from requirements.txt.") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    # Reserve time for fixed-model calibration, evaluation and final writes.
    training_deadline = start + args.timeout_seconds * 0.88
    artifacts_dir = artifacts_path()
    prepared_dir = artifacts_dir / "prepared" / "task02_features"
    task_dir = artifacts_dir / "task02_features"
    task_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_source, splits, prepared_summary = load_prepared(prepared_dir)
    atomic_write_json(
        task_dir / "schema.json",
        {
            "feature_version": FEATURE_VERSION,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "pixel_only": True,
        },
    )
    atomic_write_json(task_dir / "arguments.json", vars(args))
    # Invalidate prior-run calibration/evaluation before replacing checkpoints,
    # so an interrupted rerun cannot pair a new model with a stale threshold.
    atomic_write_json(
        task_dir / "threshold.json",
        {"complete": False, "feature_version": FEATURE_VERSION},
    )
    atomic_write_json(
        task_dir / "metrics.json",
        {"complete": False, "feature_version": FEATURE_VERSION},
    )

    _, weights_by_class = balanced_sample_weights(train_y)
    classifier_class_weights = {
        int(label): weight for label, weight in weights_by_class.items()
    }
    model = ExtraTreesClassifier(
        n_estimators=min(args.stage_estimators, args.n_estimators),
        criterion="gini",
        max_features="sqrt",
        min_samples_leaf=args.min_samples_leaf,
        # Explicit fixed weights are warm-start safe and exactly equivalent to
        # sklearn's binary ``balanced`` preset for this training set.
        class_weight=classifier_class_weights,
        bootstrap=False,
        n_jobs=8,
        random_state=args.seed,
        warm_start=True,
    )

    stage_history = []
    completed_iteration = 0
    while completed_iteration < args.n_estimators:
        if time.time() >= training_deadline:
            if completed_iteration == 0:
                raise TimeoutError("Training timeout elapsed before the first checkpoint.")
            break
        if completed_iteration > 0:
            remaining_seconds = training_deadline - time.time()
            last_stage_seconds = stage_history[-1]["seconds"]
            if remaining_seconds <= max(5.0, 1.2 * last_stage_seconds):
                break

        target_estimators = min(
            completed_iteration + args.stage_estimators,
            args.n_estimators,
        )
        model.set_params(n_estimators=target_estimators)
        stage_start = time.time()
        model.fit(train_x, train_y)
        iteration = len(model.estimators_)
        if iteration != target_estimators:
            raise RuntimeError(f"ExtraTrees fitted {iteration} trees; expected {target_estimators}.")
        fit_seconds = time.time() - stage_start
        # Save every checkpoint in deterministic serial-inference mode. Restore
        # eight workers only while constructing the next tree stage.
        model.set_params(n_jobs=1)
        bundle = model_bundle(model, args, iteration)
        atomic_dump(task_dir / "model.joblib", bundle)
        stage = {
            "iteration": iteration,
            "estimators_added": int(iteration - completed_iteration),
            "fit_seconds": round(fit_seconds, 3),
            "seconds": round(time.time() - stage_start, 3),
            "cumulative_seconds": round(time.time() - start, 3),
        }
        stage_history.append(stage)
        completed_iteration = iteration
        print(json.dumps(stage, sort_keys=True), flush=True)
        if completed_iteration < args.n_estimators:
            model.set_params(n_jobs=8)

    if completed_iteration == 0:
        raise RuntimeError("Training ended before a checkpoint could be written.")

    # Reload the checkpoint so evaluation exercises the exact inference artifact.
    final_bundle = joblib.load(task_dir / "model.joblib")
    final_model = final_bundle["model"]
    # Tree construction uses eight workers; fixed-order score aggregation uses
    # one worker so thresholds and floating-point metrics are bitwise repeatable.
    final_model.set_params(n_jobs=1)
    calibration_x, calibration_y, calibration_source = splits["calibration"]
    calibration_scores = predict_scores(final_model, calibration_x)
    threshold_info = threshold_at_bounded_fpr(calibration_scores, calibration_y)
    threshold_info["feature_version"] = FEATURE_VERSION
    threshold_info["model_estimators"] = completed_iteration
    threshold_info["complete"] = True
    atomic_write_json(task_dir / "threshold.json", threshold_info)
    threshold = float(threshold_info["threshold"])

    # Validation splits are evaluated exactly once, after the fixed final stage
    # and calibration-only threshold selection.  They never affect training.
    evaluation = {
        "calibration": metrics_at_threshold(
            calibration_scores,
            calibration_y,
            calibration_source,
            threshold,
        )
    }
    for split_name in ("validation", "validation_augmented"):
        features, labels, source_classes = splits[split_name]
        scores = predict_scores(final_model, features)
        evaluation[split_name] = metrics_at_threshold(
            scores, labels, source_classes, threshold
        )

    feature_importance = np.asarray(final_model.feature_importances_, dtype=np.float64)
    importance_order = np.argsort(-feature_importance, kind="stable")[:100]
    top_feature_importance = [
        {
            "feature": FEATURE_NAMES[int(index)],
            "importance": float(feature_importance[index]),
        }
        for index in importance_order
        if feature_importance[index] > 0
    ]

    metrics = {
        "complete": True,
        "feature_version": FEATURE_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "pixel_only": True,
        "model_family": "ExtraTreesClassifier",
        "scikit_learn_version": sklearn.__version__,
        "classifier_parameters": final_model.get_params(deep=False),
        "training_workers": 8,
        "inference_workers": 1,
        "class_weights": weights_by_class,
        "train": {
            "rows": int(len(train_y)),
            "real_rows": int(np.sum(train_y == 0)),
            "ai_rows": int(np.sum(train_y == 1)),
            "source_class_counts": {
                str(source_class): int(np.sum(train_source == source_class))
                for source_class in sorted(
                    int(value) for value in np.unique(train_source)
                )
            },
        },
        "completed_estimators": completed_iteration,
        "stage_history": stage_history,
        "top_feature_importance": top_feature_importance,
        "threshold_calibration": threshold_info,
        **evaluation,
        "seconds": round(time.time() - start, 2),
        "args": vars(args),
        "prepared_seconds": prepared_summary.get("seconds"),
    }
    atomic_write_json(task_dir / "metrics.json", metrics)
    print(
        json.dumps(
            {
                name: metrics[name]
                for name in ("calibration", "validation", "validation_augmented")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
