"""Run Task 2 inference with the engineered-feature model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "8"

_CACHE_ROOT = Path(__file__).resolve().parent / "artifacts" / "cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE_ROOT / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(_CACHE_ROOT)

import numpy as np

from clean import DEFAULT_IMAGE_SIZE, clean_image_bytes
from prepare import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features,
)


def paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    data = root / "data"
    if not data.exists():
        data = root / "data-readonly"
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return data, artifacts


def parquet_rows(split_dir: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("predict_features.py needs pyarrow from requirements.txt.") from exc

    parquet_paths = sorted(split_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No prediction parquet files found in {split_dir}")
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256, columns=["row_id", "image"]):
            data = batch.to_pydict()
            for row_id, image in zip(data["row_id"], data["image"]):
                yield int(row_id), image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def validate_bundle(bundle: dict) -> None:
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"Model feature version {bundle.get('feature_version')!r} does not match {FEATURE_VERSION!r}."
        )
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("Model feature names do not match the current extractor.")
    if bundle.get("pixel_only") is not True:
        raise RuntimeError("Refusing a model that is not marked as pixel-only.")
    if bundle.get("model_type") != "sklearn_extra_trees":
        raise RuntimeError(f"Unsupported feature model type {bundle.get('model_type')!r}.")
    if bundle.get("model") is None:
        raise RuntimeError("Feature model bundle does not contain a fitted classifier.")


def validate_schema(schema: dict) -> None:
    if schema.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError("Saved feature schema version does not match the extractor.")
    if schema.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("Saved feature schema names do not match the extractor.")
    if schema.get("pixel_only") is not True:
        raise RuntimeError("Saved feature schema is not marked as pixel-only.")


def score_batch(model, image_bytes_batch: list[bytes], deadline: float) -> np.ndarray:
    feature_rows = []
    for image_bytes in image_bytes_batch:
        if time.time() >= deadline:
            raise TimeoutError("Prediction timed out during image feature extraction.")
        feature_rows.append(
            extract_features(clean_image_bytes(image_bytes, DEFAULT_IMAGE_SIZE))
        )
    features = np.stack(feature_rows)
    return np.asarray(
        model.predict_proba(features)[:, 1],
        dtype=np.float64,
    )


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0 or args.batch_size <= 0:
        raise SystemExit("timeout and batch size must be positive")
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit("predict_features.py needs scikit-learn from requirements.txt.") from exc

    start = time.time()
    deadline = start + max(1, args.timeout_seconds - 5)
    data_dir, artifacts_dir = paths()
    feature_task_dir = artifacts_dir / "task02_features"
    bundle = joblib.load(feature_task_dir / "model.joblib")
    validate_bundle(bundle)
    validate_schema(json.loads((feature_task_dir / "schema.json").read_text()))
    model = bundle["model"]
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    threshold_payload = json.loads((feature_task_dir / "threshold.json").read_text())
    if threshold_payload.get("complete") is not True:
        raise RuntimeError("Training did not complete threshold calibration.")
    if threshold_payload.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError("Saved threshold feature version does not match the extractor.")
    if threshold_payload.get("model_estimators") != bundle.get("iteration"):
        raise RuntimeError("Saved threshold does not belong to the current model checkpoint.")
    threshold = float(threshold_payload["threshold"])

    predictions: list[tuple[int, int]] = []
    batch_ids: list[int] = []
    batch_images: list[bytes] = []

    def flush_batch() -> None:
        if not batch_images:
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Prediction timed out after {len(predictions)} completed rows")
        scores = score_batch(model, batch_images, deadline)
        predictions.extend(
            (row_id, int(score >= threshold)) for row_id, score in zip(batch_ids, scores)
        )
        batch_ids.clear()
        batch_images.clear()

    for row_id, image_bytes in parquet_rows(data_dir / "predict"):
        batch_ids.append(row_id)
        batch_images.append(image_bytes)
        if len(batch_images) >= args.batch_size:
            flush_batch()
            print(f"\rpredicted {len(predictions)} rows", end="", flush=True)
    flush_batch()
    print()

    ordered = sorted(predictions)
    row_ids = [row_id for row_id, _ in ordered]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("Prediction input contains duplicate row_id values.")

    output_dir = artifacts_dir / "task02"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.csv"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["row_id", "predicted_label"])
        writer.writerows(ordered)
    temporary.replace(output_path)
    print(f"wrote {len(ordered)} predictions to {output_path} in {time.time() - start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
