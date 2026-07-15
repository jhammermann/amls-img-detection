"""Task 3 inference using the shared engineered-feature pipeline."""

from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np

from clean import DEFAULT_IMAGE_SIZE, clean_image_bytes
from predict_features import (
    parse_args,
    parquet_rows,
    paths,
    score_batch,
    validate_bundle,
    validate_schema,
)


def main() -> int:
    args = parse_args()
    import joblib

    started = time.time()
    deadline = started + max(1, args.timeout_seconds - 5)
    data_dir, artifacts = paths()
    model_dir = artifacts / "task03_features"
    bundle = joblib.load(model_dir / "model.joblib")
    validate_bundle(bundle)
    validate_schema(json.loads((model_dir / "schema.json").read_text()))
    threshold_info = json.loads((model_dir / "threshold.json").read_text())
    if not threshold_info.get("complete"):
        raise RuntimeError("Task 3 threshold calibration is incomplete.")
    if threshold_info.get("model_estimators") != bundle.get("iteration"):
        raise RuntimeError("Task 3 model and threshold do not belong together.")
    model = bundle["model"]
    model.set_params(n_jobs=1)
    quality_edges = np.asarray(threshold_info["quality_edges"], dtype=np.float64)
    quality_thresholds = np.asarray(threshold_info["quality_thresholds"], dtype=np.float64)

    predictions, batch_ids, batch_images = [], [], []

    def flush() -> None:
        if not batch_images:
            return
        scores = score_batch(model, batch_images, deadline)
        contrasts = []
        for image_bytes in batch_images:
            image = clean_image_bytes(image_bytes, DEFAULT_IMAGE_SIZE).astype(np.float32)
            luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            contrasts.append(float(luminance.std()))
        bins = np.digitize(contrasts, quality_edges[1:-1])
        thresholds = quality_thresholds[bins]
        predictions.extend(
            (row_id, int(score >= threshold))
            for row_id, score, threshold in zip(batch_ids, scores, thresholds)
        )
        batch_ids.clear()
        batch_images.clear()

    for row_id, image_bytes in parquet_rows(data_dir / "predict"):
        if time.time() >= deadline:
            raise TimeoutError(f"Prediction timed out after {len(predictions)} rows")
        batch_ids.append(row_id)
        batch_images.append(image_bytes)
        if len(batch_images) >= args.batch_size:
            flush()
    flush()

    ordered = sorted(predictions)
    if len(ordered) != len({row_id for row_id, _ in ordered}):
        raise RuntimeError("Prediction input contains duplicate row IDs.")
    output_dir = artifacts / "task03"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "predictions.csv"
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["row_id", "predicted_label"])
        writer.writerows(ordered)
    temporary.replace(output)
    print(f"wrote {len(ordered)} predictions to {output} in {time.time() - started:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
