from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from clean import DEFAULT_IMAGE_SIZE, clean_image_bytes


def paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    data = root / "data"
    if not data.exists():
        data = root / "data-readonly"
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return data, artifacts


def build_model(k: int = 32) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(k, 2 * k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4 * k, 2),
    )


def normalize_batch(images: np.ndarray, device: torch.device, image_size: int) -> torch.Tensor:
    import torch.nn.functional as F

    x = torch.as_tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return (x - 0.5) / 0.5


def parquet_rows(split_dir: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("predict.py needs pyarrow from requirements.txt.") from exc
    for path in sorted(split_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["row_id", "image"]):
            data = batch.to_pydict()
            for row_id, image in zip(data["row_id"], data["image"]):
                yield int(row_id), image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")
    data_dir, artifacts_dir = paths()
    task_dir = artifacts_dir / "task02"
    checkpoint = torch.load(task_dir / "model.pt", map_location=device, weights_only=True)
    threshold = json.loads((task_dir / "threshold.json").read_text())["threshold"]
    image_size = int(checkpoint.get("image_size", 256))
    model = build_model(int(checkpoint.get("channels", 32))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictions, batch_ids, batch_images = [], [], []
    for row_id, image_bytes in parquet_rows(data_dir / "predict"):
        batch_ids.append(row_id)
        batch_images.append(clean_image_bytes(image_bytes, DEFAULT_IMAGE_SIZE))
        if len(batch_images) == args.batch_size:
            x = normalize_batch(np.stack(batch_images), device, image_size)
            scores = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            predictions.extend(zip(batch_ids, (scores >= threshold).astype(np.int8)))
            batch_ids, batch_images = [], []
    if batch_images:
        x = normalize_batch(np.stack(batch_images), device, image_size)
        scores = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
        predictions.extend(zip(batch_ids, (scores >= threshold).astype(np.int8)))

    out_dir = task_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["row_id", "predicted_label"])
        writer.writerows(sorted((int(row_id), int(label)) for row_id, label in predictions))
    print(f"wrote {len(predictions)} predictions to {out_dir / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
