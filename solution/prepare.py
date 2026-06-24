from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from clean import DEFAULT_IMAGE_SIZE, clean_image_bytes


def paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    data = root / "data"
    if not data.exists():
        data = root / "data-readonly"
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return data, artifacts


def parquet_rows(split_dir: Path, columns: list[str]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("prepare.py needs pyarrow from requirements.txt.") from exc
    for path in sorted(split_dir.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            data = batch.to_pydict()
            for i in range(len(next(iter(data.values())))):
                yield {name: values[i] for name, values in data.items()}


def prepare_split(data_dir: Path, out_dir: Path, split: str) -> dict:
    images, labels, source_classes = [], [], []
    for row in parquet_rows(data_dir / split, ["image", "source_class"]):
        source_class = int(row["source_class"])
        images.append(clean_image_bytes(row["image"], DEFAULT_IMAGE_SIZE))
        labels.append(0 if source_class == 0 else 1)
        source_classes.append(source_class)
    output = out_dir / f"{split}.npz"
    np.savez_compressed(
        output,
        images=np.stack(images).astype(np.uint8),
        labels=np.asarray(labels, dtype=np.int8),
        source_class=np.asarray(source_classes, dtype=np.int8),
    )
    return {"split": split, "rows": len(labels), "output": str(output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    data_dir, artifacts_dir = paths()
    out_dir = artifacts_dir / "prepared" / "task02"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "image_size": list(DEFAULT_IMAGE_SIZE),
        "splits": [
            prepare_split(data_dir, out_dir, "calibration"),
            prepare_split(data_dir, out_dir, "validation"),
            prepare_split(data_dir, out_dir, "validation_augmented"),
        ],
        "seconds": round(time.time() - start, 2),
        "timeout_seconds": args.timeout_seconds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
