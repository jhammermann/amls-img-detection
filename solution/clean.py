"""Explore and deterministically clean the training image parquets.

Outputs are written under ``artifacts/`` because the mounted data directory is
read-only during evaluation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


LABEL_NAMES = {0: "real", 1: "ai_generated"}
SOURCE_CLASS_NAMES = {
    0: "real",
    1: "SD 2.1",
    2: "SDXL",
    3: "SD 3",
    4: "DALL-E 3",
    5: "Midjourney",
}
DEFAULT_IMAGE_SIZE = (128, 128)


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "clean.py needs pyarrow to read parquet files. Install project "
            "dependencies first, for example: pip install pyarrow"
        ) from exc
    return pq


def project_paths() -> tuple[Path, Path]:
    solution_dir = Path(__file__).resolve().parent
    data_dir = solution_dir / "data"
    if not data_dir.exists():
        data_dir = solution_dir / "data-readonly"
    artifacts_dir = solution_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, artifacts_dir


def train_parquet_paths(data_dir: Path) -> list[Path]:
    paths = sorted((data_dir / "train").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No train parquet files found in {data_dir / 'train'}")
    return paths


def iter_parquet_rows(parquet_paths: list[Path], columns: list[str] | None = None):
    pq = require_pyarrow()
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=256, columns=columns):
            batch_dict = batch.to_pydict()
            row_count = len(next(iter(batch_dict.values())))
            for i in range(row_count):
                yield parquet_path.name, {name: values[i] for name, values in batch_dict.items()}


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
        "min": round(ordered[0], 3),
        "p05": round(float(np.percentile(ordered, 5)), 3),
        "p25": round(float(np.percentile(ordered, 25)), 3),
        "p75": round(float(np.percentile(ordered, 75)), 3),
        "p95": round(float(np.percentile(ordered, 95)), 3),
        "max": round(ordered[-1], 3),
    }


def decode_image_info(image_bytes: bytes) -> tuple[int | None, int | None, str | None]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.width, image.height, image.format
    except (UnidentifiedImageError, OSError):
        return None, None, None


def analyze_training_data(parquet_paths: list[Path]) -> dict:
    rows = []
    class_counts = Counter()
    source_class_counts = Counter()
    decode_failures = 0
    by_class = defaultdict(lambda: defaultdict(list))

    for file_name, row in iter_parquet_rows(parquet_paths, columns=["image", "source_class"]):
        image_bytes = row["image"]
        source_class = int(row["source_class"])
        label = 0 if source_class == 0 else 1
        width, height, image_format = decode_image_info(image_bytes)
        byte_length = len(image_bytes)
        aspect_ratio = (width / height) if width and height else None

        class_counts[label] += 1
        source_class_counts[source_class] += 1
        if width is None or height is None:
            decode_failures += 1

        record = {
            "file": file_name,
            "source_class": source_class,
            "source_class_name": SOURCE_CLASS_NAMES.get(source_class, str(source_class)),
            "binary_label": label,
            "class_name": LABEL_NAMES.get(label, str(label)),
            "byte_length": byte_length,
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "format": image_format,
        }
        rows.append(record)

        by_class[label]["byte_length"].append(byte_length)
        if width is not None:
            by_class[label]["width"].append(width)
        if height is not None:
            by_class[label]["height"].append(height)
        if aspect_ratio is not None:
            by_class[label]["aspect_ratio"].append(aspect_ratio)

    return {
        "rows": rows,
        "class_counts": dict(sorted(class_counts.items())),
        "source_class_counts": dict(sorted(source_class_counts.items())),
        "decode_failures": decode_failures,
        "overall": {
            "byte_length": describe([r["byte_length"] for r in rows]),
            "width": describe([r["width"] for r in rows if r["width"] is not None]),
            "height": describe([r["height"] for r in rows if r["height"] is not None]),
            "aspect_ratio": describe([r["aspect_ratio"] for r in rows if r["aspect_ratio"] is not None]),
        },
        "by_class": {
            str(label): {metric: describe(values) for metric, values in metrics.items()}
            for label, metrics in sorted(by_class.items())
        },
    }


def save_rows_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "file",
        "source_class",
        "source_class_name",
        "binary_label",
        "class_name",
        "byte_length",
        "width",
        "height",
        "aspect_ratio",
        "format",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_class_distribution(class_counts: dict[int, int], output_path: Path) -> None:
    labels = sorted(class_counts)
    counts = np.array([class_counts[label] for label in labels], dtype=float)
    percentages = 100.0 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    names = [f"{label}: {LABEL_NAMES.get(label, label)}" for label in labels]
    bars = ax.bar(names, percentages, color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Share of training rows (%)")
    ax.set_title("Class distribution")
    ax.set_ylim(0, max(100, percentages.max() + 5))
    for bar, percentage, count in zip(bars, percentages, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1f}%\n(n={int(count)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_source_distribution(source_class_counts: dict[int, int], output_path: Path) -> None:
    labels = sorted(source_class_counts)
    counts = np.array([source_class_counts[label] for label in labels], dtype=float)
    percentages = 100.0 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f"{label}: {SOURCE_CLASS_NAMES.get(label, label)}" for label in labels]
    bars = ax.bar(names, percentages, color="#72b7b2")
    ax.set_ylabel("Share of training rows (%)")
    ax.set_title("Original source-class distribution")
    ax.tick_params(axis="x", labelrotation=20)
    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_size_distribution(rows: list[dict], output_path: Path) -> None:
    byte_lengths = np.array([row["byte_length"] for row in rows], dtype=float)
    widths = [row["width"] for row in rows if row["width"] is not None]
    heights = [row["height"] for row in rows if row["height"] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(byte_lengths / 1024, bins=40, color="#54a24b")
    axes[0].set_title("Encoded image byte length")
    axes[0].set_xlabel("KiB")
    axes[0].set_ylabel("Images")

    axes[1].scatter(widths, heights, s=10, alpha=0.35, color="#e45756")
    axes[1].set_title("Decoded image dimensions")
    axes[1].set_xlabel("Width (px)")
    axes[1].set_ylabel("Height (px)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def largest_class_gaps(analysis: dict) -> list[str]:
    by_class = analysis["by_class"]
    if "0" not in by_class or "1" not in by_class:
        return ["Only one class was found, so class-separating characteristics cannot be compared."]

    notes = []
    for metric in ["byte_length", "width", "height", "aspect_ratio"]:
        stats_0 = by_class["0"].get(metric, {})
        stats_1 = by_class["1"].get(metric, {})
        if not stats_0 or not stats_1:
            continue
        med_0 = stats_0["median"]
        med_1 = stats_1["median"]
        denominator = max(abs(med_0), abs(med_1), 1)
        relative_gap = abs(med_1 - med_0) / denominator
        notes.append(
            f"- `{metric}` median: class 0 = {med_0}, class 1 = {med_1} "
            f"(relative gap {relative_gap:.1%})."
        )
    return notes


def write_report(analysis: dict, output_path: Path, cleaned_dir: Path, image_size: tuple[int, int]) -> None:
    total = sum(analysis["class_counts"].values())
    class_lines = []
    for label, count in sorted(analysis["class_counts"].items()):
        class_lines.append(
            f"- Class {label} (`{LABEL_NAMES.get(label, label)}`): "
            f"{count} rows ({100 * count / total:.2f}%)"
        )

    source_lines = []
    for label, count in sorted(analysis["source_class_counts"].items()):
        source_lines.append(
            f"- Source {label} (`{SOURCE_CLASS_NAMES.get(label, label)}`): "
            f"{count} rows ({100 * count / total:.2f}%)"
        )

    stats_lines = []
    for metric, stats in analysis["overall"].items():
        stats_lines.append(f"- `{metric}`: {stats}")

    text = f"""# Training Data Exploration and Cleaning

## Class Distribution
The task is binary classification. Source class `0` is label `0`, while source
classes `1..5` are merged into label `1`.

{os.linesep.join(class_lines)}

Original source classes:

{os.linesep.join(source_lines)}

## Image Size and Descriptive Statistics
Decoded image dimensions are reported in pixels. `byte_length` is the encoded
binary size from the parquet file and can reflect both resolution and compression.

{os.linesep.join(stats_lines)}

Decode failures: {analysis["decode_failures"]}

## Possible Class-Leaking Characteristics
These comparisons flag characteristics that could make the class easier to infer
without learning image content. Large gaps should be treated carefully during
modeling and validation.

{os.linesep.join(largest_class_gaps(analysis))}

## Deterministic Cleaning Pipeline
The cleaned dataset is regenerated by this script from the original train
parquets. Each image is decoded with Pillow, EXIF orientation is respected,
converted to RGB, center-cropped to the target aspect ratio, and resized to
{image_size[0]}x{image_size[1]} pixels using bicubic interpolation. Labels are
kept unchanged.

The output format is chunked `.npz` files in `{cleaned_dir}`. Each file contains:

- `images`: `uint8` array shaped `(N, height, width, 3)`
- `labels`: `int8` array shaped `(N,)`, with 0 = real and 1 = ai_generated
- `source_class`: original six-way source class, kept for analysis only
- `source_file`: parquet filename for traceability

This format is simple to load with NumPy, deterministic, CPU-friendly, and keeps
the read-only source data untouched.
"""
    output_path.write_text(text)


def center_crop_to_aspect(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    source_width, source_height = image.size
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        new_width = int(round(source_height * target_ratio))
        left = (source_width - new_width) // 2
        box = (left, 0, left + new_width, source_height)
    else:
        new_height = int(round(source_width / target_ratio))
        top = (source_height - new_height) // 2
        box = (0, top, source_width, top + new_height)

    return image.crop(box)


def clean_image_bytes(image_bytes: bytes, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = center_crop_to_aspect(image, image_size[0], image_size[1])
        image = image.resize(image_size, Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def clean_training_data(
    parquet_paths: list[Path],
    output_dir: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    written_files = []

    for parquet_index, parquet_path in enumerate(parquet_paths):
        images = []
        labels = []
        source_classes = []
        source_files = []
        skipped = 0

        for _, row in iter_parquet_rows([parquet_path], columns=["image", "source_class"]):
            try:
                source_class = int(row["source_class"])
                images.append(clean_image_bytes(row["image"], image_size))
                labels.append(0 if source_class == 0 else 1)
                source_classes.append(source_class)
                source_files.append(parquet_path.name)
            except (UnidentifiedImageError, OSError, ValueError):
                skipped += 1

        if images:
            output_path = output_dir / f"train_cleaned_{parquet_index:03d}.npz"
            np.savez_compressed(
                output_path,
                images=np.stack(images).astype(np.uint8),
                labels=np.asarray(labels, dtype=np.int8),
                source_class=np.asarray(source_classes, dtype=np.int8),
                source_file=np.asarray(source_files),
            )
            written_files.append(output_path.name)

        metadata_rows.append(
            {
                "source_parquet": parquet_path.name,
                "written_npz": written_files[-1] if images else "",
                "rows_written": len(images),
                "rows_skipped": skipped,
            }
        )

    metadata_path = output_dir / "manifest.csv"
    with metadata_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["source_parquet", "written_npz", "rows_written", "rows_skipped"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    return {
        "output_dir": str(output_dir),
        "files": written_files,
        "manifest": str(metadata_path),
        "image_size": image_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--skip_cleaned_dataset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.time()
    image_size = (args.width, args.height)
    data_dir, artifacts_dir = project_paths()
    train_paths = train_parquet_paths(data_dir)

    exploration_dir = artifacts_dir / "clean_exploration"
    exploration_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {len(train_paths)} train parquet files from {data_dir / 'train'}")
    analysis = analyze_training_data(train_paths)
    save_rows_csv(analysis["rows"], exploration_dir / "train_image_stats.csv")
    plot_class_distribution(analysis["class_counts"], exploration_dir / "class_distribution.png")
    plot_source_distribution(analysis["source_class_counts"], exploration_dir / "source_class_distribution.png")
    plot_image_size_distribution(analysis["rows"], exploration_dir / "image_size_distribution.png")

    cleaned_dir = artifacts_dir / "cleaned_train_npz"
    clean_result = {"skipped": True, "output_dir": str(cleaned_dir), "image_size": image_size}
    if not args.skip_cleaned_dataset:
        clean_result = clean_training_data(train_paths, cleaned_dir, image_size=image_size)

    write_report(analysis, exploration_dir / "clean_report.md", cleaned_dir, image_size)
    summary = {
        "class_counts": analysis["class_counts"],
        "source_class_counts": analysis["source_class_counts"],
        "decode_failures": analysis["decode_failures"],
        "overall": analysis["overall"],
        "cleaned_dataset": clean_result,
        "seconds": round(time.time() - start_time, 2),
    }
    (exploration_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
