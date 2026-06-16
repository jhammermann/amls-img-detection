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
DEFAULT_LARGE_IMAGE_SIZE = (320, 320)
DEFAULT_SMALL_IMAGE_SIZE = (270, 270)
DEFAULT_SIZE_THRESHOLD = 320


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


def plot_image_dimensions_by_class(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0: "#4c78a8", 1: "#f58518"}
    for label in sorted(LABEL_NAMES):
        label_rows = [
            row for row in rows
            if row["binary_label"] == label and row["width"] is not None and row["height"] is not None
        ]
        ax.scatter(
            [row["width"] for row in label_rows],
            [row["height"] for row in label_rows],
            s=10,
            alpha=0.35,
            color=colors[label],
            label=f"{label}: {LABEL_NAMES[label]}",
        )
    ax.set_title("Decoded image dimensions by binary class")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def top_value_counts(rows: list[dict], field: str, limit: int = 3) -> list[tuple[int, int]]:
    counts = Counter(row[field] for row in rows if row[field] is not None)
    return counts.most_common(limit)


def plot_top_dimensions(rows: list[dict], output_path: Path) -> None:
    top_widths = top_value_counts(rows, "width")
    top_heights = top_value_counts(rows, "height")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, values in [
        (axes[0], "Top 3 widths", top_widths),
        (axes[1], "Top 3 heights", top_heights),
    ]:
        labels = [f"{value}px" for value, _ in values]
        counts = [count for _, count in values]
        bars = ax.bar(labels, counts, color="#72b7b2")
        ax.set_title(title)
        ax.set_ylabel("Images")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_top_dimensions_by_class(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fields = [("width", "Widths"), ("height", "Heights")]
    colors = {0: "#4c78a8", 1: "#f58518"}

    for row_index, label in enumerate(sorted(LABEL_NAMES)):
        label_rows = [row for row in rows if row["binary_label"] == label]
        for col_index, (field, title) in enumerate(fields):
            values = top_value_counts(label_rows, field)
            ax = axes[row_index][col_index]
            labels = [f"{value}px" for value, _ in values]
            counts = [count for _, count in values]
            bars = ax.bar(labels, counts, color=colors[label])
            ax.set_title(f"{LABEL_NAMES[label]}: top 3 {title.lower()}")
            ax.set_ylabel("Images")
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def real_min_side_counts(rows: list[dict], thresholds: tuple[int, ...] = (270, 320)) -> list[dict]:
    real_rows = [
        row for row in rows
        if row["binary_label"] == 0 and row["width"] is not None and row["height"] is not None
    ]
    total = len(real_rows)
    counts = []
    for threshold in thresholds:
        below = sum(min(row["width"], row["height"]) < threshold for row in real_rows)
        counts.append(
            {
                "threshold": threshold,
                "below": below,
                "at_least": total - below,
                "total": total,
                "below_percent": 100 * below / total if total else 0,
            }
        )
    return counts


def plot_real_min_side_thresholds(rows: list[dict], output_path: Path) -> None:
    counts = real_min_side_counts(rows)
    labels = [f"< {item['threshold']}px\nmin side" for item in counts]
    values = [item["below"] for item in counts]
    percentages = [item["below_percent"] for item in counts]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color="#4c78a8")
    ax.set_title("Real images smaller than AI dimension thresholds")
    ax.set_ylabel("Real images")
    for bar, value, percentage in zip(bars, values, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def choose_clean_size(
    width: int,
    height: int,
    small_size: tuple[int, int] = DEFAULT_SMALL_IMAGE_SIZE,
    large_size: tuple[int, int] = DEFAULT_LARGE_IMAGE_SIZE,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> tuple[int, int]:
    if width >= size_threshold and height >= size_threshold:
        return large_size
    return small_size


def target_size_counts(
    rows: list[dict],
    small_size: tuple[int, int] = DEFAULT_SMALL_IMAGE_SIZE,
    large_size: tuple[int, int] = DEFAULT_LARGE_IMAGE_SIZE,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> dict[int, Counter]:
    counts = {0: Counter(), 1: Counter()}
    for row in rows:
        if row["width"] is None or row["height"] is None:
            continue
        target_size = choose_clean_size(
            row["width"],
            row["height"],
            small_size=small_size,
            large_size=large_size,
            size_threshold=size_threshold,
        )
        counts[row["binary_label"]][target_size[0]] += 1
    return counts


def ai_small_keep_limit(
    rows: list[dict],
    small_size: tuple[int, int] = DEFAULT_SMALL_IMAGE_SIZE,
    large_size: tuple[int, int] = DEFAULT_LARGE_IMAGE_SIZE,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
) -> int | None:
    counts = target_size_counts(
        rows,
        small_size=small_size,
        large_size=large_size,
        size_threshold=size_threshold,
    )
    small_key = small_size[0]
    large_key = large_size[0]
    real_small = counts[0][small_key]
    real_large = counts[0][large_key]
    ai_small = counts[1][small_key]
    ai_large = counts[1][large_key]

    if ai_small == 0:
        return None
    if real_small == 0:
        return 0
    if real_large == 0:
        return None

    real_small_fraction = real_small / (real_small + real_large)
    ai_small_fraction = ai_small / (ai_small + ai_large)
    if ai_small_fraction <= real_small_fraction:
        return None

    return min(ai_small, int(ai_large * real_small / real_large))


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


def write_report(
    analysis: dict,
    output_path: Path,
    cleaned_dir: Path,
    small_size: tuple[int, int],
    large_size: tuple[int, int],
    size_threshold: int,
    balance_ai_small: bool,
) -> None:
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

    top_width_lines = [
        f"- {value}px: {count} images"
        for value, count in top_value_counts(analysis["rows"], "width")
    ]
    top_height_lines = [
        f"- {value}px: {count} images"
        for value, count in top_value_counts(analysis["rows"], "height")
    ]
    top_by_class_lines = []
    for label in sorted(LABEL_NAMES):
        label_rows = [row for row in analysis["rows"] if row["binary_label"] == label]
        widths = ", ".join(f"{value}px ({count})" for value, count in top_value_counts(label_rows, "width"))
        heights = ", ".join(f"{value}px ({count})" for value, count in top_value_counts(label_rows, "height"))
        top_by_class_lines.append(f"- {LABEL_NAMES[label]} widths: {widths}")
        top_by_class_lines.append(f"- {LABEL_NAMES[label]} heights: {heights}")
    threshold_lines = [
        f"- Real images with minimum side below {item['threshold']}px: "
        f"{item['below']} / {item['total']} ({item['below_percent']:.2f}%)"
        for item in real_min_side_counts(analysis["rows"])
    ]
    size_counts = target_size_counts(analysis["rows"])
    small_keep_limit = ai_small_keep_limit(analysis["rows"])
    size_lines = [
        f"- Real target sizes: 270px = {size_counts[0][270]}, 320px = {size_counts[0][320]}",
        f"- AI target sizes before balancing: 270px = {size_counts[1][270]}, 320px = {size_counts[1][320]}",
    ]
    if small_keep_limit is None:
        size_lines.append("- AI 270px balancing limit: not applied")
    else:
        size_lines.append(f"- AI 270px balancing limit: keep {small_keep_limit}")
    size_lines.append(f"- AI 270px balancing enabled in cleaning: {balance_ai_small}")

    plot_lines = [
        "- `class_distribution.png`: Binary class distribution.",
        "- `source_class_distribution.png`: Original six-way source-class distribution.",
        "- `image_size_distribution.png`: Encoded byte length and decoded width/height scatter.",
        "- `image_dimensions_by_class.png`: Decoded dimensions colored by binary class.",
        "- `top_dimension_values.png`: Three most common widths and heights.",
        "- `top_dimension_values_by_class.png`: Three most common widths and heights per class.",
        "- `real_min_side_thresholds.png`: Real images below 270px and 320px minimum side.",
    ]

    text = f"""# Cleaning Statistics

## Class Counts
{os.linesep.join(class_lines)}

## Source-Class Counts
{os.linesep.join(source_lines)}

## Descriptive Statistics
{os.linesep.join(stats_lines)}

## Frequent Dimensions
Widths:
{os.linesep.join(top_width_lines)}

Heights:
{os.linesep.join(top_height_lines)}

By binary class:
{os.linesep.join(top_by_class_lines)}

## Real Images Below Thresholds
{os.linesep.join(threshold_lines)}

## Cleaned-Size Buckets
{os.linesep.join(size_lines)}

## Class-Correlated Size Statistics
{os.linesep.join(largest_class_gaps(analysis))}

## Cleaning Parameters
- Small target size: {small_size[0]}x{small_size[1]}
- Large target size: {large_size[0]}x{large_size[1]}
- Large-size threshold: both original dimensions >= {size_threshold}px
- Crop: deterministic center square crop
- Resize interpolation: bicubic
- Decode failures: {analysis["decode_failures"]}
- Output folder: `{cleaned_dir}`
- Output format: `.npz`

## NPZ Fields
- `images`: uint8 array `(N, H, W, 3)`
- `labels`: int8 binary label, 0 = real, 1 = ai_generated
- `source_class`: int8 original source class
- `source_file`: source parquet filename
- `target_size`: int16 cleaned square size
- `original_width`: int32 decoded source width
- `original_height`: int32 decoded source height
- `original_byte_length`: int32 encoded source byte length

## Plot Captions
{os.linesep.join(plot_lines)}
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
    analysis: dict,
    small_size: tuple[int, int] = DEFAULT_SMALL_IMAGE_SIZE,
    large_size: tuple[int, int] = DEFAULT_LARGE_IMAGE_SIZE,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
    balance_ai_small: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    written_files = []
    small_key = small_size[0]
    ai_small_limit = None
    if balance_ai_small:
        ai_small_limit = ai_small_keep_limit(
            analysis["rows"],
            small_size=small_size,
            large_size=large_size,
            size_threshold=size_threshold,
        )
    ai_small_seen = 0
    ai_small_skipped = 0

    for parquet_index, parquet_path in enumerate(parquet_paths):
        chunks = {
            small_key: {
                "images": [],
                "labels": [],
                "source_classes": [],
                "source_files": [],
                "target_sizes": [],
                "original_widths": [],
                "original_heights": [],
                "original_byte_lengths": [],
            },
            large_size[0]: {
                "images": [],
                "labels": [],
                "source_classes": [],
                "source_files": [],
                "target_sizes": [],
                "original_widths": [],
                "original_heights": [],
                "original_byte_lengths": [],
            },
        }
        skipped = 0

        for _, row in iter_parquet_rows([parquet_path], columns=["image", "source_class"]):
            try:
                source_class = int(row["source_class"])
                label = 0 if source_class == 0 else 1
                image_bytes = row["image"]
                width, height, _ = decode_image_info(image_bytes)
                if width is None or height is None:
                    skipped += 1
                    continue

                image_size = choose_clean_size(
                    width,
                    height,
                    small_size=small_size,
                    large_size=large_size,
                    size_threshold=size_threshold,
                )
                target_size = image_size[0]
                if label == 1 and target_size == small_key and ai_small_limit is not None:
                    ai_small_seen += 1
                    if ai_small_seen > ai_small_limit:
                        ai_small_skipped += 1
                        skipped += 1
                        continue

                chunk = chunks[target_size]
                chunk["images"].append(clean_image_bytes(image_bytes, image_size))
                chunk["labels"].append(label)
                chunk["source_classes"].append(source_class)
                chunk["source_files"].append(parquet_path.name)
                chunk["target_sizes"].append(target_size)
                chunk["original_widths"].append(width)
                chunk["original_heights"].append(height)
                chunk["original_byte_lengths"].append(len(image_bytes))
            except (UnidentifiedImageError, OSError, ValueError):
                skipped += 1

        parquet_written_files = []
        rows_written = 0
        for target_size, chunk in sorted(chunks.items()):
            if not chunk["images"]:
                continue
            output_path = output_dir / f"train_cleaned_{parquet_index:03d}_{target_size}.npz"
            np.savez(
                output_path,
                images=np.stack(chunk["images"]).astype(np.uint8),
                labels=np.asarray(chunk["labels"], dtype=np.int8),
                source_class=np.asarray(chunk["source_classes"], dtype=np.int8),
                source_file=np.asarray(chunk["source_files"]),
                target_size=np.asarray(chunk["target_sizes"], dtype=np.int16),
                original_width=np.asarray(chunk["original_widths"], dtype=np.int32),
                original_height=np.asarray(chunk["original_heights"], dtype=np.int32),
                original_byte_length=np.asarray(chunk["original_byte_lengths"], dtype=np.int32),
            )
            written_files.append(output_path.name)
            parquet_written_files.append(output_path.name)
            rows_written += len(chunk["images"])

        metadata_rows.append(
            {
                "source_parquet": parquet_path.name,
                "written_npz": "|".join(parquet_written_files),
                "rows_written": rows_written,
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
        "small_size": small_size,
        "large_size": large_size,
        "size_threshold": size_threshold,
        "balance_ai_small": balance_ai_small,
        "ai_small_keep_limit": ai_small_limit,
        "ai_small_skipped": ai_small_skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--small_size", type=int, default=DEFAULT_SMALL_IMAGE_SIZE[0])
    parser.add_argument("--large_size", type=int, default=DEFAULT_LARGE_IMAGE_SIZE[0])
    parser.add_argument("--size_threshold", type=int, default=DEFAULT_SIZE_THRESHOLD)
    parser.add_argument("--balance_ai_small", action="store_true")
    parser.add_argument("--skip_cleaned_dataset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.time()
    small_size = (args.small_size, args.small_size)
    large_size = (args.large_size, args.large_size)
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
    plot_image_dimensions_by_class(analysis["rows"], exploration_dir / "image_dimensions_by_class.png")
    plot_top_dimensions(analysis["rows"], exploration_dir / "top_dimension_values.png")
    plot_top_dimensions_by_class(analysis["rows"], exploration_dir / "top_dimension_values_by_class.png")
    plot_real_min_side_thresholds(analysis["rows"], exploration_dir / "real_min_side_thresholds.png")

    cleaned_dir = artifacts_dir / "cleaned_train_npz"
    clean_result = {
        "skipped": True,
        "output_dir": str(cleaned_dir),
        "small_size": small_size,
        "large_size": large_size,
        "size_threshold": args.size_threshold,
        "balance_ai_small": args.balance_ai_small,
    }
    if not args.skip_cleaned_dataset:
        clean_result = clean_training_data(
            train_paths,
            cleaned_dir,
            analysis=analysis,
            small_size=small_size,
            large_size=large_size,
            size_threshold=args.size_threshold,
            balance_ai_small=args.balance_ai_small,
        )

    write_report(
        analysis,
        exploration_dir / "clean_report.md",
        cleaned_dir,
        small_size=small_size,
        large_size=large_size,
        size_threshold=args.size_threshold,
        balance_ai_small=args.balance_ai_small,
    )
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
