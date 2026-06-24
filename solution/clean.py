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
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


LABEL_NAMES = {0: "real", 1: "ai_generated"}
DISPLAY_LABEL_NAMES = {0: "Real", 1: "AI-generated"}
SOURCE_CLASS_NAMES = {
    0: "real",
    1: "SD 2.1",
    2: "SDXL",
    3: "SD 3",
    4: "DALL-E 3",
    5: "Midjourney",
}
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_LARGE_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
DEFAULT_SMALL_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
DEFAULT_SIZE_THRESHOLD = DEFAULT_IMAGE_SIZE[0]


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
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "format": image_format,
        }
        rows.append(record)

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


def plot_size_pairs_by_class(rows: list[dict], output_path: Path, limit: int = 6) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    colors = {0: "#4c78a8", 1: "#f58518"}
    for label in sorted(LABEL_NAMES):
        label_rows = [
            row for row in rows
            if row["binary_label"] == label and row["width"] is not None and row["height"] is not None
        ]
        counts = Counter((row["width"], row["height"]) for row in label_rows)
        top_pairs = counts.most_common(limit)
        labels = [f"{width}x{height}" for (width, height), _ in top_pairs]
        values = [100 * count / len(label_rows) for _, count in top_pairs]
        raw_counts = [count for _, count in top_pairs]

        ax = axes[label]
        y_positions = np.arange(len(labels))
        bars = ax.barh(y_positions, values, color=colors[label])
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(f"{DISPLAY_LABEL_NAMES[label]}: common original sizes")
        ax.set_xlabel("Share within class (%)")
        for bar, value, count in zip(bars, values, raw_counts):
            label_text = f"{value:.1f}% (n={count})"
            if value > 70:
                text_x = bar.get_width() - 1.0
                horizontal_alignment = "right"
                text_color = "white"
            else:
                text_x = bar.get_width()
                horizontal_alignment = "left"
                text_color = "black"
            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                label_text if horizontal_alignment == "right" else f" {label_text}",
                va="center",
                ha=horizontal_alignment,
                fontsize=9,
                color=text_color,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_dimensions_by_class(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0: "#4c78a8", 1: "#f58518"}
    markers = {0: "o", 1: "s"}
    for label in sorted(LABEL_NAMES):
        label_rows = [
            row for row in rows
            if row["binary_label"] == label and row["width"] is not None and row["height"] is not None
        ]
        counts = Counter((row["width"], row["height"]) for row in label_rows)
        ax.scatter(
            [width for width, _ in counts],
            [height for _, height in counts],
            s=[max(20, min(500, count / 4)) for count in counts.values()],
            alpha=0.45,
            color=colors[label],
            marker=markers[label],
            label=f"{label}: {LABEL_NAMES[label]}",
        )
    ax.set_title("Unique original dimensions by class")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[label],
            color="none",
            label=f"{label}: {LABEL_NAMES[label]}",
            markerfacecolor=colors[label],
            markeredgecolor=colors[label],
            markersize=8,
            alpha=0.45,
        )
        for label in sorted(LABEL_NAMES)
    ]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


UPSCALE_RISK_THRESHOLDS = (224, 256, 270, 320)


def min_side_threshold_counts(
    rows: list[dict],
    thresholds: tuple[int, ...] = UPSCALE_RISK_THRESHOLDS,
) -> list[dict]:
    counts = []
    for label in sorted(LABEL_NAMES):
        label_rows = [row for row in rows if row["binary_label"] == label]
        rows_with_dimensions = [
            row for row in label_rows
            if row["width"] is not None and row["height"] is not None
        ]
        total = len(label_rows)
        for threshold in thresholds:
            below = sum(min(row["width"], row["height"]) < threshold for row in rows_with_dimensions)
            counts.append(
                {
                    "label": label,
                    "class_name": LABEL_NAMES[label],
                    "threshold": threshold,
                    "below": below,
                    "at_least": total - below,
                    "total": total,
                    "below_percent": 100 * below / total if total else 0,
                }
            )
    return counts


def plot_min_side_thresholds_by_class(rows: list[dict], output_path: Path) -> None:
    counts = min_side_threshold_counts(rows)
    thresholds = sorted({item["threshold"] for item in counts})
    colors = {0: "#4c78a8", 1: "#f58518"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(thresholds))
    width = 0.36
    for ax, title, ylabel, field, formatter in [
        (
            axes[0],
            "Images that would need upscaling",
            "Images (symlog scale)",
            "below",
            lambda value: str(int(value)),
        ),
        (
            axes[1],
            "Share that would need upscaling",
            "Share within class (%)",
            "below_percent",
            lambda value: f"{value:.1f}%",
        ),
    ]:
        for offset, label in [(-width / 2, 0), (width / 2, 1)]:
            values = [
                next(item[field] for item in counts if item["label"] == label and item["threshold"] == threshold)
                for threshold in thresholds
            ]
            bars = ax.bar(x + offset, values, width, color=colors[label], label=LABEL_NAMES[label])
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    formatter(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [f"{threshold}px target" for threshold in thresholds], rotation=15)
        ax.legend()
    axes[0].set_yscale("symlog", linthresh=10)
    max_absolute = max(item["below"] for item in counts)
    axes[0].set_ylim(0, max_absolute * 1.6 if max_absolute else 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def update_reference_example(examples: dict[str, dict | None], record: dict) -> None:
    width = record["width"]
    height = record["height"]
    crop_side = record["crop_side"]
    label = record["label"]

    if label == 0:
        largest = examples["largest_real"]
        smallest = examples["smallest_real"]
        if largest is None or crop_side > largest["crop_side"]:
            examples["largest_real"] = record
        if smallest is None or crop_side < smallest["crop_side"]:
            examples["smallest_real"] = record
    elif width == 320 and height == 320 and examples["ai_320"] is None:
        examples["ai_320"] = record
    elif width == 270 and height == 270 and examples["ai_270"] is None:
        examples["ai_270"] = record


def find_reference_examples_from_npz(cleaned_dir: Path) -> dict[str, dict | None] | None:
    npz_paths = sorted(cleaned_dir.glob("train_cleaned_*.npz"))
    if not npz_paths:
        print(f"Warning: no cleaned NPZ files found in {cleaned_dir}; skipping reference example plot.")
        return None

    examples: dict[str, dict | None] = {
        "largest_real": None,
        "smallest_real": None,
        "ai_320": None,
        "ai_270": None,
    }

    for npz_path in npz_paths:
        with np.load(npz_path) as data:
            images = data["images"]
            labels = data["labels"]
            original_widths = data["original_width"]
            original_heights = data["original_height"]
            for i in range(len(labels)):
                width = int(original_widths[i])
                height = int(original_heights[i])
                record = {
                    "image": np.asarray(images[i]).copy(),
                    "file": npz_path.name,
                    "label": int(labels[i]),
                    "width": width,
                    "height": height,
                    "crop_side": min(width, height),
                }
                update_reference_example(examples, record)

        if all(examples.values()) and examples["largest_real"]["crop_side"] == 640:
            break

    return examples


def plot_cropped_reference_examples(
    cleaned_dir: Path,
    output_path: Path,
) -> None:
    examples = find_reference_examples_from_npz(cleaned_dir)
    if examples is None:
        return

    titles = [
        ("largest_real", "Largest real after crop"),
        ("smallest_real", "Smallest real after crop/resize"),
        ("ai_320", "AI 320px original after crop"),
        ("ai_270", "AI 270px original after crop"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (key, title) in zip(axes, titles):
        example = examples[key]
        ax.axis("off")
        if example is None:
            ax.text(0.5, 0.5, "No matching image", ha="center", va="center", wrap=True)
            ax.set_title(title, fontsize=10)
            continue

        image = Image.fromarray(example["image"])
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.text(
            0.5,
            -0.08,
            f"{example['width']}x{example['height']} -> {image.width}x{image.height}",
            transform=ax.transAxes,
            ha="center",
            va="top",
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
    if small_size == large_size:
        return small_size
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


def class_size_signal_lines(analysis: dict) -> list[str]:
    by_class = analysis["by_class"]
    if "0" not in by_class or "1" not in by_class:
        return ["- Only one class was found, so class-separating characteristics cannot be compared."]

    notes = []
    for metric in ["width", "height", "aspect_ratio"]:
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


def compact_stats(stats: dict[str, float]) -> str:
    return (
        f"median {stats['median']}, p05 {stats['p05']}, "
        f"p95 {stats['p95']}, range {stats['min']}..{stats['max']}"
    )


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

    stats_lines = [
        f"- Width: {compact_stats(analysis['overall']['width'])}",
        f"- Height: {compact_stats(analysis['overall']['height'])}",
        f"- Aspect ratio: {compact_stats(analysis['overall']['aspect_ratio'])}",
    ]
    threshold_lines = [
        f"- {item['class_name']} images with minimum side below {item['threshold']}px: "
        f"{item['below']} / {item['total']} ({item['below_percent']:.2f}%)"
        for item in min_side_threshold_counts(analysis["rows"])
    ]
    size_counts = target_size_counts(
        analysis["rows"],
        small_size=small_size,
        large_size=large_size,
        size_threshold=size_threshold,
    )
    target_sizes = sorted(set(size_counts[0]) | set(size_counts[1]))
    size_lines = []
    for label, label_name in [(0, "Real"), (1, "AI")]:
        counts_text = ", ".join(
            f"{target_size}px = {size_counts[label][target_size]}"
            for target_size in target_sizes
        )
        size_lines.append(f"- {label_name} target sizes: {counts_text}")
    if small_size != large_size:
        small_keep_limit = ai_small_keep_limit(
            analysis["rows"],
            small_size=small_size,
            large_size=large_size,
            size_threshold=size_threshold,
        )
        size_lines.append(f"- AI small-size balancing enabled in cleaning: {balance_ai_small}")
        if balance_ai_small and small_keep_limit is not None:
            size_lines.append(f"- AI small-size balancing limit: keep {small_keep_limit}")

    if small_size == large_size:
        cleaning_size_lines = [
            f"- Target size: {small_size[0]}x{small_size[1]}",
        ]
    else:
        cleaning_size_lines = [
            f"- Small target size: {small_size[0]}x{small_size[1]}",
            f"- Large target size: {large_size[0]}x{large_size[1]}",
            f"- Large-size threshold: both original dimensions >= {size_threshold}px",
        ]

    text = f"""# Cleaning Summary

## Dataset
{os.linesep.join(class_lines)}
- Decode failures: {analysis["decode_failures"]}

## Image Size Stats
{os.linesep.join(stats_lines)}

## Size By Class
{os.linesep.join(class_size_signal_lines(analysis))}
{os.linesep.join(threshold_lines)}

## Cleaning Settings
- Crop: deterministic center square crop
- Resize interpolation: bicubic
{os.linesep.join(cleaning_size_lines)}

## Cleaned-Size Buckets
{os.linesep.join(size_lines)}
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
    for stale_path in output_dir.glob("train_cleaned_*.npz"):
        stale_path.unlink()
    (output_dir / "manifest.csv").unlink(missing_ok=True)

    metadata_rows = []
    written_files = []
    small_key = small_size[0]
    ai_small_limit = None
    if balance_ai_small and small_size != large_size:
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
            },
            large_size[0]: {
                "images": [],
                "labels": [],
                "source_classes": [],
                "source_files": [],
                "target_sizes": [],
                "original_widths": [],
                "original_heights": [],
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
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--size_threshold", type=int, default=DEFAULT_SIZE_THRESHOLD)
    parser.add_argument("--balance_ai_small", action="store_true")
    parser.add_argument("--skip_cleaned_dataset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.time()
    small_size = (args.image_size, args.image_size)
    large_size = (args.image_size, args.image_size)
    data_dir, artifacts_dir = project_paths()
    train_paths = train_parquet_paths(data_dir)

    exploration_dir = artifacts_dir / "clean_exploration"
    exploration_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = artifacts_dir / "cleaned_train_npz"

    print(f"Reading {len(train_paths)} train parquet files from {data_dir / 'train'}")
    analysis = analyze_training_data(train_paths)
    save_rows_csv(analysis["rows"], exploration_dir / "train_image_stats.csv")
    for obsolete_plot in [
        "top_dimension_values.png",
        "top_dimension_values_by_class.png",
    ]:
        (exploration_dir / obsolete_plot).unlink(missing_ok=True)
    plot_class_distribution(analysis["class_counts"], exploration_dir / "class_distribution.png")
    plot_image_dimensions_by_class(analysis["rows"], exploration_dir / "image_dimensions_by_class.png")
    plot_size_pairs_by_class(analysis["rows"], exploration_dir / "size_pairs_by_class.png")
    plot_min_side_thresholds_by_class(analysis["rows"], exploration_dir / "min_side_thresholds_by_class.png")
    plot_cropped_reference_examples(
        cleaned_dir,
        exploration_dir / "cropped_reference_examples.png",
    )

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
