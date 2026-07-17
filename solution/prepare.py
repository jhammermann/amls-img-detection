"""Prepare deterministic, pixel-only engineered features for Task 2.

The training images come from ``clean.py`` outputs.  Calibration and
validation images are cleaned with the same function before feature
extraction.  This script deliberately never reads ``data/predict``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Feature extraction parallelizes images explicitly.  Keep each numerical
# operation single-threaded so eight image workers cannot oversubscribe CPUs.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

# ``clean.py`` imports matplotlib.  Keep its config and any library caches in
# the only writable project area permitted by the exercise.
_CACHE_ROOT = Path(__file__).resolve().parent / "artifacts" / "cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE_ROOT / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(_CACHE_ROOT)

import numpy as np
from clean import DEFAULT_IMAGE_SIZE, clean_image_bytes


FEATURE_VERSION = "patch_forensics_v3"
PATCH_SIZE = 32
PATCH_GRID_SIZE = DEFAULT_IMAGE_SIZE[0] // PATCH_SIZE
PATCH_COUNT = PATCH_GRID_SIZE * PATCH_GRID_SIZE
PATCH_POOL_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
MAX_FEATURE_WORKERS = 8
FEATURE_FAMILY_PREFIXES = (
    "patch_all_",
    "texture_contrast_",
    "lowbit_",
    "compression_",
)

_WORKER_IMAGES: np.ndarray | None = None


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

    parquet_paths = sorted(split_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {split_dir}")
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            values = batch.to_pydict()
            for row_index in range(len(next(iter(values.values())))):
                yield {name: column[row_index] for name, column in values.items()}


def _patch_descriptor_matrix(image: np.ndarray) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Describe every non-overlapping native 32x32 patch.

    Reshaping covers the complete cleaned 256x256 image: all 64 patches and
    every pixel contribute to the pooled descriptors.  These independently
    implemented descriptors are inspired by published four-direction texture
    contrast and low-bit-plane methods.
    """

    patches_u8 = (
        image.reshape(PATCH_GRID_SIZE, PATCH_SIZE, PATCH_GRID_SIZE, PATCH_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(PATCH_COUNT, PATCH_SIZE, PATCH_SIZE, 3)
    )
    patches = patches_u8.astype(np.float32) / 255.0
    gray = 0.299 * patches[..., 0] + 0.587 * patches[..., 1] + 0.114 * patches[..., 2]

    horizontal = np.abs(np.diff(patches, axis=2)).mean(axis=(1, 2, 3))
    vertical = np.abs(np.diff(patches, axis=1)).mean(axis=(1, 2, 3))
    diagonal = np.abs(patches[:, 1:, 1:] - patches[:, :-1, :-1]).mean(axis=(1, 2, 3))
    anti_diagonal = np.abs(patches[:, 1:, :-1] - patches[:, :-1, 1:]).mean(axis=(1, 2, 3))
    diversity = horizontal + vertical + diagonal + anti_diagonal

    cross = np.abs(gray[:, :-1, :-1] + gray[:, 1:, 1:] - gray[:, :-1, 1:] - gray[:, 1:, :-1])
    laplacian = np.abs(
        -4.0 * gray[:, 1:-1, 1:-1]
        + gray[:, :-2, 1:-1]
        + gray[:, 2:, 1:-1]
        + gray[:, 1:-1, :-2]
        + gray[:, 1:-1, 2:]
    )

    descriptor_names: list[str] = []
    descriptor_columns: list[np.ndarray] = []

    def column(name: str, values: np.ndarray) -> None:
        descriptor_names.append(name)
        descriptor_columns.append(np.asarray(values, dtype=np.float32))

    rgb_means = patches.mean(axis=(1, 2))
    rgb_stds = patches.std(axis=(1, 2))
    for channel_index, channel_name in enumerate(("red", "green", "blue")):
        column(f"rgb_{channel_name}_mean", rgb_means[:, channel_index])
        column(f"rgb_{channel_name}_std", rgb_stds[:, channel_index])
    column("luminance_mean", gray.mean(axis=(1, 2)))
    column("luminance_std", gray.std(axis=(1, 2)))
    for name, values in (
        ("diversity_horizontal", horizontal),
        ("diversity_vertical", vertical),
        ("diversity_diagonal", diagonal),
        ("diversity_anti_diagonal", anti_diagonal),
        ("diversity_total", diversity),
    ):
        column(name, values)
    for name, residual in (("cross_residual", cross), ("laplacian", laplacian)):
        column(f"{name}_absolute_mean", residual.mean(axis=(1, 2)))
        column(f"{name}_std", residual.std(axis=(1, 2)))
        column(f"{name}_q90", np.quantile(residual, 0.90, axis=(1, 2)))

    low_bit_u8 = patches_u8 & np.uint8(7)
    low_bit = low_bit_u8.astype(np.float32) / 7.0
    # Count every patch/channel/level in one pass instead of scanning the image
    # once for each of the eight levels.
    histogram_offsets = (
        (np.arange(PATCH_COUNT, dtype=np.int32)[:, None, None, None] * 3)
        + np.arange(3, dtype=np.int32)[None, None, None, :]
    ) * 8
    histogram_indices = low_bit_u8.astype(np.int32) + histogram_offsets
    low_histograms = np.bincount(
        histogram_indices.reshape(-1), minlength=PATCH_COUNT * 3 * 8
    ).reshape(PATCH_COUNT, 3, 8).astype(np.float32)
    low_histograms /= float(PATCH_SIZE * PATCH_SIZE)
    low_entropy = -np.sum(
        np.where(low_histograms > 0, low_histograms * np.log2(np.maximum(low_histograms, 1e-12)), 0.0),
        axis=2,
    )
    low_differences = {
        "horizontal": np.abs(np.diff(low_bit, axis=2)).mean(axis=(1, 2)),
        "vertical": np.abs(np.diff(low_bit, axis=1)).mean(axis=(1, 2)),
        "diagonal": np.abs(low_bit[:, 1:, 1:] - low_bit[:, :-1, :-1]).mean(axis=(1, 2)),
        "anti_diagonal": np.abs(low_bit[:, 1:, :-1] - low_bit[:, :-1, 1:]).mean(axis=(1, 2)),
    }
    low_bit_means = low_bit.mean(axis=(1, 2))
    low_bit_stds = low_bit.std(axis=(1, 2))
    for channel_index, channel_name in enumerate(("red", "green", "blue")):
        column(f"lowbit_{channel_name}_mean", low_bit_means[:, channel_index])
        column(f"lowbit_{channel_name}_std", low_bit_stds[:, channel_index])
        column(f"lowbit_{channel_name}_entropy", low_entropy[:, channel_index])
        for level in range(8):
            column(f"lowbit_{channel_name}_hist_{level}", low_histograms[:, channel_index, level])
        for direction, differences in low_differences.items():
            column(f"lowbit_{channel_name}_{direction}_gradient", differences[:, channel_index])

    descriptors = np.column_stack(descriptor_columns).astype(np.float32, copy=False)
    low_gradient_score = sum(low_differences.values()).mean(axis=1)
    return descriptors, tuple(descriptor_names), diversity, low_gradient_score


def _feature_vector(image: np.ndarray, collect_names: bool = False) -> tuple[np.ndarray, tuple[str, ...]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got shape {image.shape}")
    expected_shape = (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3)
    if image.shape != expected_shape:
        raise ValueError(f"Expected a cleaned RGB image with shape {expected_shape}, got {image.shape}")

    image = np.asarray(image, dtype=np.uint8)

    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        if collect_names:
            names.append(name)
        values.append(float(value))

    patch_descriptors, patch_names, diversity, low_gradient_score = _patch_descriptor_matrix(image)
    quantiles = np.quantile(patch_descriptors, PATCH_POOL_QUANTILES, axis=0)
    patch_aggregates = {
        "mean": patch_descriptors.mean(axis=0),
        "std": patch_descriptors.std(axis=0),
        "min": patch_descriptors.min(axis=0),
        "q10": quantiles[0],
        "q25": quantiles[1],
        "q50": quantiles[2],
        "q75": quantiles[3],
        "q90": quantiles[4],
        "max": patch_descriptors.max(axis=0),
    }
    for descriptor_index, descriptor_name in enumerate(patch_names):
        for aggregate_name, aggregate_values in patch_aggregates.items():
            add(f"patch_all_{descriptor_name}_{aggregate_name}", aggregate_values[descriptor_index])

    order = np.argsort(diversity, kind="stable")
    extreme_count = max(1, PATCH_COUNT // 3)
    poor_mean = patch_descriptors[order[:extreme_count]].mean(axis=0)
    rich_mean = patch_descriptors[order[-extreme_count:]].mean(axis=0)
    for descriptor_index, descriptor_name in enumerate(patch_names):
        add(f"texture_contrast_{descriptor_name}_poor_mean", poor_mean[descriptor_index])
        add(f"texture_contrast_{descriptor_name}_rich_mean", rich_mean[descriptor_index])
        add(f"texture_contrast_{descriptor_name}_rich_minus_poor", rich_mean[descriptor_index] - poor_mean[descriptor_index])

    max_gradient_patch = patch_descriptors[int(np.argmax(low_gradient_score))]
    for descriptor_index, descriptor_name in enumerate(patch_names):
        add(f"lowbit_max_gradient_patch_{descriptor_name}", max_gradient_patch[descriptor_index])

    full_rgb = image.astype(np.float32) / 255.0
    full_gray = 0.299 * full_rgb[..., 0] + 0.587 * full_rgb[..., 1] + 0.114 * full_rgb[..., 2]
    horizontal_differences = np.abs(np.diff(full_gray, axis=1))
    vertical_differences = np.abs(np.diff(full_gray, axis=0))
    for period in (8, 16):
        horizontal_boundary = (np.arange(1, full_gray.shape[1]) % period) == 0
        vertical_boundary = (np.arange(1, full_gray.shape[0]) % period) == 0
        boundary_mean = 0.5 * (
            float(np.mean(horizontal_differences[:, horizontal_boundary]))
            + float(np.mean(vertical_differences[vertical_boundary, :]))
        )
        interior_mean = 0.5 * (
            float(np.mean(horizontal_differences[:, ~horizontal_boundary]))
            + float(np.mean(vertical_differences[~vertical_boundary, :]))
        )
        add(f"compression_block_boundary_{period}_mean", boundary_mean)
        add(f"compression_block_boundary_{period}_ratio", boundary_mean / max(interior_mean, 1e-8))

    vector = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return vector, tuple(names) if collect_names else ()


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return features derived only from the supplied cleaned RGB pixels."""

    vector, _ = _feature_vector(image, collect_names=False)
    if vector.shape != (len(FEATURE_NAMES),):
        raise RuntimeError(f"Feature schema mismatch: expected {len(FEATURE_NAMES)}, got {len(vector)}")
    return vector


_, FEATURE_NAMES = _feature_vector(
    np.zeros((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3), dtype=np.uint8),
    collect_names=True,
)

def _extract_worker(index: int) -> np.ndarray:
    if _WORKER_IMAGES is None:
        raise RuntimeError("Feature worker was started without an image array.")
    return extract_features(_WORKER_IMAGES[index])


def extract_many(images: np.ndarray, deadline: float, description: str) -> np.ndarray:
    """Extract in deterministic order with at most eight CPU processes.

    Linux ``fork`` lets workers read the already-loaded image shard through
    copy-on-write memory instead of serializing each 256x256 RGB image.  Only
    compact feature rows cross process boundaries.
    """

    global _WORKER_IMAGES
    features = np.empty((len(images), len(FEATURE_NAMES)), dtype=np.float32)
    worker_count = min(MAX_FEATURE_WORKERS, os.cpu_count() or 1, max(1, len(images)))
    if worker_count == 1 or "fork" not in mp.get_all_start_methods():
        iterator = (extract_features(image) for image in images)
        for index, feature_row in enumerate(iterator):
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out while preparing {description} at row {index}/{len(images)}")
            features[index] = feature_row
            if (index + 1) % 250 == 0 or index + 1 == len(images):
                print(f"\rfeatures {description}: {index + 1}/{len(images)}", end="", flush=True)
        print()
        return features

    _WORKER_IMAGES = images
    pool = mp.get_context("fork").Pool(processes=worker_count)
    try:
        iterator = pool.imap(_extract_worker, range(len(images)), chunksize=8)
        for index, feature_row in enumerate(iterator):
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out while preparing {description} at row {index}/{len(images)}")
            features[index] = feature_row
            if (index + 1) % 250 == 0 or index + 1 == len(images):
                print(f"\rfeatures {description}: {index + 1}/{len(images)}", end="", flush=True)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
        _WORKER_IMAGES = None
    print()
    return features


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def prepare_training(artifacts_dir: Path, output_dir: Path, deadline: float) -> list[dict]:
    cleaned_paths = sorted((artifacts_dir / "cleaned_train_npz").glob("train_cleaned_*.npz"))
    if not cleaned_paths:
        raise FileNotFoundError("No cleaned training NPZ files found; run clean.py first.")

    outputs = []
    for index, cleaned_path in enumerate(cleaned_paths):
        with np.load(cleaned_path) as cleaned:
            images = cleaned["images"]
            expected_shape = (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3)
            if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
                raise RuntimeError(
                    f"{cleaned_path.name} contains image shape {images.shape[1:]}; expected {expected_shape}. "
                    "Rerun clean.py with its default --image_size before preparing features."
                )
            labels = cleaned["labels"].astype(np.int8)
            source_classes = cleaned["source_class"].astype(np.int8)
            features = extract_many(images, deadline, cleaned_path.name)
        output_path = output_dir / f"train_features_{index:03d}.npz"
        atomic_save_npz(output_path, features=features, labels=labels, source_class=source_classes)
        outputs.append({"input": cleaned_path.name, "output": output_path.name, "rows": len(labels)})
    return outputs


def prepare_labeled_split(data_dir: Path, output_dir: Path, split: str, deadline: float) -> dict:
    image_rows: list[np.ndarray] = []
    labels: list[int] = []
    source_classes: list[int] = []
    for row_index, row in enumerate(parquet_rows(data_dir / split, ["image", "source_class"])):
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out while decoding {split} at row {row_index}")
        source_class = int(row["source_class"])
        image_rows.append(clean_image_bytes(row["image"], DEFAULT_IMAGE_SIZE))
        labels.append(0 if source_class == 0 else 1)
        source_classes.append(source_class)

    features = extract_many(np.stack(image_rows), deadline, split)
    output_path = output_dir / f"{split}.npz"
    atomic_save_npz(
        output_path,
        features=features,
        labels=np.asarray(labels, dtype=np.int8),
        source_class=np.asarray(source_classes, dtype=np.int8),
    )
    return {"output": output_path.name, "rows": len(labels)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout_seconds must be positive")
    start = time.time()
    deadline = start + max(1, args.timeout_seconds - 5)
    data_dir, artifacts_dir = paths()
    output_dir = artifacts_dir / "prepared" / "task02_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_family_counts = {
        prefix.removesuffix("_"): sum(name.startswith(prefix) for name in FEATURE_NAMES)
        for prefix in FEATURE_FAMILY_PREFIXES
    }
    if sum(feature_family_counts.values()) != len(FEATURE_NAMES):
        raise RuntimeError("Every feature must belong to one declared feature family.")
    schema = {
        "feature_version": FEATURE_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_dtype": "float32",
        "feature_family_counts": feature_family_counts,
        "pixel_only": True,
        "uses_source_metadata": False,
        "implementation": "independent_literature_inspired",
        "clean_image_size": list(DEFAULT_IMAGE_SIZE),
        "native_patch_coverage": {
            "patch_size": [PATCH_SIZE, PATCH_SIZE],
            "patch_grid": [PATCH_GRID_SIZE, PATCH_GRID_SIZE],
            "patch_count": PATCH_COUNT,
            "all_cleaned_pixels_used": True,
        },
    }
    atomic_write_json(output_dir / "schema.json", schema)
    atomic_write_json(output_dir / "summary.json", {"complete": False, **schema})

    training_outputs = prepare_training(artifacts_dir, output_dir, deadline)
    split_outputs = {
        split: prepare_labeled_split(data_dir, output_dir, split, deadline)
        for split in ("calibration", "validation", "validation_augmented")
    }
    summary = {
        "complete": True,
        **schema,
        "training": training_outputs,
        "splits": split_outputs,
        "seconds": round(time.time() - start, 2),
        "timeout_seconds": args.timeout_seconds,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "complete": True,
                "feature_version": FEATURE_VERSION,
                "feature_count": len(FEATURE_NAMES),
                "training_rows": sum(item["rows"] for item in training_outputs),
                "split_rows": {name: item["rows"] for name, item in split_outputs.items()},
                "seconds": summary["seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
