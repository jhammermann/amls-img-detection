from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def artifacts_path() -> Path:
    path = Path(__file__).resolve().parent / "artifacts"
    path.mkdir(exist_ok=True)
    return path


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


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.20) -> float:
    real_scores = np.sort(scores[labels == 0])
    allowed_fp = int(np.floor(max_fpr * len(real_scores)))
    if allowed_fp <= 0:
        return float(np.nextafter(real_scores[-1], np.inf))
    return float(real_scores[-allowed_fp])


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    real = labels == 0
    ai = labels == 1
    return {
        "threshold": float(threshold),
        "fpr_real": float(np.mean(pred[real] == 1)),
        "recall_ai": float(np.mean(pred[ai] == 1)),
        "accuracy": float(np.mean(pred == labels)),
        "rows": int(len(labels)),
        "real_rows": int(np.sum(real)),
        "ai_rows": int(np.sum(ai)),
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def cleaned_train_files(artifacts_dir: Path) -> list[Path]:
    cleaned_dir = artifacts_dir / "cleaned_train_npz"
    files = sorted(cleaned_dir.glob("train_cleaned_*.npz"))
    if not files:
        raise FileNotFoundError(f"No cleaned train files found in {cleaned_dir}; run clean.py first.")
    return files


def class_counts(files: list[Path]) -> tuple[int, int]:
    counts = [0, 0]
    for path in files:
        with np.load(path) as data:
            labels = data["labels"]
            counts[0] += int(np.sum(labels == 0))
            counts[1] += int(np.sum(labels == 1))
    return counts[0], counts[1]


def progress_line(epoch: int, epochs: int, file_index: int, file_count: int, batch_index: int, batch_count: int) -> None:
    width = 24
    done = int(width * batch_index / max(1, batch_count))
    bar = "#" * done + "." * (width - done)
    print(
        f"\rtrain epoch {epoch}/{epochs} file {file_index}/{file_count} "
        f"[{bar}] {batch_index}/{batch_count} batches",
        end="",
        flush=True,
    )


def train_one_file(
    model,
    optimizer,
    criterion,
    path: Path,
    batch_size: int,
    image_size: int,
    device,
    deadline: float,
    epoch: int,
    epochs: int,
    file_index: int,
    file_count: int,
) -> dict:
    model.train()
    losses, correct, rows = [], 0, 0
    with np.load(path) as data:
        images = data["images"]
        labels = data["labels"].astype(np.int64)
        order = np.random.permutation(len(labels))
        batch_count = int(np.ceil(len(order) / batch_size))
        for batch_index, start in enumerate(range(0, len(order), batch_size), start=1):
            if time.time() > deadline:
                break
            progress_line(epoch, epochs, file_index, file_count, batch_index, batch_count)
            idx = order[start : start + batch_size]
            x = normalize_batch(images[idx], device, image_size)
            y = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(1) == y).sum().detach().cpu())
            rows += len(idx)
    return {"loss": float(np.mean(losses)) if losses else 0.0, "accuracy": correct / max(1, rows), "rows": rows}


@torch.inference_mode()
def predict_npz(model, path: Path, batch_size: int, image_size: int, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with np.load(path) as data:
        images = data["images"]
        labels = data["labels"].astype(np.int8)
        scores = []
        for start in range(0, len(labels), batch_size):
            x = normalize_batch(images[start : start + batch_size], device, image_size)
            scores.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores), labels


def run_id(args: argparse.Namespace) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return (
        f"{stamp}_cnn_s{args.image_size}_c{args.channels}_"
        f"e{args.epochs}_b{args.batch_size}_lr{args.lr:g}_seed{args.seed}"
    )


def update_runs_summary(task_dir: Path) -> None:
    rows = []
    for metrics_path in sorted((task_dir / "runs").glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        args = metrics["args"]
        validation = metrics["validation"]
        augmented = metrics["validation_augmented"]
        rows.append(
            {
                "run": metrics_path.parent.name,
                "image_size": args["image_size"],
                "channels": args["channels"],
                "epochs": args["epochs"],
                "batch_size": args["batch_size"],
                "lr": args["lr"],
                "seconds": metrics["seconds"],
                "val_fpr_real": validation["fpr_real"],
                "val_recall_ai": validation["recall_ai"],
                "val_accuracy": validation["accuracy"],
                "val_aug_fpr_real": augmented["fpr_real"],
                "val_aug_recall_ai": augmented["recall_ai"],
            }
        )
    if not rows:
        return
    with (task_dir / "runs_summary.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.time()
    seed_everything(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")

    artifacts_dir = artifacts_path()
    task_dir = artifacts_dir / "task02"
    task_dir.mkdir(parents=True, exist_ok=True)
    run_dir = task_dir / "runs" / run_id(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = artifacts_dir / "prepared" / "task02"
    files = cleaned_train_files(artifacts_dir)
    real_count, ai_count = class_counts(files)

    model = build_model(args.channels).to(device)
    weights = torch.tensor(
        [(real_count + ai_count) / (2 * real_count), (real_count + ai_count) / (2 * ai_count)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    deadline = start_time + args.timeout_seconds * 0.88

    for epoch in range(args.epochs):
        shuffled_files = [Path(path) for path in np.random.permutation(files)]
        for file_index, path in enumerate(shuffled_files, start=1):
            if time.time() > deadline:
                break
            stats = train_one_file(
                model,
                optimizer,
                criterion,
                path,
                args.batch_size,
                args.image_size,
                device,
                deadline,
                epoch + 1,
                args.epochs,
                file_index,
                len(shuffled_files),
            )
            stats.update({"epoch": epoch + 1, "file": Path(path).name})
            history.append(stats)
        print()
        checkpoint = {"model": model.state_dict(), "channels": args.channels, "image_size": args.image_size}
        torch.save(checkpoint, task_dir / "model.pt")
        torch.save(checkpoint, run_dir / "model.pt")
        if time.time() > deadline:
            break

    calibration_scores, calibration_labels = predict_npz(
        model, prepared_dir / "calibration.npz", args.batch_size, args.image_size, device
    )
    threshold = threshold_at_fpr(calibration_scores, calibration_labels, max_fpr=0.20)
    validation_scores, validation_labels = predict_npz(
        model, prepared_dir / "validation.npz", args.batch_size, args.image_size, device
    )
    augmented_scores, augmented_labels = predict_npz(
        model, prepared_dir / "validation_augmented.npz", args.batch_size, args.image_size, device
    )

    metrics = {
        "train": {"real_rows": real_count, "ai_rows": ai_count, "history": history},
        "calibration": metrics_at_threshold(calibration_scores, calibration_labels, threshold),
        "validation": metrics_at_threshold(validation_scores, validation_labels, threshold),
        "validation_augmented": metrics_at_threshold(augmented_scores, augmented_labels, threshold),
        "seconds": round(time.time() - start_time, 2),
        "args": vars(args),
    }
    save_json(task_dir / "threshold.json", {"threshold": threshold, "max_fpr": 0.20})
    save_json(task_dir / "metrics.json", metrics)
    save_json(run_dir / "threshold.json", {"threshold": threshold, "max_fpr": 0.20})
    save_json(run_dir / "metrics.json", metrics)
    for history_path in [task_dir / "train_history.csv", run_dir / "train_history.csv"]:
        with history_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["epoch", "file", "loss", "accuracy", "rows"])
            writer.writeheader()
            writer.writerows(history)
    update_runs_summary(task_dir)
    print(metrics["validation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
