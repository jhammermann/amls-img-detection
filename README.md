AMLS AI Image Detection
=======================

This Readme is only for our internal use.

# Local Setup
```bash
cd solution

sudo apt update
sudo apt install -y python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt

mkdir -p artifacts data-readonly
```
Place the downloaded dataset in `solution/data-readonly/`.

Scripts look for `solution/data/` first and fall back to `solution/data-readonly/`, so the local folder name can stay as `data-readonly`.

Torch is installed separately from the CPU wheel index and is intentionally not listed in `requirements.txt`. The Dockerfile does the same.

# Task 2 Pipeline

```bash
python clean.py --timeout_seconds 600
python prepare.py --timeout_seconds 600
python train.py --timeout_seconds 1800
python predict.py --timeout_seconds 600
```

Main outputs:

```text
artifacts/task02/model.pt
artifacts/task02/metrics.json
artifacts/task02/predictions.csv
```

## Engineered-Feature Task 2 Alternative

The classical alternative derives its inputs only from the cleaned RGB pixels.
It does not use image dimensions, aspect ratio, encoded size, or file format.
The implementation is original code inspired by published methods; it does not
include third-party source code.

Every cleaned pixel is covered by an exhaustive 8x8 grid of native-resolution
32x32 patches, so the image is not reduced to or cropped into a single 32x32
region. The 836 retained features pool patch color, residual, texture, and
three-lowest-bit-plane statistics across the entire image, compare its
texture-rich and texture-poor patches, and add four block-boundary signals.
These choices are
inspired by [PatchCraft](https://arxiv.org/abs/2311.12397),
[LOTA](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LOTA_Bit-Planes_Guided_AI-Generated_Image_Detection_ICCV_2025_paper.html).
Unused global, HOG/LBP/GLCM/wavelet, and periodic-spectrum branches were removed
after the selected model was shown not to consume them. The
classifier is a deterministic, binary-class-balanced 600-tree Extra Trees
ensemble. Its decision threshold is selected only from real calibration scores
with a tie-safe Neyman-Pearson order statistic and checked with one-sided Wilson
and exact Clopper-Pearson FPR bounds. Validation data never selects the model or
threshold.

Run it after `clean.py`:

```bash
python prepare_features.py --timeout_seconds 600
python train_features.py --timeout_seconds 1800
python predict_features.py --timeout_seconds 600
```

Its prepared features, model, calibration threshold, and metrics are written to:

```text
artifacts/prepared/task02_features/
artifacts/task02_features/model.joblib
artifacts/task02_features/threshold.json
artifacts/task02_features/metrics.json
artifacts/task02_features/arguments.json
```

Like the CNN pipeline, inference writes the required submission file to
`artifacts/task02/predictions.csv`. Running either predictor replaces that file.

# Hyperparameter Runs

For report comparisons, each `train.py` run is archived under `artifacts/task02/runs/` and summarized in:

```text
artifacts/task02/runs_summary.csv
```

Example CPU-friendly runs:

```bash
python train.py --timeout_seconds 1800 --epochs 5 --batch_size 128 --channels 32 --lr 0.001 --image_size 128
python train.py --timeout_seconds 1800 --epochs 4 --batch_size 96 --channels 32 --lr 0.001 --image_size 192
python train.py --timeout_seconds 1800 --epochs 2 --batch_size 64 --channels 32 --lr 0.001 --image_size 256
python train.py --timeout_seconds 1800 --epochs 6 --batch_size 128 --channels 16 --lr 0.001 --image_size 192
```

Use `runs_summary.csv` for the report table. `train_history.csv` is only per-chunk training loss/accuracy.


# Git
We will work in feature branches and merge into main when the feature is complete. The pull request will need to be approved by another team member, nobody can push onto main directly. We will squash the commits, so don't worry about doing too many commits on your feature branch.

### Branch naming schema:
```text
<category>/<short-description>
```
### Four Main Categories:

* **`feat/`** – For new features or code additions.
* *Example:* `feat/user-login`
* **`fix/`** – For fixing broken code or bugs.
* *Example:* `fix/jwt-expiration`
* **`chore/`** – For routine maintenance, updating dependencies, or tooling/Docker changes.
* *Example:* `chore/add-dockerfile`
* **`docs/`** – For documentation updates only.
* *Example:* `docs/update-readme`


# Important Requirements
Some important information excerpts cited from the tasks pdf:

The contents of `solution/data/` will be identical to the provided download except the contents of `solution/data/predict/`, which will be used to evaluate your models. During execution, the `solution/data/` directory is mounted into the container as read-only input. Your code must therefore treat this directory as read-only and write all derived files, caches, models, and predictions only under `solution/artifacts/`. The script execution order is:

1. `python clean.py --timeout_seconds 600`
2. `python prepare.py --timeout_seconds 600`
3. `python train.py --timeout_seconds 1800`
4. `python predict.py --timeout_seconds 600`
5. `python train_augmented.py --timeout_seconds 1800`
6. `python predict_augmented.py --timeout_seconds 600`

Note that `prepare.py` should not prepare data from `solution/data/predict/` as it may change after training. Keep in mind that each script is terminated after a timeout that is given as CLI input. Hence, it makes sense to regularly write the best model checkpoint into `solution/artifacts/`.
