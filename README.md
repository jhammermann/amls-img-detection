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
