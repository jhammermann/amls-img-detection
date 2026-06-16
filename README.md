AMLS AI Image Detection
=======================

This Readme is only for our internal use.

# Installation
```bash
cd solution/

sudo apt update && sudo apt install -y python3.11-venv && python3.11 -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt

mkdir artifacts && mkdir data-readonly
```
Then, paste the downloaded data into 'data-readonly'.

The data folder is read-only (requirement from the task pdf) and as a reminder, I named it like that. We'll change the name to 'data' later.


# Git
We will work in feature branches and merge into main when the feature is complete. The merge request will need to be approved by another team member, nobody can push onto main directly. We will squash the commits, so don't worry about doing too many commits on your feature branch.

# Important requirements
Some important information excerpts cited from the tasks pdf:

The contents of `solution/data/` will be identical to the provided download except the contents of `solution/data/predict/`, which will be used to evaluate your models. During execution, the `solution/data/` directory is mounted into the container as read-only input. Your code must therefore treat this directory as read-only and write all derived files, caches, models, and predictions only under `solution/artifacts/`. The script execution order is:

1. `python clean.py --timeout_seconds 600`
2. `python prepare.py --timeout_seconds 600`
3. `python train.py --timeout_seconds 1800`
4. `python predict.py --timeout_seconds 600`
5. `python train_augmented.py --timeout_seconds 1800`
6. `python predict_augmented.py --timeout_seconds 600`

Note that `prepare.py` should not prepare data from `solution/data/predict/` as it may change after training. Keep in mind that each script is terminated after a timeout that is given as CLI input. Hence, it makes sense to regularly write the best model checkpoint into `solution/artifacts/`.