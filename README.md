AMLS Image Detection
====================

Starter project for the AMLS 2026 image detection exercise.

This repository is scaffolded from the DrivenData Cookiecutter Data Science
template and adjusted for a Python image-processing / machine-learning workflow.
It keeps data, notebooks, source code, trained models, and reports separated so
experiments can grow without turning the project root into soup.

Getting Started
---------------

Create and activate an environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```


Typical Workflow
----------------

1. Put original image data in `data/raw/`.
2. Keep exploratory work in `notebooks/`.
3. Move reusable data preparation code into `src/data/`.
4. Put feature extraction and preprocessing code in `src/features/`.
5. Put training and inference code in `src/models/`.
6. Save generated plots and writeups under `reports/`.

Large datasets, trained model binaries, caches, and local environment files are
ignored by git by default.

