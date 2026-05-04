# ForestEcho

ML system for recognizing animal vocalizations in the wild and identifying species from audio.

## Overview

ForestEcho ingests field audio recordings and classifies species by fine-tuning a **pretrained Audio Spectrogram Transformer (AST)** — `MIT/ast-finetuned-audioset-10-10-0.4593` — which was trained on AudioSet and already recognizes a wide range of animal and bird vocalizations. Only a small classifier head is trained by default (backbone frozen), so you can fine-tune on a few hundred clips per species.

## Structure

```
src/forestecho/   # core package
  data.py         # dataset loading & augmentation
  features.py     # spectrogram / MFCC extraction
  model.py        # CNN classifier
  train.py        # training loop
  infer.py        # inference on new clips
configs/          # YAML configs (data, model, training)
data/raw/         # raw audio files (gitignored)
data/processed/   # cached features
models/           # trained checkpoints
notebooks/        # exploration
tests/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m forestecho.train --config configs/default.yaml
python -m forestecho.infer --audio path/to/clip.wav
python -m forestecho.app                     # launch the web UI at http://127.0.0.1:7860
```

## Datasets

### Quick demo (small, no auth)

```bash
python scripts/fetch_demo_data.py
```

Downloads ESC-50 animal classes (10 classes, ~400 clips) into `data/raw/<species>/`.

### Larger local training set (birds/animals)

```bash
pip install -r requirements.txt
python scripts/fetch_hf_animals.py --subset birds --max-clips 2000 --min-per-class 12
```

This pulls from `cgeorgiaw/animal-sounds` on Hugging Face and exports to the same
`data/raw/<species>/` format used by `forestecho.train`.

Useful subsets to try:
- `birds` (best for your current goal)
- `dogs`
- `zebra_finch`
- `orcas`

Then train:

```bash
python -m forestecho.train --config configs/default.yaml
```

## Evaluation Reports

After each training run, ForestEcho now writes evaluation artifacts to:

`reports/<timestamp>/`

Contents:
- `val_confusion_matrix.png` and `test_confusion_matrix.png`
- `val_classification_report.csv` and `test_classification_report.csv`
- `val_classification_report.json` and `test_classification_report.json`
- `run_summary.json` (quick run-level metrics)

## Data Quality Filters

Training now applies clip-level quality filtering before dataset split:
- low RMS rejection
- excessive silence rejection
- clipping rejection

Configure in [default.yaml](/Users/aankansarkar/ForestEcho/configs/default.yaml):
- `data.quality.min_rms`
- `data.quality.max_silence_ratio`
- `data.quality.max_clipping_ratio`
- `data.quality.silence_amplitude`

During training you will see:
- kept/rejected counts
- rejection reason counts
- class balance summary (`min`/`max` samples per class)

## Training Recipe Upgrades

The training loop now includes:
- class-weighted cross-entropy (for imbalanced classes)
- label smoothing
- cosine annealing LR scheduler
- early stopping on validation accuracy

Tune in [default.yaml](/Users/aankansarkar/ForestEcho/configs/default.yaml) under:
- `train.recipe.use_class_weights`
- `train.recipe.label_smoothing`
- `train.recipe.min_lr`
- `train.recipe.early_stopping_patience`

## Active Learning Retraining

When you mark predictions as wrong in the app, corrections are saved to:

`data/feedback/corrections.jsonl`

Training now reads this queue and applies corrected labels before data split.

Config:
- `train.feedback.use_queue`
- `train.feedback.path`

You will see a startup log line like:
- `feedback_queue: loaded=X applied=Y from=data/feedback/corrections.jsonl`

## Noise Robustness Augmentation

Training now supports train-only waveform augmentation to improve field robustness:
- random gain jitter
- additive gaussian noise
- random time masking (dropout in time axis)

Config:
- `train.augmentation.enabled`
- `train.augmentation.gain_min`
- `train.augmentation.gain_max`
- `train.augmentation.noise_prob`
- `train.augmentation.noise_level`
- `train.augmentation.time_mask_prob`
- `train.augmentation.time_mask_max_ratio`

## Run Registry (Reproducibility)

Each training run now creates a run record under:

`runs/<timestamp>/`

Files:
- `config.yaml` (exact config snapshot used)
- `metadata.json` (python version, git commit/branch/dirty flag, command)
- `summary.json` (core metrics and pointers)
- `artifacts.json` (checkpoint/report paths)

Reports still live under:

`reports/<timestamp>/`

Quick pointer to most recent run:

`runs/latest.json`

Compare runs quickly:

```bash
python scripts/list_runs.py --sort test_accuracy --desc --limit 10
```

## Species Catalog

Species names, scientific names, group labels, and descriptions can now be edited in:

`configs/species_catalog.json`

The app loads this file at startup. If missing or invalid, it falls back to built-in defaults.

## Production Configs

Two ready-to-run configs are available:

- [birds_prod.yaml](/Users/aankansarkar/ForestEcho/configs/birds_prod.yaml): bird-focused, full fine-tuning (`freeze_backbone: false`, low LR).
- [mixed_prod.yaml](/Users/aankansarkar/ForestEcho/configs/mixed_prod.yaml): mixed animals, head-focused fine-tuning (`freeze_backbone: true`).

Bird-focused training:

```bash
python -m forestecho.train --config configs/birds_prod.yaml
python scripts/list_runs.py --sort test_accuracy --desc --limit 5
python -m forestecho.app --ckpt models/best.pt --port 7868
```

Mixed-animal training:

```bash
python -m forestecho.train --config configs/mixed_prod.yaml
python scripts/list_runs.py --sort test_accuracy --desc --limit 5
python -m forestecho.app --ckpt models/best.pt --port 7868
```

## Status

Foundation only — modules are stubs. Next steps: wire a real dataset, implement feature extraction, train baseline CNN.
