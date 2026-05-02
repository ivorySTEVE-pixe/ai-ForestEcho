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

## Status

Foundation only — modules are stubs. Next steps: wire a real dataset, implement feature extraction, train baseline CNN.
