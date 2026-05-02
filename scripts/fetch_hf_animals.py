"""Fetch a larger birds/animals dataset from Hugging Face for local training.

Source dataset:
  cgeorgiaw/animal-sounds  (CC-BY-4.0)

This script exports clips into:
  data/raw/<label_slug>/<clip_id>.wav

Usage:
  python scripts/fetch_hf_animals.py
  python scripts/fetch_hf_animals.py --subset birds --max-clips 2000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

DEFAULT_DATASET = "cgeorgiaw/animal-sounds"
DEFAULT_SUBSET = "birds"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--subset", default=DEFAULT_SUBSET, help="e.g. birds, dogs, orcas")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--max-clips", type=int, default=1200)
    ap.add_argument("--min-per-class", type=int, default=12)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"loading dataset={args.dataset} subset={args.subset} split={args.split}")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.max_clips > 0:
        ds = ds.select(range(min(len(ds), args.max_clips)))

    counts: dict[str, int] = {}
    saved = 0
    for i, row in enumerate(ds):
        if "audio" not in row or "label" not in row:
            continue

        label = slugify(str(row["label"]))
        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if len(arr) == 0:
            continue

        cls_dir = out_root / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{args.subset}_{i:06d}.wav"
        out_path = cls_dir / fname
        sf.write(out_path, arr, sr)
        counts[label] = counts.get(label, 0) + 1
        saved += 1

        if saved % 200 == 0:
            print(f"  saved {saved} clips...")

    # Drop tiny classes to avoid unstable training on 1-2 clips.
    removed = []
    for label, n in sorted(counts.items()):
        if n >= args.min_per_class:
            continue
        cls_dir = out_root / label
        for p in cls_dir.glob("*.wav"):
            p.unlink()
        try:
            cls_dir.rmdir()
        except OSError:
            pass
        removed.append((label, n))

    kept = {k: v for k, v in counts.items() if v >= args.min_per_class}
    print("\nsummary:")
    print(f"  total saved: {saved}")
    print(f"  classes kept (>= {args.min_per_class}): {len(kept)}")
    if removed:
        print(f"  classes removed (< {args.min_per_class}): {len(removed)}")
    for label, n in sorted(kept.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        print(f"    {label:24s} {n}")

    print(f"\nready: {out_root}")
    print("next: python -m forestecho.train --config configs/default.yaml")


if __name__ == "__main__":
    main()
