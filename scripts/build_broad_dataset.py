"""Build a broader multi-subset dataset for ForestEcho.

This script pulls multiple subsets from `cgeorgiaw/animal-sounds` and merges
them into one training folder:

  data/raw_broad/<class_name>/*.wav

Usage:
  python scripts/build_broad_dataset.py
  python scripts/build_broad_dataset.py --preset broad_plus --out data/raw_broad --min-per-class 20
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf


PRESETS: dict[str, dict[str, int]] = {
    # Good general starting point.
    "broad": {
        "birds": 3000,
        "dogs": 1500,
        "orca": 1200,
        "zebra_finch": 1200,
    },
    # Wider set, slower download.
    "broad_plus": {
        "birds": 5000,
        "dogs": 2000,
        "orca": 1800,
        "zebra_finch": 1800,
    },
    # Fast smoke build.
    "quick": {
        "birds": 1200,
        "dogs": 600,
        "orca": 500,
    },
}


def count_wavs(root: Path) -> tuple[int, int]:
    classes = 0
    clips = 0
    for d in root.iterdir() if root.exists() else []:
        if not d.is_dir():
            continue
        n = len(list(d.glob("*.wav")))
        if n > 0:
            classes += 1
            clips += n
    return classes, clips


def prune_classes(root: Path, min_per_class: int) -> tuple[int, int]:
    if min_per_class <= 1:
        return 0, 0
    removed_classes = 0
    removed_clips = 0
    for d in root.iterdir() if root.exists() else []:
        if not d.is_dir():
            continue
        files = list(d.glob("*.wav"))
        if len(files) >= min_per_class:
            continue
        removed_classes += 1
        removed_clips += len(files)
        for p in files:
            p.unlink()
        try:
            d.rmdir()
        except OSError:
            pass
    return removed_classes, removed_clips


def normalize_wavs(root: Path) -> int:
    """Repair any malformed WAV headers by re-writing PCM float32."""
    repaired = 0
    for wav in root.rglob("*.wav"):
        try:
            arr, sr = sf.read(wav, dtype="float32", always_2d=False)
            sf.write(wav, arr, sr)
            repaired += 1
        except Exception:
            continue
    return repaired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="broad")
    ap.add_argument("--out", default="data/raw_broad")
    ap.add_argument("--min-per-class", type=int, default=20)
    ap.add_argument(
        "--subset-min-per-class",
        type=int,
        default=1,
        help="Minimum clips per class during each subset fetch (keep low for broad merge).",
    )
    ap.add_argument("--clear", action="store_true", help="Clear output folder before building.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for p in out.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    plan = PRESETS[args.preset]
    print(f"=== Building preset: {args.preset} -> {out} ===")
    for subset, max_clips in plan.items():
        cmd = [
            sys.executable,
            "scripts/fetch_hf_animals.py",
            "--subset",
            subset,
            "--max-clips",
            str(max_clips),
            "--min-per-class",
            str(args.subset_min_per_class),
            "--out",
            str(out),
        ]
        print(f"\n--- Fetch subset={subset} max_clips={max_clips} ---")
        subprocess.run(cmd, check=True)

    removed_classes, removed_clips = prune_classes(out, args.min_per_class)
    repaired = normalize_wavs(out)

    classes, clips = count_wavs(out)
    print("\n=== Broad Dataset Ready ===")
    print(f"  output:  {out}")
    print(f"  classes: {classes}")
    print(f"  clips:   {clips}")
    if removed_classes:
        print(f"  pruned:  {removed_classes} sparse classes / {removed_clips} clips")
    print(f"  wav repaired: {repaired}")
    print("\nNext:")
    print("  python -m forestecho.train --config configs/broad_prod.yaml")


if __name__ == "__main__":
    main()
