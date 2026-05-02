"""Download a small open-licensed demo dataset for ForestEcho.

Uses ESC-50 (https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0), which
includes 10 animal classes with 40 five-second WAV clips each. We copy
those into data/raw/<class>/ so training works out of the box.

Usage:
    python scripts/fetch_demo_data.py
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
ZIP_NAME = "ESC-50-master.zip"

# ESC-50 animal categories (target → human-friendly folder name)
ANIMAL_CATEGORIES = {
    "dog": "dog",
    "rooster": "rooster",
    "pig": "pig",
    "cow": "cow",
    "frog": "frog",
    "cat": "cat",
    "hen": "hen",
    "insects": "insects",
    "sheep": "sheep",
    "crow": "crow",
}

UA = {"User-Agent": "ForestEcho-demo-fetch/1.0"}


def download_zip(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"  using cached {dest}")
        return dest
    print(f"  downloading {ZIP_URL}  (~600 MB, this may take a few minutes)")
    req = urllib.request.Request(ZIP_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        chunk = 1 << 20  # 1 MB
        total = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            total += len(buf)
            print(f"    {total / 1_048_576:.1f} MB", end="\r")
    print()
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--cache-dir", default=".cache")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    print("=== Fetch ESC-50 (open-licensed animal sounds) ===")
    zip_path = download_zip(cache / ZIP_NAME)

    print("\n=== Extracting animal classes ===")
    counts = {k: 0 for k in ANIMAL_CATEGORIES}
    with zipfile.ZipFile(zip_path) as zf:
        # Read the metadata CSV
        meta_name = "ESC-50-master/meta/esc50.csv"
        with zf.open(meta_name) as fh:
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
            rows = list(reader)

        for row in rows:
            cat = row["category"]
            if cat not in ANIMAL_CATEGORIES:
                continue
            src_in_zip = f"ESC-50-master/audio/{row['filename']}"
            target_dir = out_root / ANIMAL_CATEGORIES[cat]
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / row["filename"]
            if target_path.exists():
                counts[cat] += 1
                continue
            try:
                with zf.open(src_in_zip) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                counts[cat] += 1
            except KeyError:
                print(f"  ! missing in zip: {src_in_zip}")

    print("\n=== Summary ===")
    for cat, n in counts.items():
        print(f"  {cat:10s}  {n} clips")
    total = sum(counts.values())
    print(f"\n{total} clips across {len(counts)} classes -> {out_root}")
    print("\nNow run:  python -m forestecho.train")


if __name__ == "__main__":
    main()
