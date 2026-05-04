"""List and compare ForestEcho training runs.

Usage:
  python scripts/list_runs.py
  python scripts/list_runs.py --sort test_accuracy --limit 10
  python scripts/list_runs.py --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_runs(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not runs_root.exists():
        return rows
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        summary = load_json(d / "summary.json")
        meta = load_json(d / "metadata.json")
        art = load_json(d / "artifacts.json")
        if not summary:
            continue
        git = meta.get("git", {}) if isinstance(meta.get("git", {}), dict) else {}
        rows.append(
            {
                "run_id": summary.get("run_id", d.name),
                "val_accuracy": float(summary.get("val_accuracy", 0.0)),
                "test_accuracy": float(summary.get("test_accuracy", 0.0)),
                "val_macro_f1": float(summary.get("val", {}).get("macro_f1", 0.0)),
                "test_macro_f1": float(summary.get("test", {}).get("macro_f1", 0.0)),
                "checkpoint": str(summary.get("checkpoint", "")),
                "report_dir": str(summary.get("report_dir", "")),
                "commit": str(git.get("commit", ""))[:12],
                "dirty": bool(git.get("dirty", False)),
                "branch": str(git.get("branch", "")),
                "best_checkpoint": str(art.get("best_checkpoint", "")),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No runs found under runs/.")
        return
    headers = [
        "run_id",
        "test_acc",
        "val_acc",
        "test_f1",
        "val_f1",
        "branch",
        "dirty",
        "commit",
    ]
    data = []
    for r in rows:
        data.append(
            [
                str(r["run_id"]),
                f"{r['test_accuracy']:.4f}",
                f"{r['val_accuracy']:.4f}",
                f"{r['test_macro_f1']:.4f}",
                f"{r['val_macro_f1']:.4f}",
                str(r["branch"]),
                "yes" if r["dirty"] else "no",
                str(r["commit"]),
            ]
        )
    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument(
        "--sort",
        default="run_id",
        choices=["run_id", "val_accuracy", "test_accuracy", "val_macro_f1", "test_macro_f1"],
    )
    ap.add_argument("--desc", action="store_true", help="Sort descending.")
    ap.add_argument("--limit", type=int, default=0, help="Show top N runs (0 = all).")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of table.")
    args = ap.parse_args()

    rows = collect_runs(Path(args.runs_dir))
    rows.sort(key=lambda x: x.get(args.sort, 0), reverse=args.desc)
    if args.limit > 0:
        rows = rows[: args.limit]

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print_table(rows)
    if rows:
        top = max(rows, key=lambda x: x["test_accuracy"])
        print("\nBest by test_accuracy:")
        print(f"  run_id={top['run_id']} test_acc={top['test_accuracy']:.4f} ckpt={top['checkpoint']}")
        print(f"  report={top['report_dir']}")


if __name__ == "__main__":
    main()
