from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import AnimalAudioDataset, collate_waveforms
from .model import ASTSpeciesClassifier


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_sizes(total: int, val_frac: float, test_frac: float) -> tuple[int, int, int]:
    """Create robust train/val/test sizes that always sum to total."""
    if total < 3:
        return total, 0, 0
    n_val = max(1, int(total * val_frac))
    n_test = max(1, int(total * test_frac))
    n_train = total - n_val - n_test
    if n_train < 1:
        # Keep at least one train sample by reducing non-train splits.
        deficit = 1 - n_train
        reduce_test = min(deficit, n_test - 1)
        n_test -= reduce_test
        deficit -= reduce_test
        reduce_val = min(deficit, n_val - 1)
        n_val -= reduce_val
        n_train = total - n_val - n_test
    return n_train, n_val, n_test


def class_weights_from_subset(train_subset, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in train_subset.indices:
        label = train_subset.dataset.samples[idx].label
        counts[label] += 1.0
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum() * float(num_classes)
    return weights


def augment_waveforms(
    waveforms: list[np.ndarray],
    cfg: dict,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if not cfg.get("enabled", False):
        return waveforms
    out: list[np.ndarray] = []
    gain_low = float(cfg.get("gain_min", 0.8))
    gain_high = float(cfg.get("gain_max", 1.2))
    noise_prob = float(cfg.get("noise_prob", 0.6))
    noise_level = float(cfg.get("noise_level", 0.01))
    mask_prob = float(cfg.get("time_mask_prob", 0.4))
    mask_max_ratio = float(cfg.get("time_mask_max_ratio", 0.2))
    for y in waveforms:
        z = y.astype(np.float32, copy=True)
        z *= rng.uniform(gain_low, gain_high)
        if rng.random() < noise_prob:
            z += rng.normal(0.0, noise_level, size=z.shape).astype(np.float32)
        if rng.random() < mask_prob and z.size > 8:
            max_len = max(1, int(z.size * mask_max_ratio))
            mlen = int(rng.integers(1, max_len + 1))
            start = int(rng.integers(0, max(1, z.size - mlen)))
            z[start : start + mlen] = 0.0
        z = np.clip(z, -1.0, 1.0)
        out.append(z)
    return out


def evaluate_split(
    model: ASTSpeciesClassifier,
    dl: DataLoader,
    classes: list[str],
    device: str,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for waveforms, y in dl:
            x = model.preprocess(waveforms, device=device)
            y = y.to(device)
            pred = model(x).argmax(1)
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
    if not y_true:
        return 0.0, y_true, y_pred
    acc = float((np.array(y_true) == np.array(y_pred)).mean())
    return acc, y_true, y_pred


def save_evaluation_report(
    report_dir: Path,
    split_name: str,
    classes: list[str],
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    if not y_true:
        return {"split": split_name, "samples": 0, "accuracy": 0.0}

    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    np.save(report_dir / f"{split_name}_confusion_matrix.npy", cm)
    with open(report_dir / f"{split_name}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save a readable per-class CSV with precision/recall/f1/support.
    csv_path = report_dir / f"{split_name}_classification_report.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1-score,support\n")
        for c in classes:
            row = report.get(c, {})
            f.write(
                f"{c},{row.get('precision', 0):.4f},{row.get('recall', 0):.4f},"
                f"{row.get('f1-score', 0):.4f},{int(row.get('support', 0))}\n"
            )

    # Plot confusion matrix.
    fig_w = max(8, int(len(classes) * 0.6))
    fig_h = max(6, int(len(classes) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest", cmap="YlGn")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{split_name.capitalize()} Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(report_dir / f"{split_name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    return {
        "split": split_name,
        "samples": len(y_true),
        "accuracy": float((np.array(y_true) == np.array(y_pred)).mean()),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
    }


def load_feedback_corrections(feedback_path: str | Path, class_to_idx: dict[str, int]) -> dict[str, int]:
    path = Path(feedback_path)
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_path = row.get("audio_path")
            correct_raw = row.get("correct_raw")
            if not audio_path or not correct_raw:
                continue
            if correct_raw not in class_to_idx:
                continue
            out[str(Path(audio_path).resolve())] = class_to_idx[correct_raw]
    return out


def git_info() -> dict[str, str | bool]:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""

    commit = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(run(["git", "status", "--porcelain"]))
    return {"commit": commit, "branch": branch, "dirty": dirty}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports") / run_id
    Path(run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "started_at": run_id,
            "config_path": str(Path(args.config).resolve()),
            "cwd": str(Path.cwd()),
            "python": os.sys.version,
            "git": git_info(),
            "argv": ["python", "-m", "forestecho.train", "--config", args.config],
        },
    )
    torch.manual_seed(cfg["train"]["seed"])
    rng = np.random.default_rng(int(cfg["train"]["seed"]))

    ds = AnimalAudioDataset(
        raw_dir=cfg["data"]["raw_dir"],
        sample_rate=cfg["data"]["sample_rate"],
        clip_seconds=cfg["data"]["clip_seconds"],
        min_rms=cfg["data"]["quality"]["min_rms"],
        max_silence_ratio=cfg["data"]["quality"]["max_silence_ratio"],
        max_clipping_ratio=cfg["data"]["quality"]["max_clipping_ratio"],
        silence_amplitude=cfg["data"]["quality"]["silence_amplitude"],
    )
    if len(ds) == 0:
        raise SystemExit(f"No audio found under {cfg['data']['raw_dir']}/<species>/*.wav")
    feedback_cfg = cfg["train"].get("feedback", {"use_queue": False, "path": "data/feedback/corrections.jsonl"})
    if feedback_cfg.get("use_queue", False):
        corr = load_feedback_corrections(feedback_cfg.get("path", "data/feedback/corrections.jsonl"), ds.class_to_idx)
        applied = ds.apply_corrections(corr)
        print(f"feedback_queue: loaded={len(corr)} applied={applied} from={feedback_cfg.get('path')}")

    q = ds.quality_report()
    print(
        "quality_filter:"
        f" kept={q['kept']}/{q['total_seen']}"
        f" rejected={q['rejected_total']}"
        f" reasons={q['rejected']}"
    )
    class_counts = q["class_counts"]
    non_zero = {k: v for k, v in class_counts.items() if v > 0}
    if non_zero:
        min_cls = min(non_zero.values())
        max_cls = max(non_zero.values())
        print(f"class_balance: classes={len(non_zero)} min={min_cls} max={max_cls}")

    n_train, n_val, n_test = split_sizes(
        total=len(ds),
        val_frac=cfg["data"]["val_split"],
        test_frac=cfg["data"]["test_split"],
    )
    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
    )

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_waveforms)
    val_dl = DataLoader(val_ds, batch_size=bs, num_workers=nw, collate_fn=collate_waveforms)
    test_dl = DataLoader(test_ds, batch_size=bs, num_workers=nw, collate_fn=collate_waveforms)

    device = pick_device()
    print(f"device={device} samples={len(ds)} train={n_train} val={n_val} test={n_test}")
    print(f"classes={', '.join(ds.classes)}")
    model = ASTSpeciesClassifier(
        num_classes=len(ds.classes),
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    recipe = cfg["train"]["recipe"]
    use_class_weights = bool(recipe["use_class_weights"])
    class_weights = None
    if use_class_weights:
        class_weights = class_weights_from_subset(train_ds, num_classes=len(ds.classes)).to(device)
        print(f"class_weights: {class_weights.detach().cpu().numpy().round(3).tolist()}")
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(recipe["label_smoothing"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, int(cfg["train"]["epochs"])),
        eta_min=float(recipe["min_lr"]),
    )

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    patience = int(recipe["early_stopping_patience"])
    epochs_without_improve = 0
    aug_cfg = cfg["train"].get("augmentation", {"enabled": False})
    if aug_cfg.get("enabled", False):
        print(
            "augmentation:"
            f" gain=[{aug_cfg.get('gain_min', 0.8)},{aug_cfg.get('gain_max', 1.2)}]"
            f" noise_prob={aug_cfg.get('noise_prob', 0.6)}"
            f" noise_level={aug_cfg.get('noise_level', 0.01)}"
            f" time_mask_prob={aug_cfg.get('time_mask_prob', 0.4)}"
            f" time_mask_max_ratio={aug_cfg.get('time_mask_max_ratio', 0.2)}"
        )

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for waveforms, y in tqdm(train_dl, desc=f"epoch {epoch} train"):
            waveforms = augment_waveforms(waveforms, aug_cfg, rng)
            x = model.preprocess(waveforms, device=device)
            y = y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            train_steps += 1

        model.eval()
        correct = total = 0
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for waveforms, y in val_dl:
                x = model.preprocess(waveforms, device=device)
                y = y.to(device)
                logits = model(x)
                val_loss += float(loss_fn(logits, y).item())
                val_steps += 1
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        mean_loss = train_loss / max(train_steps, 1)
        mean_val_loss = val_loss / max(val_steps, 1)
        current_lr = opt.param_groups[0]["lr"]
        print(
            f"epoch {epoch} train_loss={mean_loss:.4f} "
            f"val_loss={mean_val_loss:.4f} val_acc={acc:.4f} lr={current_lr:.6g}"
        )
        if acc > best_acc:
            best_acc = acc
            epochs_without_improve = 0
            torch.save(
                {"head": model.head.state_dict(), "classes": ds.classes, "cfg": cfg},
                ckpt_dir / "best.pt",
            )
            print(f"saved best checkpoint: {ckpt_dir / 'best.pt'}")
        else:
            epochs_without_improve += 1

        scheduler.step()
        if epochs_without_improve >= patience:
            print(f"early stopping at epoch {epoch} (patience={patience})")
            break

    torch.save(
        {"head": model.head.state_dict(), "classes": ds.classes, "cfg": cfg},
        ckpt_dir / "last.pt",
    )
    print(f"saved last checkpoint: {ckpt_dir / 'last.pt'}")

    # Final evaluation artifacts (using best checkpoint when available).
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.head.load_state_dict(best_state["head"])
        model.eval()

    report_dir = reports_dir
    val_acc, yv_true, yv_pred = evaluate_split(model, val_dl, ds.classes, device)
    test_acc, yt_true, yt_pred = evaluate_split(model, test_dl, ds.classes, device)
    val_summary = save_evaluation_report(report_dir, "val", ds.classes, yv_true, yv_pred)
    test_summary = save_evaluation_report(report_dir, "test", ds.classes, yt_true, yt_pred)

    run_summary = {
        "run_id": run_id,
        "created_at": run_id,
        "checkpoint": str(best_ckpt if best_ckpt.exists() else ckpt_dir / "last.pt"),
        "classes": ds.classes,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "val": val_summary,
        "test": test_summary,
        "report_dir": str(report_dir),
        "run_dir": str(run_dir),
    }
    with open(report_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    write_json(run_dir / "summary.json", run_summary)
    write_json(
        run_dir / "artifacts.json",
        {
            "best_checkpoint": str((ckpt_dir / "best.pt").resolve()) if (ckpt_dir / "best.pt").exists() else "",
            "last_checkpoint": str((ckpt_dir / "last.pt").resolve()),
            "report_dir": str(report_dir.resolve()),
            "report_summary": str((report_dir / "run_summary.json").resolve()),
        },
    )
    latest_run = Path("runs") / "latest.json"
    write_json(latest_run, {"run_id": run_id, "run_dir": str(run_dir.resolve())})
    print(f"saved run registry: {run_dir}")
    print(f"saved evaluation report: {report_dir}")


if __name__ == "__main__":
    main()
