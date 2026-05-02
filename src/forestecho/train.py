from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg["train"]["seed"])

    ds = AnimalAudioDataset(
        raw_dir=cfg["data"]["raw_dir"],
        sample_rate=cfg["data"]["sample_rate"],
        clip_seconds=cfg["data"]["clip_seconds"],
    )
    if len(ds) == 0:
        raise SystemExit(f"No audio found under {cfg['data']['raw_dir']}/<species>/*.wav")

    n_train, n_val, n_test = split_sizes(
        total=len(ds),
        val_frac=cfg["data"]["val_split"],
        test_frac=cfg["data"]["test_split"],
    )
    train_ds, val_ds, _ = random_split(
        ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
    )

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_waveforms)
    val_dl = DataLoader(val_ds, batch_size=bs, num_workers=nw, collate_fn=collate_waveforms)

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
    loss_fn = torch.nn.CrossEntropyLoss()

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for waveforms, y in tqdm(train_dl, desc=f"epoch {epoch} train"):
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
        with torch.no_grad():
            for waveforms, y in val_dl:
                x = model.preprocess(waveforms, device=device)
                y = y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        mean_loss = train_loss / max(train_steps, 1)
        print(f"epoch {epoch} train_loss={mean_loss:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {"head": model.head.state_dict(), "classes": ds.classes, "cfg": cfg},
                ckpt_dir / "best.pt",
            )
            print(f"saved best checkpoint: {ckpt_dir / 'best.pt'}")

    torch.save(
        {"head": model.head.state_dict(), "classes": ds.classes, "cfg": cfg},
        ckpt_dir / "last.pt",
    )
    print(f"saved last checkpoint: {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
