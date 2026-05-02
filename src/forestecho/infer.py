from __future__ import annotations

import argparse

import torch

from .features import load_audio
from .model import ASTSpeciesClassifier


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict(audio_path: str, ckpt_path: str = "models/best.pt", top_k: int = 3):
    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    classes = ckpt["classes"]

    model = ASTSpeciesClassifier(
        num_classes=len(classes),
        pretrained=cfg["model"]["pretrained"],
        freeze_backbone=True,
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.head.load_state_dict(ckpt["head"])
    model.eval()

    y = load_audio(audio_path, cfg["data"]["sample_rate"], cfg["data"]["clip_seconds"])
    x = model.preprocess([y.astype("float32")], device=device)
    with torch.no_grad():
        probs = model(x).softmax(-1)[0].detach().cpu()
    top = torch.topk(probs, k=min(top_k, len(classes)))
    return [(classes[i], float(p)) for p, i in zip(top.values, top.indices)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    for species, p in predict(args.audio, args.ckpt, args.top_k):
        print(f"{species}\t{p:.4f}")


if __name__ == "__main__":
    main()
