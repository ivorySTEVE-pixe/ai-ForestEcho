from __future__ import annotations

import argparse

import numpy as np
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


def _window_waveform(y: np.ndarray, sr: int, clip_seconds: float, hop_seconds: float) -> list[np.ndarray]:
    clip = max(1, int(sr * clip_seconds))
    hop = max(1, int(sr * hop_seconds))
    if len(y) <= clip:
        if len(y) < clip:
            y = np.pad(y, (0, clip - len(y)))
        return [y.astype("float32")]
    out = []
    for start in range(0, len(y) - clip + 1, hop):
        out.append(y[start : start + clip].astype("float32"))
    return out


def predict_long(
    audio_path: str,
    ckpt_path: str = "models/best.pt",
    top_k: int = 3,
    hop_seconds: float = 1.5,
    aggregation: str = "mean",
):
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

    sr = int(cfg["data"]["sample_rate"])
    clip_seconds = float(cfg["data"]["clip_seconds"])
    y = load_audio(audio_path, sr, clip_seconds=None).astype("float32")
    windows = _window_waveform(y, sr, clip_seconds=clip_seconds, hop_seconds=hop_seconds)
    x = model.preprocess(windows, device=device)
    with torch.no_grad():
        probs = model(x).softmax(-1).detach().cpu()  # [num_windows, num_classes]

    mode = aggregation.lower().strip()
    if mode == "max":
        agg = probs.max(dim=0).values
    elif mode == "vote":
        votes = torch.zeros(probs.shape[1], dtype=torch.float32)
        winners = probs.argmax(dim=1)
        for w in winners:
            votes[int(w)] += 1.0
        agg = votes / max(1, probs.shape[0])
    else:
        agg = probs.mean(dim=0)

    top = torch.topk(agg, k=min(top_k, len(classes)))
    return [(classes[i], float(p)) for p, i in zip(top.values, top.indices)]


def apply_unknown_threshold(
    results: list[tuple[str, float]],
    threshold: float,
) -> tuple[bool, list[tuple[str, float]]]:
    if not results:
        return True, []
    return results[0][1] < threshold, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ckpt", default="models/best.pt")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--long-audio", action="store_true")
    ap.add_argument("--hop-seconds", type=float, default=1.5)
    ap.add_argument("--aggregation", choices=["mean", "max", "vote"], default="mean")
    ap.add_argument("--unknown-threshold", type=float, default=0.35)
    args = ap.parse_args()
    fn = predict_long if args.long_audio else predict
    kwargs = (
        {
            "audio_path": args.audio,
            "ckpt_path": args.ckpt,
            "top_k": args.top_k,
            "hop_seconds": args.hop_seconds,
            "aggregation": args.aggregation,
        }
        if args.long_audio
        else {"audio_path": args.audio, "ckpt_path": args.ckpt, "top_k": args.top_k}
    )
    unknown, out = apply_unknown_threshold(fn(**kwargs), threshold=args.unknown_threshold)
    if unknown:
        print(f"unknown\t<{args.unknown_threshold:.2f}")
    for species, p in out:
        print(f"{species}\t{p:.4f}")


if __name__ == "__main__":
    main()
