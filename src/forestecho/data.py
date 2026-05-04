from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import load_audio


@dataclass
class Sample:
    path: Path
    label: int


class AnimalAudioDataset(Dataset):
    """Returns raw waveforms (1-D float32) and labels.

    Expected layout: raw_dir/<species_name>/*.wav
    Feature extraction is handled by the pretrained model's processor.
    """

    AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")

    def __init__(
        self,
        raw_dir: str | Path,
        sample_rate: int,
        clip_seconds: float,
        min_rms: float = 0.005,
        max_silence_ratio: float = 0.98,
        max_clipping_ratio: float = 0.20,
        silence_amplitude: float = 0.003,
    ):
        self.raw_dir = Path(raw_dir)
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.min_rms = float(min_rms)
        self.max_silence_ratio = float(max_silence_ratio)
        self.max_clipping_ratio = float(max_clipping_ratio)
        self.silence_amplitude = float(silence_amplitude)

        species = sorted(d.name for d in self.raw_dir.iterdir() if d.is_dir())
        self.classes = species
        self.class_to_idx = {c: i for i, c in enumerate(species)}

        self.total_seen = 0
        self.rejected: dict[str, int] = {"low_rms": 0, "too_silent": 0, "too_clipped": 0, "load_error": 0}
        self.correction_map: dict[str, int] = {}
        self.correction_applied = 0
        self.samples: list[Sample] = []
        for c in species:
            for ext in self.AUDIO_EXTS:
                for p in (self.raw_dir / c).glob(f"*{ext}"):
                    self.total_seen += 1
                    ok, reason = self._is_valid_clip(p)
                    if ok:
                        self.samples.append(Sample(p, self.class_to_idx[c]))
                    elif reason is not None:
                        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def _is_valid_clip(self, path: Path) -> tuple[bool, str | None]:
        try:
            y = load_audio(str(path), self.sample_rate, self.clip_seconds).astype(np.float32)
        except Exception:
            return False, "load_error"
        if y.size == 0:
            return False, "load_error"
        rms = float(np.sqrt(np.mean(np.square(y))))
        if rms < self.min_rms:
            return False, "low_rms"
        silence_ratio = float(np.mean(np.abs(y) < self.silence_amplitude))
        if silence_ratio > self.max_silence_ratio:
            return False, "too_silent"
        clipping_ratio = float(np.mean(np.abs(y) >= 0.99))
        if clipping_ratio > self.max_clipping_ratio:
            return False, "too_clipped"
        return True, None

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for s in self.samples:
            counts[self.classes[s.label]] += 1
        return counts

    def quality_report(self) -> dict:
        kept = len(self.samples)
        rejected_total = self.total_seen - kept
        return {
            "total_seen": self.total_seen,
            "kept": kept,
            "rejected_total": rejected_total,
            "rejected": dict(self.rejected),
            "class_counts": self.class_counts(),
            "corrections_applied": self.correction_applied,
        }

    def apply_corrections(self, path_to_label: dict[str, int]) -> int:
        """Apply corrected labels from feedback queue.

        `path_to_label` maps absolute/relative file path string -> label index.
        """
        norm = {str(Path(k).resolve()): int(v) for k, v in path_to_label.items()}
        applied = 0
        self.correction_map = {}
        for s in self.samples:
            key = str(s.path.resolve())
            if key in norm and 0 <= norm[key] < len(self.classes):
                self.correction_map[key] = norm[key]
                applied += 1
        self.correction_applied = applied
        return applied

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        s = self.samples[idx]
        y = load_audio(str(s.path), self.sample_rate, self.clip_seconds)
        label = self.correction_map.get(str(s.path.resolve()), s.label)
        return y.astype(np.float32), int(label)


def collate_waveforms(batch):
    """Stack variable items as a list of waveforms; processor batches them."""
    waveforms = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return waveforms, labels
