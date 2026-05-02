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

    def __init__(self, raw_dir: str | Path, sample_rate: int, clip_seconds: float):
        self.raw_dir = Path(raw_dir)
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds

        species = sorted(d.name for d in self.raw_dir.iterdir() if d.is_dir())
        self.classes = species
        self.class_to_idx = {c: i for i, c in enumerate(species)}

        self.samples: list[Sample] = []
        for c in species:
            for ext in self.AUDIO_EXTS:
                for p in (self.raw_dir / c).glob(f"*{ext}"):
                    self.samples.append(Sample(p, self.class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        s = self.samples[idx]
        y = load_audio(str(s.path), self.sample_rate, self.clip_seconds)
        return y.astype(np.float32), s.label


def collate_waveforms(batch):
    """Stack variable items as a list of waveforms; processor batches them."""
    waveforms = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return waveforms, labels
