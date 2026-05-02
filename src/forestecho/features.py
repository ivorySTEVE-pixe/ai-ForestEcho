from __future__ import annotations

import numpy as np
import librosa


def load_audio(path: str, sample_rate: int, clip_seconds: float | None = None) -> np.ndarray:
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    if clip_seconds is not None:
        target = int(sample_rate * clip_seconds)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
    return y


def mel_spectrogram(
    y: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int | None = None,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    return librosa.power_to_db(mel, ref=np.max)
