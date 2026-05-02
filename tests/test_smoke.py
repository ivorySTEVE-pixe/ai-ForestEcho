import numpy as np

from forestecho.features import mel_spectrogram


def test_mel_shape():
    y = np.random.randn(16000 * 2).astype(np.float32)
    spec = mel_spectrogram(y, 16000, n_mels=64)
    assert spec.shape[0] == 64


# Note: AST model test omitted from the smoke suite — instantiating it
# downloads ~350 MB of weights. Run an integration test manually when needed.
