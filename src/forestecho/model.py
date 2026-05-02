from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTModel


class ASTSpeciesClassifier(nn.Module):
    """AST backbone (pretrained on AudioSet) + linear classifier head.

    Input: raw waveform tensor (B, T) at 16 kHz.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained)
        self.backbone = ASTModel.from_pretrained(pretrained)
        hidden = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.sample_rate = self.feature_extractor.sampling_rate

    def preprocess(self, waveforms: list, device: torch.device | str = "cpu") -> torch.Tensor:
        """Convert a list/array of 1-D waveforms into AST input_values."""
        inputs = self.feature_extractor(
            waveforms, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        return inputs["input_values"].to(device)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_values=input_values)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(1)
        return self.head(pooled)
