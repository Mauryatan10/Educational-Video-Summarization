from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class RobertaBoundaryConfig:
    backbone_name: str = "roberta-base"
    dropout: float = 0.1
    classifier_hidden: int = 256

    def to_dict(self) -> Dict:
        return asdict(self)


class RobertaBoundaryModel(nn.Module):
    def __init__(self, config: RobertaBoundaryConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.backbone_name)
        hidden_size = int(self.encoder.config.hidden_size)
        hidden_mid = max(32, min(config.classifier_hidden, hidden_size))

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, hidden_mid),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_mid, 1),
        )

    def _pooled_output(self, encoder_outputs) -> torch.Tensor:
        # RoBERTa generally relies on the first token representation.
        pooler_output = getattr(encoder_outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        return encoder_outputs.last_hidden_state[:, 0]

    def forward(self, **inputs) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        pooled = self._pooled_output(outputs)
        logits = self.classifier(pooled).squeeze(-1)
        return logits
