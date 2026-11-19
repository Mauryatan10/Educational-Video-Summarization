"""src/segmentation/dbca_adapter.py - 2-Layer Decoupled Bidirectional Context Adapter (DBCA)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DBCAConfig:
    """Hyperparameters for Decoupled Bidirectional Context Adapter."""

    hidden_size: int = 1024          # H = 1024
    num_heads: int = 8               # heads = 8 (head_dim = 128)
    d_ff: int = 4096                 # d_ff = 4096
    num_layers: int = 2              # 2-layer transformer encoder
    max_sentences: int = 1024        # Max supported sentences in a lecture
    dropout: float = 0.1
    eps: float = 1e-5


# Alias for backward compatibility
DBCAAdapterConfig = DBCAConfig


class DBCABlock(nn.Module):
    """Pre-normalized Bidirectional Transformer Encoder Layer (NormFirst)."""

    def __init__(self, config: DBCAConfig):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(config.dropout)

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.hidden_size),
        )
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for single DBCA block.

        Args:
            x: [B, S, H] sentence context representations.
            key_padding_mask: [B, S] bool tensor (True for ignored/pad positions).

        Returns:
            [B, S, H] contextual representations.
        """
        # 1. Multihead Bidirectional Self-Attention (Pre-Norm)
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # 2. Feedforward Network (Pre-Norm)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class DecoupledBidirectionalContextAdapter(nn.Module):
    """2-Layer DBCA bridging causal LLM future discourse blindness across full lecture.

    Receives the sequence of sentence representations [x_1, ..., x_N] and routes them
    through a 2-layer unmasked bidirectional Transformer stack (H=1024, heads=8, d_ff=4096),
    enabling each sentence to contextualize against both preceding and succeeding discourse.
    """

    def __init__(self, config: Optional[DBCAConfig] = None):
        super().__init__()
        self.config = config or DBCAConfig()
        self.pos_embedding = nn.Embedding(self.config.max_sentences, self.config.hidden_size)
        self.layers = nn.ModuleList([DBCABlock(self.config) for _ in range(self.config.num_layers)])
        self.final_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.eps)
        self.dropout = nn.Dropout(self.config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        sentence_representations: torch.Tensor,
        sentence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the DBCA stack.

        Args:
            sentence_representations: [B, S, H] or [S, H] fused sentence vectors.
            sentence_mask: Optional [B, S] or [S] bool tensor (True for valid sentences, False for padding).

        Returns:
            contextual_states: [B, S, H] or [S, H] bidirectional contextual representations.
        """
        is_2d = sentence_representations.dim() == 2
        if is_2d:
            sentence_representations = sentence_representations.unsqueeze(0)  # [1, S, H]
            if sentence_mask is not None:
                sentence_mask = sentence_mask.unsqueeze(0)  # [1, S]

        B, S, H = sentence_representations.shape
        device = sentence_representations.device

        # Learned sentence positional embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        positions = torch.clamp(positions, 0, self.config.max_sentences - 1)
        x = sentence_representations + self.pos_embedding(positions)
        x = self.dropout(x)

        # PyTorch MHA key_padding_mask: True = IGNORED (pad), False = ATTENDED (valid)
        key_padding_mask = None
        if sentence_mask is not None:
            key_padding_mask = ~sentence_mask.bool()
            # If a sequence in the batch is entirely masked, unmask the first element to avoid NaN
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                key_padding_mask = key_padding_mask & ~all_masked

        # Route through 2-layer bidirectional stack
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        out = self.final_norm(x)
        # Residual skip connection preserves local semantic distinction and prevents over-smoothing
        out = out + sentence_representations

        # Zero out padding representations if mask provided
        if sentence_mask is not None:
            out = out * sentence_mask.unsqueeze(-1).to(dtype=out.dtype)

        return out.squeeze(0) if is_2d else out


# Alias for backward compatibility
DBCAAdapter = DecoupledBidirectionalContextAdapter
