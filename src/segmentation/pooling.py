"""src/segmentation/pooling.py - Learnable query-based attention pooling and acoustic fusion."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoolingConfig:
    """Configuration for query-based attention pooling and acoustic prosody fusion."""

    hidden_size: int = 1024          # Input token hidden dimension D
    query_proj_dim: int = 256        # Dimension K of attention projection W_p
    adapter_dim: int = 1024          # Output fused dimension d_model for DBCA
    acoustic_dim: int = 4            # Pause (1) + Pitch (1) + Speaker Emb (2)
    speaker_vocab_size: int = 4      # Speaker categories (0: None, 1: Lecturer, 2: Student, 3: Other)
    speaker_emb_dim: int = 2         # Speaker embedding dimension
    dropout: float = 0.1
    eps: float = 1e-5


# Alias for backward compatibility
QueryAttentionPoolingConfig = PoolingConfig


class LearnableQueryAttentionPooling(nn.Module):
    """Computes alpha_{i,t} = softmax(q^T tanh(W_p H_{i,t})), u_i = sum_t alpha_{i,t} H_{i,t}.

    Robustly handles token-level padding masks, variable sequence lengths, and half-precision.
    """

    def __init__(self, config: Optional[PoolingConfig] = None):
        super().__init__()
        self.config = config or PoolingConfig()
        self.W_p = nn.Linear(self.config.hidden_size, self.config.query_proj_dim, bias=True)
        self.q = nn.Parameter(torch.empty(self.config.query_proj_dim))
        self.dropout = nn.Dropout(self.config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_p.weight)
        nn.init.zeros_(self.W_p.bias)
        nn.init.normal_(self.q, mean=0.0, std=1.0 / math.sqrt(self.config.query_proj_dim))

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool token hidden states into a single sentence vector.

        Args:
            token_hidden_states: [..., T, D] token states (e.g., [T, D], [N, T, D], or [B, S, T, D]).
            token_mask: Optional matching shape [..., T] bool or float mask (1 for valid, 0 for pad).

        Returns:
            pooled_reps: [..., D] pooled sentence representations u_i.
            attn_weights: [..., T] query attention distribution alpha_{i,t}.
        """
        shape = token_hidden_states.shape
        T, D = shape[-2], shape[-1]
        prefix_shape = shape[:-2]
        flat_hidden = token_hidden_states.reshape(-1, T, D)
        N = flat_hidden.size(0)

        # 1. Project tokens: tanh(W_p H_{i,t}) -> [N, T, K]
        projected = torch.tanh(self.W_p(flat_hidden))

        # 2. Score with learnable query vector q: [N, T]
        scores = torch.matmul(projected, self.q)

        # 3. Apply token padding mask
        all_masked = None
        if token_mask is not None:
            flat_mask = token_mask.reshape(N, T).bool()
            # If an entire sequence is masked, guard against -inf producing NaN
            all_masked = ~flat_mask.any(dim=-1, keepdim=True)  # [N, 1]
            safe_mask = flat_mask | all_masked
            scores = scores.masked_fill(~safe_mask, -1e4)

        # 4. Numerically stable softmax
        attn_weights = F.softmax(scores, dim=-1)  # [N, T]

        if all_masked is not None:
            # If row was all masked, force attention weights to zero
            attn_weights = attn_weights.masked_fill(all_masked, 0.0)

        # 5. Weighted summation: u_i = sum_t alpha_{i,t} H_{i,t}
        attn_dropped = self.dropout(attn_weights)
        pooled = torch.bmm(attn_dropped.unsqueeze(1), flat_hidden).squeeze(1)  # [N, D]

        # Reshape back to prefix dimensions
        if prefix_shape:
            pooled = pooled.view(*prefix_shape, D)
            attn_weights = attn_weights.view(*prefix_shape, T)
        else:
            pooled = pooled.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return pooled, attn_weights


# Alias for backward compatibility
QueryAttentionPooling = LearnableQueryAttentionPooling


class AcousticProsodyFusion(nn.Module):
    """Concatenates [u_i || pause_before || pitch_reset || speaker_emb] and projects to d_model.

    Provides seamless fallback to pure linguistic projection when acoustic features are omitted.
    """

    def __init__(self, config: Optional[PoolingConfig] = None):
        super().__init__()
        self.config = config or PoolingConfig()
        self.speaker_embedding = nn.Embedding(
            self.config.speaker_vocab_size, self.config.speaker_emb_dim
        )
        total_acoustic_dim = 2 + self.config.speaker_emb_dim  # pause + pitch + speaker_emb

        # Fused projection: [D + total_acoustic_dim] -> adapter_dim
        self.fused_proj = nn.Sequential(
            nn.Linear(self.config.hidden_size + total_acoustic_dim, self.config.adapter_dim),
            nn.LayerNorm(self.config.adapter_dim, eps=self.config.eps),
            nn.Dropout(self.config.dropout),
        )

        # Text-only fallback projection: D -> adapter_dim
        self.text_only_proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.adapter_dim),
            nn.LayerNorm(self.config.adapter_dim, eps=self.config.eps),
            nn.Dropout(self.config.dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.speaker_embedding.weight, std=0.02)
        for m in [self.fused_proj, self.text_only_proj]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(
        self,
        pooled_sentence_reps: torch.Tensor,
        pause_before_ms: Optional[torch.Tensor] = None,
        pitch_reset: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse text sentence representations with optional acoustic prosody markers.

        Args:
            pooled_sentence_reps: [..., D] text sentence vectors u_i.
            pause_before_ms: Optional [...] preceding pause in ms.
            pitch_reset: Optional [...] delta F0 pitch in semitones.
            speaker_ids: Optional [...] speaker category indices (0..speaker_vocab_size-1).

        Returns:
            fused_reps: [..., adapter_dim] projected fused representations.
        """
        # If all acoustic features are None, take direct text-only path
        if pause_before_ms is None and pitch_reset is None and speaker_ids is None:
            return self.text_only_proj(pooled_sentence_reps)

        shape = pooled_sentence_reps.shape
        D = shape[-1]
        prefix_shape = shape[:-1]
        flat_reps = pooled_sentence_reps.reshape(-1, D)
        N = flat_reps.size(0)
        device = pooled_sentence_reps.device
        dtype = pooled_sentence_reps.dtype

        # Normalize pause duration: log(1 + pause_sec)
        if pause_before_ms is not None:
            flat_pause = pause_before_ms.reshape(N).to(dtype=dtype, device=device)
            pause_norm = torch.log1p(torch.clamp(flat_pause, min=0.0) / 1000.0).unsqueeze(-1)
        else:
            pause_norm = torch.zeros(N, 1, device=device, dtype=dtype)

        # Normalize pitch reset: clamp to [-2.0, 2.0]
        if pitch_reset is not None:
            flat_pitch = pitch_reset.reshape(N).to(dtype=dtype, device=device)
            pitch_norm = torch.clamp(flat_pitch / 12.0, min=-2.0, max=2.0).unsqueeze(-1)
        else:
            pitch_norm = torch.zeros(N, 1, device=device, dtype=dtype)

        # Speaker embedding
        if speaker_ids is not None:
            flat_spk = speaker_ids.reshape(N).to(device=device, dtype=torch.long)
            flat_spk = torch.clamp(flat_spk, 0, self.config.speaker_vocab_size - 1)
            spk_emb = self.speaker_embedding(flat_spk).to(dtype=dtype)
        else:
            spk_emb = torch.zeros(N, self.config.speaker_emb_dim, device=device, dtype=dtype)

        # Concatenate: [u_i || pause_norm || pitch_norm || speaker_emb]
        fused_cat = torch.cat([flat_reps, pause_norm, pitch_norm, spk_emb], dim=-1)
        out = self.fused_proj(fused_cat)

        if prefix_shape:
            out = out.view(*prefix_shape, self.config.adapter_dim)
        else:
            out = out.squeeze(0)

        return out


class TriModalDiscourseFusion(nn.Module):
    """Fuses text sentence representations with acoustic prosody and visual slide embeddings.

    Maintains 100% backward compatibility: falls back strictly to AcousticProsodyFusion
    when visual features are omitted.
    """

    def __init__(
        self,
        config: Optional[PoolingConfig] = None,
        vision_dim: int = 1024,
        vision_fusion_alpha: float = 0.35,
    ) -> None:
        super().__init__()
        self.config = config or PoolingConfig()
        self.vision_dim = vision_dim
        self.vision_fusion_alpha = vision_fusion_alpha
        self.acoustic_fusion = AcousticProsodyFusion(self.config)

        # Visual projection: [vision_dim (1024) + visual_shift (1)] -> adapter_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(self.vision_dim + 1, self.config.adapter_dim),
            nn.LayerNorm(self.config.adapter_dim, eps=self.config.eps),
            nn.Dropout(self.config.dropout),
        )
        # Zero-initialize visual projection so fusion starts as exact identity residual
        nn.init.zeros_(self.visual_proj[0].weight)
        nn.init.zeros_(self.visual_proj[0].bias)

        # Learnable Cross-Modal Sigmoid Gate: [adapter_dim (text) + adapter_dim (vis)] -> 1
        self.gate = nn.Sequential(
            nn.Linear(2 * self.config.adapter_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -3.0)

    def forward(
        self,
        pooled_sentence_reps: torch.Tensor,
        pause_before_ms: Optional[torch.Tensor] = None,
        pitch_reset: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        visual_shifts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse text, acoustic prosody, and visual slide features into unified sentence vectors."""
        # 1. Base linguistic + acoustic representation
        h_base = self.acoustic_fusion(
            pooled_sentence_reps,
            pause_before_ms=pause_before_ms,
            pitch_reset=pitch_reset,
            speaker_ids=speaker_ids,
        )

        if visual_embeddings is None and visual_shifts is None:
            return h_base

        device = h_base.device
        dtype = h_base.dtype
        shape = h_base.shape
        prefix_shape = shape[:-1]

        if visual_embeddings is not None:
            v_emb = visual_embeddings.to(dtype=dtype, device=device)
        else:
            v_emb = torch.zeros(*prefix_shape, self.vision_dim, dtype=dtype, device=device)

        if visual_shifts is not None:
            v_shift = visual_shifts.to(dtype=dtype, device=device)
            if v_shift.dim() == len(prefix_shape):
                v_shift = v_shift.unsqueeze(-1)
        else:
            v_shift = torch.zeros(*prefix_shape, 1, dtype=dtype, device=device)

        # Gated residual visual projection with learned cross-modal gating
        v_feat = torch.cat([v_emb, v_shift], dim=-1)
        h_visual = self.visual_proj(v_feat)
        gate_input = torch.cat([h_base, h_visual], dim=-1)
        g_i = self.gate(gate_input)
        fused = h_base + g_i * h_visual
        return fused

