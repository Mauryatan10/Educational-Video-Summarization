"""src/segmentation/model.py - Unified DiscourseAwareSegmentationModel combining all Phase 2 components."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.segmentation.dbca_adapter import DBCAConfig, DecoupledBidirectionalContextAdapter
from src.segmentation.dual_head import (
    ContrastiveProjectionHead,
    DiscriminativeTransitionHead,
    DualHeadConfig,
    GenerativeTitleHead,
)
from src.segmentation.losses import MultiTaskSegmentationLoss, SegmentationLossOutput
from src.segmentation.nms import Boundary1DNMS, non_maximum_suppression_1d
from src.segmentation.pooling import (
    AcousticProsodyFusion,
    LearnableQueryAttentionPooling,
    PoolingConfig,
    TriModalDiscourseFusion,
)
from src.segmentation.modernbert_extractor import ModernBERTSentenceRepresentationExtractor


@dataclass
class DiscourseAwareSegmentationConfig:
    """Comprehensive configuration for Unified Discourse-Aware Segmentation Model."""

    hidden_size: int = 768           # Token hidden size D from ModernBERT backbone
    adapter_dim: int = 1024          # DBCA hidden dimension H
    query_proj_dim: int = 256        # Query projection dimension K
    num_heads: int = 8               # DBCA attention heads
    d_ff: int = 4096                 # DBCA feedforward dimension
    num_layers: int = 2              # DBCA layer count
    max_sentences: int = 1024        # Max supported sentences
    dropout: float = 0.1
    transition_hidden: int = 512     # Transition MLP hidden dimension
    contrastive_dim: int = 256       # InfoNCE hypersphere dimension
    llm_hidden_size: int = 2048      # Generative title head dimension
    nms_threshold: float = 0.35      # Probability threshold for 1D-NMS
    nms_radius: int = 5              # Elimination radius for 1D-NMS (realistic topic segment length)
    lambda_focal: float = 10.0       # Weight for Soft-Margin Focal Loss
    lambda_contrastive: float = 0.05 # Rebalanced contrastive loss weight (85%+ focal boundary gradient)
    lambda_generative: float = 0.5   # Weight for Generative Title Loss
    sigma: float = 0.5               # Focal loss Gaussian smoothing bandwidth
    gamma: float = 2.0               # Focal loss modulating factor
    alpha: float = 0.25              # Focal loss class weight (0.25 penalizes false alarms)
    tau: float = 0.07                # InfoNCE temperature
    tau_temp: float = 10.0           # Temporal decay bandwidth
    vision_dim: int = 1024           # Native Qwen 3.5 Vision Tower hidden dimension
    vision_fusion_alpha: float = 0.35 # Gated residual visual blending factor
    use_vision: bool = True          # Enable multimodal visual fusion if visual tensors supplied


# Alias for backward compatibility
SegmentationModelConfig = DiscourseAwareSegmentationConfig


@dataclass
class SegmentationModelOutput:
    """Strongly typed output container for DiscourseAwareSegmentationModel."""

    loss: Optional[torch.Tensor] = None
    loss_output: Optional[SegmentationLossOutput] = None
    loss_dict: Optional[Dict[str, torch.Tensor]] = None
    boundary_logits: Optional[torch.Tensor] = None
    boundary_probabilities: Optional[torch.Tensor] = None
    predicted_boundaries: Optional[List[List[int]]] = None
    contextual_states: Optional[torch.Tensor] = None
    contrastive_reps: Optional[torch.Tensor] = None
    pooled_sentence_reps: Optional[torch.Tensor] = None


class DiscourseAwareSegmentationModel(nn.Module):
    """Unified Discourse-Aware Topic Segmentation Model uniting:
    1. Learnable query-based attention token pooling
    2. Acoustic prosody fusion (pause, pitch reset, speaker identity)
    3. 2-Layer Decoupled Bidirectional Context Adapter (DBCA)
    4. Heterogeneous dual heads (Discriminative transition head + Generative title head)
    5. Calibrated multi-task loss management (Focal, InfoNCE, Generative)
    6. 1D Non-Maximum Suppression inference engine (radius r=2)
    """

    def __init__(self, config: Optional[DiscourseAwareSegmentationConfig] = None):
        super().__init__()
        self.config = config or DiscourseAwareSegmentationConfig()

        # 1. Pooling & Acoustic Fusion
        pool_cfg = PoolingConfig(
            hidden_size=self.config.hidden_size,
            query_proj_dim=self.config.query_proj_dim,
            adapter_dim=self.config.adapter_dim,
            dropout=self.config.dropout,
        )
        self.pooling = LearnableQueryAttentionPooling(pool_cfg)
        self.acoustic_fusion = AcousticProsodyFusion(pool_cfg)
        self.trimodal_fusion = TriModalDiscourseFusion(
            pool_cfg,
            vision_dim=self.config.vision_dim,
            vision_fusion_alpha=self.config.vision_fusion_alpha,
        )

        # 2. Decoupled Bidirectional Context Adapter
        dbca_cfg = DBCAConfig(
            hidden_size=self.config.adapter_dim,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            num_layers=self.config.num_layers,
            max_sentences=self.config.max_sentences,
            dropout=self.config.dropout,
        )
        self.dbca = DecoupledBidirectionalContextAdapter(dbca_cfg)

        # 3. Dual Heads & Contrastive Projection
        head_cfg = DualHeadConfig(
            adapter_dim=self.config.adapter_dim,
            transition_hidden=self.config.transition_hidden,
            contrastive_dim=self.config.contrastive_dim,
            llm_hidden_size=self.config.llm_hidden_size,
            dropout=self.config.dropout,
        )
        self.boundary_head = DiscriminativeTransitionHead(head_cfg)
        self.contrastive_head = ContrastiveProjectionHead(head_cfg)
        self.generative_head = GenerativeTitleHead(head_cfg)

        # 4. Multi-Task Loss Function
        self.loss_fn = MultiTaskSegmentationLoss(
            lambda_focal=self.config.lambda_focal,
            lambda_contrastive=self.config.lambda_contrastive,
            lambda_generative=self.config.lambda_generative,
            sigma=self.config.sigma,
            gamma=self.config.gamma,
            alpha=self.config.alpha,
            tau=self.config.tau,
            tau_temp=self.config.tau_temp,
        )

        # 5. 1D-NMS Boundary Decoder
        self.nms = Boundary1DNMS(
            threshold=self.config.nms_threshold,
            radius=self.config.nms_radius,
        )

    def forward(
        self,
        token_hidden_states: Optional[torch.Tensor] = None,
        sentence_reps: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        sentence_mask: Optional[torch.Tensor] = None,
        pause_before_ms: Optional[torch.Tensor] = None,
        pitch_reset: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        visual_shifts: Optional[torch.Tensor] = None,
        gold_boundaries: Optional[Union[Set[int], List[int], torch.Tensor]] = None,
        topic_segment_ids: Optional[Union[List[int], torch.Tensor]] = None,
        generative_loss: Optional[torch.Tensor] = None,
    ) -> SegmentationModelOutput:
        """Forward pass for DiscourseAwareSegmentationModel.

        Args:
            token_hidden_states: Optional [B, S, T, D] token hidden states from LLM backbone.
            sentence_reps: Optional [B, S, D] pre-pooled sentence representations.
            token_mask: Optional [B, S, T] token padding mask.
            sentence_mask: Optional [B, S] sentence validity mask.
            pause_before_ms: Optional [B, S] silence interval preceding sentences.
            pitch_reset: Optional [B, S] fundamental frequency delta.
            speaker_ids: Optional [B, S] speaker tags.
            gold_boundaries: Optional gold boundary labels for loss computation.
            topic_segment_ids: Optional integer segment IDs for contrastive InfoNCE.
            generative_loss: Optional scalar cross-entropy loss from Head B.

        Returns:
            SegmentationModelOutput containing predictions, loss components, and representations.
        """
        # Step 1: Obtain base sentence representations
        if sentence_reps is not None:
            pooled_reps = sentence_reps
        elif token_hidden_states is not None:
            pooled_reps, _ = self.pooling(token_hidden_states, token_mask=token_mask)
        else:
            raise ValueError("Either token_hidden_states or sentence_reps must be provided.")

        is_2d = pooled_reps.dim() == 2
        if is_2d:
            pooled_reps = pooled_reps.unsqueeze(0)  # [1, S, D]
            if sentence_mask is not None:
                sentence_mask = sentence_mask.unsqueeze(0)
            if pause_before_ms is not None:
                pause_before_ms = pause_before_ms.unsqueeze(0)
            if pitch_reset is not None:
                pitch_reset = pitch_reset.unsqueeze(0)
            if speaker_ids is not None:
                speaker_ids = speaker_ids.unsqueeze(0)
            if visual_embeddings is not None and visual_embeddings.dim() == 2:
                visual_embeddings = visual_embeddings.unsqueeze(0)
            if visual_shifts is not None and visual_shifts.dim() == 1:
                visual_shifts = visual_shifts.unsqueeze(0)

        B, S, _ = pooled_reps.shape
        device = pooled_reps.device

        if sentence_mask is None:
            sentence_mask = torch.ones(B, S, device=device, dtype=torch.bool)

        # Step 2: Acoustic-Prosodic & Visual Multimodal Fusion [B, S, D] -> [B, S, H]
        if visual_embeddings is not None or visual_shifts is not None:
            fused_reps = self.trimodal_fusion(
                pooled_reps,
                pause_before_ms=pause_before_ms,
                pitch_reset=pitch_reset,
                speaker_ids=speaker_ids,
                visual_embeddings=visual_embeddings,
                visual_shifts=visual_shifts,
            )
        else:
            fused_reps = self.acoustic_fusion(
                pooled_reps,
                pause_before_ms=pause_before_ms,
                pitch_reset=pitch_reset,
                speaker_ids=speaker_ids,
            )

        # Step 3: Decoupled Bidirectional Context Adapter [B, S, H] -> [B, S, H]
        contextual_states = self.dbca(fused_reps, sentence_mask=sentence_mask)

        # Step 4: Heterogeneous Heads
        # Head A: Boundary classification over 4H features, pad_first=True yields [B, S] logits
        boundary_logits = self.boundary_head(contextual_states, pad_first=True)
        # Head B contrastive projection: [B, S, contrastive_dim]
        contrastive_reps = self.contrastive_head(contextual_states)

        boundary_probs = torch.sigmoid(boundary_logits)
        boundary_probs = boundary_probs * sentence_mask.float()

        # Step 5: Multi-Task Loss Computation (if ground truth provided)
        total_loss = None
        loss_output = None
        loss_dict = None
        if gold_boundaries is not None and topic_segment_ids is not None:
            # Mask position 0 (not a candidate) and any padding sentences
            loss_mask = sentence_mask.bool().clone()
            loss_mask[:, 0] = False

            loss_output = self.loss_fn(
                boundary_logits=boundary_logits,
                gold_boundaries=gold_boundaries,
                sentence_reps=contrastive_reps,
                segment_labels=topic_segment_ids,
                generative_loss=generative_loss,
                mask=loss_mask,
            )
            total_loss = loss_output.total
            loss_dict = {
                "loss_total": loss_output.total,
                "loss_focal": loss_output.focal,
                "loss_contrastive": loss_output.contrastive,
                "loss_generative": loss_output.generative,
            }

        # Step 6: 1D-NMS Boundary Detection for Inference
        predicted_boundaries: List[List[int]] = []
        for b in range(B):
            valid_len = int(sentence_mask[b].sum().item())
            valid_probs = boundary_probs[b, :valid_len]
            bounds = non_maximum_suppression_1d(
                valid_probs,
                threshold=self.config.nms_threshold,
                radius=self.config.nms_radius,
            )
            predicted_boundaries.append(bounds)

        if is_2d:
            boundary_logits = boundary_logits.squeeze(0)
            boundary_probs = boundary_probs.squeeze(0)
            contextual_states = contextual_states.squeeze(0)
            contrastive_reps = contrastive_reps.squeeze(0)
            pooled_reps = pooled_reps.squeeze(0)

        return SegmentationModelOutput(
            loss=total_loss,
            loss_output=loss_output,
            loss_dict=loss_dict,
            boundary_logits=boundary_logits,
            boundary_probabilities=boundary_probs,
            predicted_boundaries=predicted_boundaries,
            contextual_states=contextual_states,
            contrastive_reps=contrastive_reps,
            pooled_sentence_reps=pooled_reps,
        )

    def predict_boundary_probabilities(
        self,
        sentence_reps: torch.Tensor,
        pause_before_ms: Optional[torch.Tensor] = None,
        pitch_reset: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference helper: compute boundary transition probabilities."""
        self.eval()
        with torch.no_grad():
            output = self.forward(
                sentence_reps=sentence_reps,
                pause_before_ms=pause_before_ms,
                pitch_reset=pitch_reset,
                speaker_ids=speaker_ids,
            )
            return output.boundary_probabilities
