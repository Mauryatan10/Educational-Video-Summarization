"""src/segmentation/losses.py - Logit-Space Soft-Margin Focal Loss and SupCon InfoNCE."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Helper Functional Implementations
# ==============================================================================

def gaussian_label_smoothing(
    gold_boundaries: Union[Set[int], List[int], torch.Tensor],
    num_candidates: int,
    sigma: float = 0.5,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute Gaussian label-smoothed target vector y^* in [0, 1] with bandwidth sigma=0.5.

    y_i^* = max_{j in Gold} exp( - (i - j)^2 / (2 * sigma^2) )
    """
    if num_candidates <= 0:
        return torch.zeros(0, device=device, dtype=dtype)

    gold_list: List[int] = []
    if isinstance(gold_boundaries, set):
        gold_list = [int(j) for j in gold_boundaries if 0 <= j < num_candidates]
    elif isinstance(gold_boundaries, (list, tuple)):
        if (
            len(gold_boundaries) == num_candidates
            and all(x in (0, 1, True, False) for x in gold_boundaries)
            and max(gold_boundaries, default=0) <= 1
            and num_candidates > 2
        ):
            gold_list = [i for i, x in enumerate(gold_boundaries) if x]
        else:
            gold_list = [int(j) for j in gold_boundaries if 0 <= j < num_candidates]
    elif isinstance(gold_boundaries, torch.Tensor):
        if gold_boundaries.dtype == torch.bool:
            gold_list = torch.nonzero(gold_boundaries).squeeze(-1).tolist()
        elif (
            gold_boundaries.numel() == num_candidates
            and torch.all((gold_boundaries == 0) | (gold_boundaries == 1))
            and num_candidates > 2
        ):
            gold_list = torch.nonzero(gold_boundaries > 0.5).squeeze(-1).tolist()
        else:
            gold_list = [int(x) for x in gold_boundaries.tolist() if 0 <= x < num_candidates]

    if not gold_list:
        return torch.zeros(num_candidates, device=device, dtype=dtype)

    indices = torch.arange(num_candidates, device=device, dtype=dtype)  # [N]
    golds = torch.tensor(gold_list, device=device, dtype=dtype).unsqueeze(1)  # [G, 1]

    # Pairwise squared distances: [G, N]
    dist_sq = (indices.unsqueeze(0) - golds) ** 2
    min_dist_sq = torch.min(dist_sq, dim=0).values  # [N]

    smoothed = torch.exp(-min_dist_sq / (2.0 * (sigma ** 2)))
    return smoothed


def compute_soft_margin_focal_loss(
    logits: torch.Tensor,
    gold_boundaries: Union[Set[int], List[int], torch.Tensor],
    sigma: float = 0.5,
    gamma: float = 2.0,
    alpha: float = 0.25,
    clamp_val: float = 50.0,
    eps: float = 1e-7,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute logit-space Soft-Margin Focal Loss in functional form."""
    loss_fn = SoftMarginFocalLoss(sigma=sigma, gamma=gamma, alpha=alpha, clamp_val=clamp_val, eps=eps)
    return loss_fn(logits, gold_boundaries, mask=mask)


def compute_contrastive_infonce_loss(
    sentence_reps: torch.Tensor,
    segment_labels: Union[List[int], torch.Tensor],
    tau: float = 0.07,
    tau_temp: float = 10.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute temporally decayed SupCon InfoNCE loss in functional form."""
    loss_fn = TemporallyDecayedSupConLoss(tau=tau, tau_temp=tau_temp)
    return loss_fn(sentence_reps, segment_labels, mask=mask)


# ==============================================================================
# Container for Multi-Task Loss Outputs
# ==============================================================================

@dataclass
class SegmentationLossOutput:
    """Strongly typed container for multi-task segmentation loss components."""

    total: torch.Tensor
    focal: torch.Tensor
    contrastive: torch.Tensor
    generative: torch.Tensor
    weights: Dict[str, float]

    def to_metrics(self) -> Dict[str, float]:
        return {
            "loss_total": float(self.total.item()),
            "loss_focal": float(self.focal.item()),
            "loss_contrastive": float(self.contrastive.item()),
            "loss_generative": float(self.generative.item()),
        }


# ==============================================================================
# 1. Logit-Space Soft-Margin Focal Loss
# ==============================================================================

class SoftMarginFocalLoss(nn.Module):
    """Logit-space numerically stable Soft-Margin Focal Loss with tightened Gaussian smoothing (sigma=0.5).

    Mathematical specification:
        y_i^* = max_{j in Gold} exp( - (i - j)^2 / (2 * sigma^2) )
        L_i = - [ alpha * y_i^* * sigma(-z_i)^gamma * log_sigmoid(z_i) +
                  (1 - alpha) * (1 - y_i^*) * sigma(z_i)^gamma * log_sigmoid(-z_i) ]
    """

    def __init__(
        self,
        sigma: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.25,
        eps: float = 1e-7,
        clamp_val: float = 50.0,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.clamp_val = clamp_val

    def compute_smoothed_targets(
        self,
        gold_boundaries: Union[Set[int], List[int], torch.Tensor],
        num_candidates: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute Gaussian label-smoothed target vector y^* in [0, 1]."""
        return gaussian_label_smoothing(
            gold_boundaries=gold_boundaries,
            num_candidates=num_candidates,
            sigma=self.sigma,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        logits: torch.Tensor,
        gold_boundaries: Union[Set[int], List[int], torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute stable Soft-Margin Focal Loss.

        Args:
            logits: Transition logits z_i of shape [N] or [B, N].
            gold_boundaries: Gold boundary index set/list for unbatched [N], or [B, N] binary tensor.
            mask: Optional binary validity mask of matching shape.

        Returns:
            Scalar focal loss tensor.
        """
        if logits.numel() == 0:
            return logits.new_zeros(())

        is_batched = logits.dim() == 2
        if not is_batched:
            logits = logits.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N = logits.shape
        device = logits.device

        # Compute smoothed Gaussian targets
        if isinstance(gold_boundaries, torch.Tensor) and gold_boundaries.dim() == 2:
            smoothed_targets = torch.zeros((B, N), device=device, dtype=logits.dtype)
            for b in range(B):
                smoothed_targets[b] = self.compute_smoothed_targets(gold_boundaries[b], N, device, logits.dtype)
        else:
            targets_1d = self.compute_smoothed_targets(gold_boundaries, N, device, logits.dtype)
            smoothed_targets = targets_1d.unsqueeze(0).expand(B, -1)

        # Numerical stabilization clamp: prevent FP16 logit overflow
        z = torch.clamp(logits, min=-self.clamp_val, max=self.clamp_val)

        # Logit-space log-sigmoid evaluations
        log_sig_z = F.logsigmoid(z)       # log(sigma(z))
        log_sig_neg_z = F.logsigmoid(-z)   # log(sigma(-z)) = log(1 - sigma(z))

        # Modulating focal factors: (1 - p_t)^gamma
        sig_z = torch.sigmoid(z).clamp(min=self.eps, max=1.0)
        sig_neg_z = torch.sigmoid(-z).clamp(min=self.eps, max=1.0)

        focal_pos = sig_neg_z.pow(self.gamma)
        focal_neg = sig_z.pow(self.gamma)

        # Individual focal terms
        term_pos = self.alpha * smoothed_targets * focal_pos * log_sig_z
        term_neg = (1.0 - self.alpha) * (1.0 - smoothed_targets) * focal_neg * log_sig_neg_z
        per_element_loss = -(term_pos + term_neg)

        # Apply validity mask if supplied
        if mask is not None:
            valid_mask = mask.float()
            denom = valid_mask.sum().clamp_min(1.0)
            loss = (per_element_loss * valid_mask).sum() / denom
        else:
            loss = per_element_loss.mean()

        return loss.to(logits.dtype)


# ==============================================================================
# 2. Supervised Contrastive InfoNCE Loss with Temporal Decay
# ==============================================================================

class TemporallyDecayedSupConLoss(nn.Module):
    """In-document Supervised Contrastive (SupCon) InfoNCE loss with exponential temporal decay:

    w_{ij} = exp(-|i - j| / tau_temp) with tau_temp=10.0, tau=0.07.
    Includes single-sentence segment protection (preventing 0/0 division).
    """

    def __init__(
        self,
        tau: float = 0.07,
        tau_temp: float = 10.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.tau_temp = tau_temp
        self.eps = eps

    def forward(
        self,
        sentence_reps: torch.Tensor,
        segment_labels: Union[List[int], torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute temporally decayed SupCon InfoNCE loss.

        Args:
            sentence_reps: [N, D] or [B, N, D] sentence representations.
            segment_labels: List of length N or [N] / [B, N] integer segment IDs.
            mask: Optional validity mask.

        Returns:
            Scalar contrastive loss.
        """
        if sentence_reps.numel() == 0:
            return sentence_reps.new_zeros(())

        if sentence_reps.dim() == 2:
            return self._forward_single_doc(sentence_reps, segment_labels, mask)

        # Batched [B, N, D]
        B = sentence_reps.size(0)
        total_loss = sentence_reps.new_zeros(())
        valid_docs = 0

        for b in range(B):
            doc_reps = sentence_reps[b]
            if isinstance(segment_labels, torch.Tensor) and segment_labels.dim() == 2:
                doc_labels = segment_labels[b]
            else:
                doc_labels = segment_labels
            doc_mask = mask[b] if mask is not None else None
            doc_loss = self._forward_single_doc(doc_reps, doc_labels, doc_mask)
            if doc_loss > 0:
                total_loss = total_loss + doc_loss
                valid_docs += 1

        if valid_docs == 0:
            return sentence_reps.new_zeros(())
        return total_loss / valid_docs

    def _forward_single_doc(
        self,
        reps: torch.Tensor,
        labels: Union[List[int], torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = reps.size(0)
        if N <= 1:
            return reps.new_zeros(())

        if mask is not None:
            valid_indices = torch.nonzero(mask.bool()).squeeze(-1)
            if valid_indices.numel() <= 1:
                return reps.new_zeros(())
            reps = reps[valid_indices]
            if isinstance(labels, torch.Tensor):
                labels = labels[valid_indices]
            else:
                labels = [labels[i] for i in valid_indices.tolist()]
            N = reps.size(0)

        device = reps.device
        dtype = reps.dtype
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=device, dtype=torch.long)
        else:
            labels = labels.to(device=device, dtype=torch.long)

        # 1. Hypersphere L2 normalization
        norm_reps = F.normalize(reps, p=2, dim=-1)

        # 2. Pairwise cosine similarity: [N, N]
        sim = torch.mm(norm_reps, norm_reps.t())

        # 3. Distance matrix & temporal decay weights: [N, N]
        idx_grid = torch.arange(N, device=device)
        dist_matrix = torch.abs(idx_grid.unsqueeze(1) - idx_grid.unsqueeze(0)).to(dtype=dtype)
        temporal_weights = torch.exp(-dist_matrix / self.tau_temp)

        total_doc_loss = reps.new_zeros(())
        valid_anchors = 0

        for i in range(N):
            curr_label = labels[i]
            pos_mask = (labels == curr_label) & (idx_grid != i)
            neg_mask = (labels != curr_label)

            pos_indices = torch.nonzero(pos_mask).squeeze(-1)
            if pos_indices.numel() == 0:
                # SINGLE-SENTENCE SEGMENT GUARD: skip singleton segment anchors
                continue

            valid_anchors += 1

            # Positives: w_{ij} * exp(sim_{ij} / tau)
            pos_exp = temporal_weights[i, pos_indices] * torch.exp(sim[i, pos_indices] / self.tau)

            # Negatives: sum exp(sim_{ik} / tau)
            neg_indices = torch.nonzero(neg_mask).squeeze(-1)
            if neg_indices.numel() > 0:
                neg_exp_sum = torch.sum(torch.exp(sim[i, neg_indices] / self.tau))
            else:
                neg_exp_sum = 0.0

            denom = pos_exp + neg_exp_sum + self.eps
            log_prob = torch.log(pos_exp / denom)
            anchor_loss = -torch.mean(log_prob)
            total_doc_loss = total_doc_loss + anchor_loss

        if valid_anchors == 0:
            return reps.new_zeros(())

        return total_doc_loss / valid_anchors


# Alias for backward compatibility
ContrastiveInfoNCELoss = TemporallyDecayedSupConLoss
SupConInfoNCELoss = TemporallyDecayedSupConLoss


# ==============================================================================
# 3. Multi-Task Joint Loss Balancing
# ==============================================================================

class MultiTaskSegmentationLoss(nn.Module):
    """Unified multi-task loss module combining focal, contrastive, and generative objectives:

    L_total = 10.0 * L_SM-FL + 1.0 * L_cont + 0.5 * L_gen
    """

    def __init__(
        self,
        lambda_focal: float = 10.0,
        lambda_contrastive: float = 1.0,
        lambda_generative: float = 0.5,
        sigma: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.25,
        tau: float = 0.07,
        tau_temp: float = 10.0,
    ) -> None:
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_contrastive = lambda_contrastive
        self.lambda_generative = lambda_generative

        self.focal_loss = SoftMarginFocalLoss(sigma=sigma, gamma=gamma, alpha=alpha)
        self.contrastive_loss = TemporallyDecayedSupConLoss(tau=tau, tau_temp=tau_temp)

    def forward(
        self,
        boundary_logits: torch.Tensor,
        gold_boundaries: Union[Set[int], List[int], torch.Tensor],
        sentence_reps: torch.Tensor,
        segment_labels: Union[List[int], torch.Tensor],
        generative_loss: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> SegmentationLossOutput:
        """Compute the joint calibrated multi-task loss."""
        # 1. Soft-Margin Focal Loss
        l_focal = self.focal_loss(boundary_logits, gold_boundaries, mask=mask)

        # 2. Supervised Contrastive Loss
        l_cont = self.contrastive_loss(sentence_reps, segment_labels, mask=mask)

        # 3. Generative Title Loss
        if generative_loss is None:
            l_gen = boundary_logits.new_zeros(())
        else:
            l_gen = generative_loss

        # Calibrated total loss: 10.0 * L_SM-FL + 1.0 * L_cont + 0.5 * L_gen
        total = (
            self.lambda_focal * l_focal
            + self.lambda_contrastive * l_cont
            + self.lambda_generative * l_gen
        )

        return SegmentationLossOutput(
            total=total,
            focal=l_focal.detach(),
            contrastive=l_cont.detach(),
            generative=l_gen.detach(),
            weights={
                "lambda_focal": self.lambda_focal,
                "lambda_contrastive": self.lambda_contrastive,
                "lambda_generative": self.lambda_generative,
            },
        )


# Alias for backward compatibility
MultiTaskLossManager = MultiTaskSegmentationLoss
