"""src/segmentation/nms.py - 1D Non-Maximum Suppression for boundary clustering elimination."""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def non_maximum_suppression_1d(
    probabilities: Union[Sequence[float], np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    radius: int = 2,
) -> List[int]:
    """1D Non-Maximum Suppression with radius r sentences.

    Algorithm:
    1. Filter candidate indices where p_i >= threshold.
    2. Sort candidates descending by probability; break ties by earliest index.
    3. Greedily select the highest peak and suppress all neighbors within |i - j| <= radius.
    4. Return sorted list of selected boundary indices.

    Args:
        probabilities: Array or tensor of boundary transition probabilities p_i in [0, 1].
        threshold: Minimum probability threshold for candidate boundaries (default 0.5).
        radius: Elimination radius in sentences (default 2).

    Returns:
        Sorted list of selected boundary indices satisfying the separation constraint.
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().float().numpy().flatten()
    elif isinstance(probabilities, np.ndarray):
        probs = probabilities.flatten().astype(np.float64)
    else:
        probs = np.array(probabilities, dtype=np.float64)

    N = len(probs)
    if N == 0:
        return []

    # Candidates exceeding decision threshold
    candidates = [i for i in range(N) if probs[i] >= threshold]
    if not candidates:
        return []

    # Sort descending by probability, tie-break by earliest index
    sorted_candidates = sorted(candidates, key=lambda idx: (-probs[idx], idx))

    suppressed = set()
    selected = []

    for idx in sorted_candidates:
        if idx in suppressed:
            continue
        selected.append(idx)
        # Suppress all neighboring candidates within distance <= radius
        for neighbor in range(max(0, idx - radius), min(N, idx + radius + 1)):
            if neighbor != idx:
                suppressed.add(neighbor)

    return sorted(selected)


# Alias for backward compatibility
apply_1d_nms = non_maximum_suppression_1d


class Boundary1DNMS(nn.Module):
    """PyTorch module wrapper for 1D Non-Maximum Suppression."""

    def __init__(self, threshold: float = 0.5, radius: int = 2) -> None:
        super().__init__()
        self.threshold = threshold
        self.radius = radius

    def forward(
        self,
        probabilities_or_logits: torch.Tensor,
        is_logits: bool = True,
    ) -> List[List[int]]:
        """Apply 1D-NMS across a batch of sequences.

        Args:
            probabilities_or_logits: Tensor of shape [B, N] or [N].
            is_logits: If True, applies sigmoid to convert logits to probabilities.

        Returns:
            List of selected boundary index lists for each sequence in the batch.
        """
        tensor = probabilities_or_logits
        if is_logits:
            tensor = torch.sigmoid(tensor)

        if tensor.dim() == 1:
            return [non_maximum_suppression_1d(tensor, self.threshold, self.radius)]

        B = tensor.size(0)
        results = []
        for b in range(B):
            selected = non_maximum_suppression_1d(tensor[b], self.threshold, self.radius)
            results.append(selected)
        return results

    def to_binary_mask(
        self,
        probabilities_or_logits: torch.Tensor,
        is_logits: bool = True,
    ) -> torch.Tensor:
        """Convert probabilities or logits into a binary boundary indicator tensor."""
        is_1d = probabilities_or_logits.dim() == 1
        batched = probabilities_or_logits.unsqueeze(0) if is_1d else probabilities_or_logits
        B, N = batched.shape
        device = probabilities_or_logits.device

        selected_lists = self.forward(batched, is_logits=is_logits)
        mask = torch.zeros((B, N), device=device, dtype=torch.bool)
        for b, indices in enumerate(selected_lists):
            if indices:
                mask[b, indices] = True

        return mask.squeeze(0) if is_1d else mask


# Alias for backward compatibility
OneDimensionalNMS = Boundary1DNMS
