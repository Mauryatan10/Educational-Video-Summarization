"""Slide Representation Encoder utilizing Qwen 3.5 0.8B Vision Transformer.

Maps canonical slide keyframes to 1024-dimensional semantic representations and aligns
them temporally to spoken transcript sentence intervals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.vision.keyframe_extractor import KeyframeInfo

logger = logging.getLogger(__name__)


class QwenSlideRepresentationEncoder:
    """Encodes canonical lecture keyframes into 1024-dim visual vectors using Qwen 3.5 Vision Tower."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        visual_model: Optional[torch.nn.Module] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.dtype = dtype

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if visual_model is not None:
            self.visual = visual_model
        else:
            base_model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
            self.visual = base_model.visual.to(self.device).eval()

    @torch.no_grad()
    def encode_keyframes(self, keyframes: List[KeyframeInfo], batch_size: int = 4) -> torch.Tensor:
        """Encode list of canonical keyframes into a tensor of shape [K, 1024]."""
        if not keyframes:
            return torch.zeros((0, 1024), dtype=torch.float32)

        embeddings: List[torch.Tensor] = []

        for i in range(0, len(keyframes), batch_size):
            batch_kfs = keyframes[i : i + batch_size]
            images: List[Image.Image] = []
            for kf in batch_kfs:
                try:
                    img = Image.open(kf.image_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to open keyframe image {kf.image_path}: {e}")
                    images.append(Image.new("RGB", (320, 240), color=(0, 0, 0)))

            inputs = self.processor(images, return_tensors="pt")
            device = next(self.visual.parameters()).device
            pixel_dtype = next(self.visual.parameters()).dtype

            pixel_values = inputs["pixel_values"].to(device, dtype=pixel_dtype)
            image_grid_thw = inputs["image_grid_thw"].to(device)

            out = self.visual(pixel_values, grid_thw=image_grid_thw)
            pooler_output = out.pooler_output
            grid_thws = inputs["image_grid_thw"].cpu().numpy()

            curr_offset = 0
            for thw in grid_thws:
                t, h, w = thw[0], thw[1], thw[2]
                n_patches = t * (h // 2) * (w // 2)
                img_patches = pooler_output[curr_offset : curr_offset + n_patches]
                curr_offset += n_patches
                img_rep = img_patches.mean(dim=0).cpu().to(torch.float32)
                embeddings.append(img_rep)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stacked = torch.stack(embeddings, dim=0)  # [K, 1024]
        return stacked

    @staticmethod
    def align_slides_to_sentences(
        sentence_timestamps: List[Tuple[float, float]],
        keyframes: List[KeyframeInfo],
        slide_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Map canonical slide embeddings to sentence timeline [N, 1024]."""
        n_sentences = len(sentence_timestamps)
        if not keyframes or slide_embeddings.shape[0] == 0:
            return torch.zeros((n_sentences, 1024), dtype=torch.float32)

        aligned = torch.zeros((n_sentences, 1024), dtype=torch.float32)
        kf_times = [kf.timestamp_sec for kf in keyframes]

        for s_idx, (t_start, t_end) in enumerate(sentence_timestamps):
            t_mid = (t_start + t_end) / 2.0
            active_k = 0
            for k_idx, kt in enumerate(kf_times):
                if kt <= t_mid:
                    active_k = k_idx
                else:
                    break
            aligned[s_idx] = slide_embeddings[active_k]

        return aligned
