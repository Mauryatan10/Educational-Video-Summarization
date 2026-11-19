"""src/segmentation/modernbert_extractor.py - ModernBERT Sentence Representation Extractor.

Extracts dense sentence representations from educational transcripts using answerdotai/ModernBERT-base.
Features:
- Native bidirectional ModernBERT-base backbone (149M params, D=768, 8,192 token context window, FP16).
- Fast Rust-based Hugging Face tokenizer (AutoTokenizer).
- Chunk-level activation checkpointing via torch.utils.checkpoint.checkpoint(..., use_reentrant=False).
- 100% full-document autograd across ALL document sentences without gradient detachment or chunk truncation.
- Masked mean pooling with attention mask, fully compatible with optional LearnableQueryAttentionPooling.
- PEFT LoRA adapter support targeting ModernBERT fused attention and MLP projections ('Wqkv', 'Wo', 'Wi').
- Strongly-typed, production-tested forward and encode_document APIs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    PeftModel = Any

logger = logging.getLogger(__name__)

DEFAULT_MODERNBERT_MODEL: str = "answerdotai/ModernBERT-base"
DEFAULT_CHUNK_SIZE: int = 16
DEFAULT_HIDDEN_SIZE: int = 768
DEFAULT_MAX_CONTEXT: int = 8192
DEFAULT_LORA_TARGET_MODULES: List[str] = ["Wqkv", "Wo", "Wi"]


class ModernBERTSentenceRepresentationExtractor(nn.Module):
    """Dense sentence representation extractor utilizing answerdotai/ModernBERT-base.

    Solves the prior gradient detachment defect by maintaining unbroken autograd history
    across all sentences in a document using chunk-level activation checkpointing.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODERNBERT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        torch_dtype: torch.dtype = torch.float16,
        device: Optional[Union[str, torch.device]] = None,
        pooling: Optional[nn.Module] = None,
        checkpointing: bool = True,
        output_dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        """Initialize ModernBERTSentenceRepresentationExtractor.

        Args:
            model_name_or_path: Hugging Face model identifier or local path.
            chunk_size: Number of sentences processed per activation checkpointing chunk.
            torch_dtype: Storage and execution dtype for ModernBERT backbone weights (default torch.float16).
            device: Optional torch.device or device string to place modules on.
            pooling: Optional token-level pooling module (e.g. LearnableQueryAttentionPooling).
                     If None, defaults to masked mean pooling over non-padding tokens.
            checkpointing: Whether to enable chunk-level activation checkpointing during training.
            output_dtype: Output dtype for sentence representations (default torch.float32 for
                          seamless downstream DBCA compatibility).
        """
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.chunk_size = chunk_size
        self.torch_dtype = torch_dtype
        self.checkpointing = checkpointing
        self.output_dtype = output_dtype
        self.is_lora = False

        # 1. Fast Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        # 2. ModernBERT-base Backbone (FP16 weights)
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
        )

        self.hidden_size: int = getattr(self.backbone.config, "hidden_size", DEFAULT_HIDDEN_SIZE)
        self.max_position_embeddings: int = getattr(
            self.backbone.config, "max_position_embeddings", DEFAULT_MAX_CONTEXT
        )

        # 3. Optional Pooling Module
        self.pooling = pooling

        if device is not None:
            self.to(device)

    @property
    def device(self) -> torch.device:
        """Return the current device of the backbone parameters."""
        return next(self.backbone.parameters()).device

    def set_pooling(self, pooling: Optional[nn.Module]) -> None:
        """Set or update the token-level pooling module."""
        self.pooling = pooling

    def apply_lora(
        self,
        r: int = 16,
        alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.05,
        init_lora_weights: Union[bool, str] = True,
    ) -> nn.Module:
        """Attach PEFT LoRA adapters to ModernBERT backbone.

        ModernBERT uses fused projections:
        - Attention: 'Wqkv' (query/key/value fused), 'Wo' (output projection)
        - FFN: 'Wi' (intermediate projection), 'Wo' (output projection)

        Args:
            r: LoRA decomposition rank.
            alpha: LoRA scaling factor.
            target_modules: Module names to target with LoRA adapters (default: ['Wqkv', 'Wo', 'Wi']).
            dropout: LoRA dropout probability.
            init_lora_weights: Whether or how to initialize LoRA weights (True/False/'gaussian').

        Returns:
            LoRA-wrapped backbone module.
        """
        if not HAS_PEFT:
            raise ImportError(
                "The 'peft' library is required to apply LoRA. Install with: pip install peft"
            )

        if target_modules is None:
            target_modules = list(DEFAULT_LORA_TARGET_MODULES)

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            task_type=TaskType.FEATURE_EXTRACTION,
            init_lora_weights=init_lora_weights,
        )

        self.backbone = get_peft_model(self.backbone, peft_config)
        self.is_lora = True
        logger.info(
            f"Successfully applied LoRA to ModernBERT (rank={r}, alpha={alpha}, targets={target_modules})"
        )
        return self.backbone

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the ModernBERT backbone (e.g. for Stage 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the ModernBERT backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_lora_only(self) -> None:
        """Unfreeze only LoRA adapter parameters while keeping base weights frozen (for Stage 2 & 3)."""
        if not self.is_lora:
            raise RuntimeError(
                "Cannot unfreeze LoRA parameters: LoRA has not been applied. Call apply_lora() first."
            )
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        """Return counts and ratios of total, trainable, and frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        pct = (trainable / total * 100.0) if total > 0 else 0.0
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_pct": pct,
        }

    def _forward_chunk(
        self,
        chunk_ids: torch.Tensor,
        chunk_mask: torch.Tensor,
        output_token_level: bool = False,
    ) -> torch.Tensor:
        """Execute forward pass for a single chunk of sentences.

        Args:
            chunk_ids: [chunk_sentences, T] token IDs on module device.
            chunk_mask: [chunk_sentences, T] attention mask on module device.
            output_token_level: If True, return [chunk_sentences, T, D]; else return [chunk_sentences, D].

        Returns:
            Chunk representations tensor.
        """
        outputs = self.backbone(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
        )
        hidden_states = outputs.last_hidden_state  # [chunk_sentences, T, D]

        if output_token_level:
            if self.output_dtype is not None:
                return hidden_states.to(self.output_dtype)
            return hidden_states

        if self.pooling is not None:
            # Ensure hidden states match pooling module parameter dtype (e.g. Float vs Half)
            pool_params = list(self.pooling.parameters())
            in_states = hidden_states.to(pool_params[0].dtype) if pool_params else hidden_states
            pooled, _ = self.pooling(in_states, token_mask=chunk_mask)
        else:
            # Masked mean pooling over non-padding tokens (FP16 underflow-safe)
            mask_exp = chunk_mask.unsqueeze(-1).to(hidden_states.dtype)  # [chunk_sentences, T, 1]
            sum_hidden = torch.sum(hidden_states * mask_exp, dim=1)      # [chunk_sentences, D]
            token_counts = torch.clamp(chunk_mask.sum(dim=1, keepdim=True), min=1).to(hidden_states.dtype)
            pooled = sum_hidden / token_counts

        if self.output_dtype is not None:
            pooled = pooled.to(self.output_dtype)
        return pooled

    def forward(
        self,
        sentence_token_ids: Union[torch.Tensor, List[str], List[List[str]]],
        attention_mask: Optional[torch.Tensor] = None,
        output_token_level: bool = False,
    ) -> torch.Tensor:
        """Extract sentence representations across all document sentences.

        Guarantees 100% full-document autograd: backpropagation flows through ALL sentences
        in the document without artificial gradient detachment (.detach()) or chunk truncation.
        Activation checkpointing at the chunk boundary bounds memory consumption.

        Args:
            sentence_token_ids: Tensor[B, S, T] or Tensor[S, T] token IDs,
                                or List[str] / List[List[str]] raw sentences.
            attention_mask: Optional Tensor matching sentence_token_ids shape.
            output_token_level: If True, returns token hidden states [B, S, T, D];
                                else returns pooled sentence representations [B, S, D].

        Returns:
            Tensor of shape [B, S, 768] (or [B, S, T, 768] if output_token_level=True).
        """
        # Convenience handler for raw string inputs
        if isinstance(sentence_token_ids, list):
            if not sentence_token_ids:
                d = self.hidden_size
                out_shape = (1, 0, 0, d) if output_token_level else (1, 0, d)
                return torch.zeros(
                    out_shape,
                    device=self.device,
                    dtype=self.output_dtype or self.torch_dtype,
                )

            if isinstance(sentence_token_ids[0], str):
                encoded = self.tokenizer(
                    sentence_token_ids,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt",
                )
                sentence_token_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
            elif isinstance(sentence_token_ids[0], list):
                doc_tensors: List[torch.Tensor] = []
                for doc in sentence_token_ids:
                    doc_rep = self.encode_document(doc, output_token_level=output_token_level)
                    doc_tensors.append(doc_rep)
                return torch.cat(doc_tensors, dim=0)

        # Standardize to 3D tensor: [B, S, T]
        is_2d = sentence_token_ids.dim() == 2
        if is_2d:
            sentence_token_ids = sentence_token_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0)

        B, S, T = sentence_token_ids.shape

        if S == 0:
            d = self.hidden_size
            out_shape = (B, 0, T, d) if output_token_level else (B, 0, d)
            return torch.zeros(
                out_shape,
                device=sentence_token_ids.device,
                dtype=self.output_dtype or self.torch_dtype,
            )

        if attention_mask is None:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                attention_mask = (sentence_token_ids != pad_id).long()
            else:
                attention_mask = torch.ones_like(sentence_token_ids)

        # Ensure tensors reside on the backbone device
        dev = self.device
        if sentence_token_ids.device != dev:
            sentence_token_ids = sentence_token_ids.to(dev)
        if attention_mask.device != dev:
            attention_mask = attention_mask.to(dev)

        # Flatten batch and sentence dimensions for chunked processing: [N, T]
        N = B * S
        flat_token_ids = sentence_token_ids.reshape(N, T)
        flat_attention_mask = attention_mask.reshape(N, T)

        # Check if activation checkpointing should be activated
        has_trainable = any(p.requires_grad for p in self.parameters())
        can_checkpoint = self.checkpointing and torch.is_grad_enabled() and has_trainable

        chunks: List[torch.Tensor] = []
        for i in range(0, N, self.chunk_size):
            chunk_ids = flat_token_ids[i : i + self.chunk_size]
            chunk_mask = flat_attention_mask[i : i + self.chunk_size]

            if output_token_level:
                def forward_chunk(c_ids: torch.Tensor, c_mask: torch.Tensor) -> torch.Tensor:
                    return self._forward_chunk(c_ids, c_mask, output_token_level=True)
            else:
                def forward_chunk(c_ids: torch.Tensor, c_mask: torch.Tensor) -> torch.Tensor:
                    return self._forward_chunk(c_ids, c_mask, output_token_level=False)

            if can_checkpoint:
                chunk_out = checkpoint(forward_chunk, chunk_ids, chunk_mask, use_reentrant=False)
            else:
                chunk_out = forward_chunk(chunk_ids, chunk_mask)

            chunks.append(chunk_out)

        all_reps = torch.cat(chunks, dim=0)

        # Reshape to original batch and sentence dimensions
        D = all_reps.size(-1)
        if output_token_level:
            all_reps = all_reps.view(B, S, T, D)
        else:
            all_reps = all_reps.view(B, S, D)

        return all_reps

    def encode_document(
        self,
        sentences: List[str],
        max_tokens_per_sentence: int = 64,
        device: Optional[torch.device] = None,
        output_token_level: bool = False,
    ) -> torch.Tensor:
        """Encode raw sentence string lists into sentence representations.

        Args:
            sentences: List of raw sentence text strings for one document.
            max_tokens_per_sentence: Max tokens to pad/truncate each sentence (default 64).
            device: Optional torch.device to override target device.
            output_token_level: If True, returns [1, S, T, 768]; else [1, S, 768].

        Returns:
            Tensor[1, S, 768] (or [1, S, T, 768]) with computation graph intact.
        """
        target_device = device or self.device
        if not sentences:
            out_shape = (1, 0, 0, self.hidden_size) if output_token_level else (1, 0, self.hidden_size)
            return torch.zeros(
                out_shape,
                device=target_device,
                dtype=self.output_dtype or self.torch_dtype,
            )

        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_tokens_per_sentence,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(target_device)
        attention_mask = encoded["attention_mask"].to(target_device)

        reps = self.forward(
            sentence_token_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            output_token_level=output_token_level,
        )
        return reps
