"""tests/test_qwen_extractor.py - Unit and Integration Tests for Qwen 3.5 0.8B Extractor."""

import pytest
import torch

from src.segmentation.qwen_extractor import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_HIDDEN_SIZE,
    QwenSentenceRepresentationExtractor,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 4-bit Qwen test")
def test_qwen_extractor_init_and_lora():
    """Verify Qwen 3.5 0.8B loads in 4-bit and attaches LoRA adapters."""
    extractor = QwenSentenceRepresentationExtractor(
        model_name_or_path=DEFAULT_QWEN_MODEL,
        load_in_4bit=True,
        chunk_size=8,
    )
    assert extractor.hidden_size == DEFAULT_QWEN_HIDDEN_SIZE
    assert extractor.tokenizer.pad_token is not None

    # Apply LoRA
    extractor.apply_lora(r=16, alpha=32)
    assert extractor.is_lora

    param_stats = extractor.count_parameters()
    assert param_stats["trainable_parameters"] > 0
    assert param_stats["trainable_pct"] < 2.0  # LoRA is ~0.74% of total params

    # Test freeze/unfreeze
    extractor.freeze_backbone()
    assert extractor.count_parameters()["trainable_parameters"] == 0

    extractor.unfreeze_lora_only()
    assert extractor.count_parameters()["trainable_parameters"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 4-bit Qwen test")
def test_qwen_extractor_full_document_autograd():
    """Verify 100% full-document autograd with chunked activation checkpointing across all sentences."""
    extractor = QwenSentenceRepresentationExtractor(
        model_name_or_path=DEFAULT_QWEN_MODEL,
        load_in_4bit=True,
        chunk_size=8,
    )
    extractor.apply_lora(r=16, alpha=32)
    extractor.unfreeze_lora_only()
    extractor.train()

    test_sentences = [
        f"Sentence {i}: Understanding asymptotic complexity and dynamic programming algorithms."
        for i in range(24)
    ]

    reps = extractor(test_sentences)
    assert reps.shape == (1, 24, DEFAULT_QWEN_HIDDEN_SIZE)
    assert reps.requires_grad

    loss = reps.sum()
    loss.backward()

    # Verify gradients flowed into LoRA adapter weights
    has_grad = False
    for name, param in extractor.backbone.named_parameters():
        if "lora_" in name and param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradients!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
            has_grad = True
    assert has_grad, "No LoRA parameters received gradients!"

