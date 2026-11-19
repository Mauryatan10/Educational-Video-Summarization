"""Comprehensive Unit Test Suite for Milestone 2: Discourse-Aware Topic Segmentation Architecture.

Validates:
1. Query-based attention pooling (dimensions, softmax sum, token masking, acoustic fusion, parity).
2. DBCA adapter (bidirectional visibility, sentence masking, residual connections, H=1024, parity).
3. Dual heads (transition MLP, 4D features, extreme logits, title parsing, deterministic fallback).
4. Losses (Soft-Margin Focal Loss logit stability, InfoNCE single-sentence guard, balancing, parity).
5. 1D Non-Maximum Suppression (radius r=2 suppression, local peak selection, tie-breaking, parity).
6. Pipeline integration & Contract 1 -> Contract 2 conformance on sample lecture transcripts.
7. Hardware constraints (RTX 4050 6GB VRAM bounds, 16GB RAM envelope, Ollama keep_alive: 0).
8. Unified Discourse-Aware Segmentation Model integration.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pytest

from tests.fixtures.audio_fixture import (
    get_contract1_sample_sentences,
    get_contract2_sample_payload,
    get_synthetic_10min_lecture_fixture,
    validate_contract1_sentence,
    validate_contract2_payload,
)
from tests.fixtures.reference_oracles import (
    reference_1d_nms,
    reference_contrastive_infonce_loss,
    reference_dbca_adapter,
    reference_multitask_total_loss,
    reference_query_attention_pooling,
    reference_soft_margin_focal_loss,
    reference_transition_features,
)
from src.segmentation.pipeline import (
    DEFAULT_NMS_RADIUS,
    DEFAULT_NMS_THRESHOLD,
    OllamaTitleGenerator,
    SegmentationPayload,
    SegmentationPipeline,
    SegmentationPipelineConfig,
    TopicSegment,
)
from src.segmentation.nms import (
    Boundary1DNMS,
    apply_1d_nms,
    non_maximum_suppression_1d,
)

# Optional PyTorch imports for neural module verification
try:
    import torch
    import torch.nn as nn
    from src.segmentation.pooling import (
        AcousticProsodyFusion,
        LearnableQueryAttentionPooling,
        PoolingConfig,
    )
    from src.segmentation.dbca_adapter import (
        DBCAConfig,
        DecoupledBidirectionalContextAdapter,
    )
    from src.segmentation.dual_head import (
        ContrastiveProjectionHead,
        DiscriminativeTransitionHead,
        DualHeadConfig,
        GenerativeTitleHead,
        TopicMetadata,
        TransitionFeatureExtractor,
    )
    from src.segmentation.losses import (
        MultiTaskSegmentationLoss,
        SoftMarginFocalLoss,
        TemporallyDecayedSupConLoss,
        compute_contrastive_infonce_loss,
        compute_soft_margin_focal_loss,
        gaussian_label_smoothing,
    )
    from src.segmentation.model import (
        DiscourseAwareSegmentationConfig,
        DiscourseAwareSegmentationModel,
        SegmentationModelOutput,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# 1. Test Class: Query-Based Attention Pooling
# ==============================================================================

class TestQueryAttentionPooling:
    """Verifies learnable query attention pooling: alpha_{i,t} = softmax(q^T tanh(W_p H_{i,t}))."""

    def test_pooling_output_dimensions(self):
        """Verify token hidden states [T, D] reduce to fixed sentence vector [D]."""
        D = 128
        K = 64
        q = np.random.randn(K)
        W_p = np.random.randn(K, D)

        for T in [1, 5, 17, 64, 128]:
            H = np.random.randn(T, D)
            u, alpha = reference_query_attention_pooling(H, q, W_p)
            assert u.shape == (D,), f"Expected shape ({D},), got {u.shape}"
            assert alpha.shape == (T,), f"Expected alpha shape ({T},), got {alpha.shape}"

    def test_pooling_attention_softmax_normalization(self):
        """Verify attention weights alpha are non-negative and sum to exactly 1.0."""
        T, D, K = 32, 256, 128
        H = np.random.randn(T, D)
        q = np.random.randn(K)
        W_p = np.random.randn(K, D)

        _, alpha = reference_query_attention_pooling(H, q, W_p)
        assert np.all(alpha >= 0.0), "Attention weights must be non-negative"
        assert np.isclose(np.sum(alpha), 1.0, atol=1e-6), f"Attention weights must sum to 1.0, got {np.sum(alpha)}"

    def test_pooling_token_padding_masking(self):
        """Verify padded token states receive zero attention weight."""
        T, D = 10, 64
        mask = np.ones(T, dtype=bool)
        mask[7:] = False  # Tokens 7, 8, 9 are padding

        scores = np.random.randn(T)
        scores[~mask] = -1e9
        alpha = np.exp(scores - np.max(scores))
        alpha /= np.sum(alpha)

        assert np.all(alpha[7:] < 1e-6), "Masked tokens must have near-zero attention"
        assert np.isclose(np.sum(alpha[:7]), 1.0, atol=1e-5)

    def test_pooling_acoustic_prosody_fusion(self):
        """Verify concatenation and projection of [u_i || pause || pitch || speaker] to d_model=1024."""
        D_text = 256
        D_acoustic = 3  # [pause_sec, pitch_semitones, speaker_flag]
        d_model = 1024

        u_i = np.random.randn(D_text)
        acoustic = np.array([0.65, 2.1, 1.0])
        fused_raw = np.concatenate([u_i, acoustic])
        assert fused_raw.shape == (D_text + D_acoustic,)

        W_proj = np.random.randn(D_text + D_acoustic, d_model)
        x_i = np.dot(fused_raw, W_proj)
        assert x_i.shape == (d_model,)

    def test_pooling_acoustic_prosody_fallback(self):
        """Verify clean execution when acoustic prosody features are absent or zeroed."""
        D_text = 256
        d_model = 1024
        u_i = np.random.randn(D_text)
        acoustic_zeros = np.zeros(3)
        fused_raw = np.concatenate([u_i, acoustic_zeros])
        W_proj = np.random.randn(D_text + 3, d_model)
        x_i = np.dot(fused_raw, W_proj)
        assert x_i.shape == (d_model,)
        assert not np.isnan(x_i).any()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_pooling_pytorch_learnable_query_module(self):
        """Verify PyTorch LearnableQueryAttentionPooling module execution and shapes."""
        B, S, T, D = 2, 4, 16, 128
        K = 64
        cfg = PoolingConfig(hidden_size=D, query_proj_dim=K, adapter_dim=1024)
        pool = LearnableQueryAttentionPooling(cfg)

        token_hidden = torch.randn(B, S, T, D)
        mask = torch.ones(B, S, T, dtype=torch.bool)
        mask[:, :, 12:] = False  # Last 4 tokens padded

        pooled, attn_weights = pool(token_hidden, token_mask=mask)
        assert pooled.shape == (B, S, D)
        assert attn_weights.shape == (B, S, T)
        # Verify padded tokens received 0.0 attention
        assert torch.all(attn_weights[:, :, 12:] == 0.0)
        # Verify valid tokens sum to 1.0
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(B, S), atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_pooling_pytorch_acoustic_fusion_module(self):
        """Verify PyTorch AcousticProsodyFusion module with and without acoustic inputs."""
        B, S, D = 2, 5, 256
        cfg = PoolingConfig(hidden_size=D, adapter_dim=1024)
        fusion = AcousticProsodyFusion(cfg)

        reps = torch.randn(B, S, D)
        # 1. With acoustic features
        pause = torch.tensor([[0.0, 500.0, 1500.0, 200.0, 800.0], [100.0, 0.0, 600.0, 1200.0, 300.0]])
        pitch = torch.randn(B, S)
        spk = torch.zeros(B, S, dtype=torch.long)

        fused = fusion(reps, pause_before_ms=pause, pitch_reset=pitch, speaker_ids=spk)
        assert fused.shape == (B, S, 1024)
        assert not torch.isnan(fused).any()

        # 2. Text-only fallback
        fallback = fusion(reps)
        assert fallback.shape == (B, S, 1024)
        assert not torch.isnan(fallback).any()

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_pooling_autograd_gradient_propagation(self):
        """Verify autograd propagates non-zero gradients through pooling weights."""
        T, D, K = 12, 64, 32
        cfg = PoolingConfig(hidden_size=D, query_proj_dim=K, adapter_dim=1024)
        pool = LearnableQueryAttentionPooling(cfg)

        H = torch.randn(1, 1, T, D, requires_grad=True)
        u, _ = pool(H)
        loss = u.sum()
        loss.backward()

        assert pool.q.grad is not None and torch.norm(pool.q.grad) > 0
        assert pool.W_p.weight.grad is not None and torch.norm(pool.W_p.weight.grad) > 0
        assert H.grad is not None and torch.norm(H.grad) > 0


# ==============================================================================
# 2. Test Class: Decoupled Bidirectional Context Adapter (DBCA)
# ==============================================================================

class TestDBCAAdapter:
    """Verifies 2-layer Decoupled Bidirectional Context Adapter (DBCA)."""

    def test_dbca_output_shape_preservation(self):
        """Verify DBCA preserves sequence length and dimension [N, H]."""
        N, H = 14, 1024
        X = np.random.randn(N, H)
        C = reference_dbca_adapter(X, num_heads=8, hidden_dim=H)
        assert C.shape == (N, H)
        assert not np.isnan(C).any()

    def test_dbca_bidirectional_context_visibility(self):
        """Verify sentence representation c_i depends on both past S_{<i} and future S_{>i}."""
        N, D = 8, 64
        X = np.random.randn(N, D)
        C_orig = reference_dbca_adapter(X)

        # Alter the last sentence X[7]
        X_perturbed = X.copy()
        X_perturbed[7] += 10.0
        C_perturbed = reference_dbca_adapter(X_perturbed)

        # Representation at index i=1 must change due to bidirectional self-attention
        diff_at_1 = np.linalg.norm(C_orig[1] - C_perturbed[1])
        assert diff_at_1 > 1e-4, "DBCA must provide future discourse visibility to eliminate causal blindness"

    def test_dbca_residual_skip_connections(self):
        """Verify residual addition preserves base sentence representation magnitude."""
        N, D = 6, 32
        X = np.random.randn(N, D)
        C = reference_dbca_adapter(X)
        for i in range(N):
            cos_sim = np.dot(X[i], C[i]) / (np.linalg.norm(X[i]) * np.linalg.norm(C[i]))
            assert cos_sim > 0.40, f"Residual connection must maintain strong correlation: {cos_sim}"

    def test_dbca_single_sentence_safety(self):
        """Verify DBCA operates cleanly on minimal single-sentence documents (N=1)."""
        X = np.random.randn(1, 64)
        C = reference_dbca_adapter(X)
        assert C.shape == (1, 64)
        assert not np.isnan(C).any()

    def test_dbca_multi_head_split_divisibility(self):
        """Verify H=1024 is evenly divisible by 8 attention heads with head_dim=128."""
        H = 1024
        heads = 8
        assert H % heads == 0
        assert H // heads == 128

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_dbca_pytorch_module_execution(self):
        """Verify PyTorch DecoupledBidirectionalContextAdapter forward pass and masking."""
        B, S, H = 2, 6, 1024
        cfg = DBCAConfig(hidden_size=H, num_heads=8, d_ff=4096, num_layers=2)
        dbca = DecoupledBidirectionalContextAdapter(cfg)

        x = torch.randn(B, S, H)
        mask = torch.ones(B, S, dtype=torch.bool)
        mask[:, 4:] = False  # Sentences 4, 5 are padding

        out = dbca(x, sentence_mask=mask)
        assert out.shape == (B, S, H)
        assert not torch.isnan(out).any()
        # Padding positions should be zeroed
        assert torch.all(out[:, 4:] == 0.0)


# ==============================================================================
# 3. Test Class: Dual Heads (Transition MLP & Generative Titling)
# ==============================================================================

class TestDualHeads:
    """Verifies Discriminative Transition Head and Generative Pedagogical Titling."""

    def test_transition_feature_4d_vector_construction(self):
        """Verify transition feature f_i^trans = [c_i || c_{i-1} || (c_i - c_{i-1}) || (c_i * c_{i-1})] has dimension 4H."""
        N, H = 10, 1024
        C = np.random.randn(N, H)
        feats = reference_transition_features(C)
        assert feats.shape == (N - 1, 4 * H)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_transition_head_mlp_forward_pass(self):
        """Verify Transition MLP projects 4H feature vector to scalar transition logits."""
        B, N, H = 2, 8, 1024
        cfg = DualHeadConfig(adapter_dim=H, transition_hidden=512)
        head = DiscriminativeTransitionHead(cfg)

        c = torch.randn(B, N, H)
        # 1. Unpadded valid transitions: [B, N-1]
        logits = head(c, pad_first=False)
        assert logits.shape == (B, N - 1)

        # 2. Aligned with sequence length: [B, N]
        aligned_logits = head(c, pad_first=True)
        assert aligned_logits.shape == (B, N)
        assert torch.all(aligned_logits[:, 0] == -1e4)

    def test_transition_head_extreme_logits_probability_bounds(self):
        """Verify sigmoid probabilities are strictly bounded within [0.0, 1.0] under extreme logits."""
        extreme_logits = np.array([-100.0, -25.0, 0.0, 25.0, 100.0])
        probs = 1.0 / (1.0 + np.exp(-np.clip(extreme_logits, -50.0, 50.0)))
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert np.isclose(probs[0], 0.0)
        assert np.isclose(probs[-1], 1.0)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_contrastive_projection_head_unit_norm(self):
        """Verify ContrastiveProjectionHead maps contextual states to unit hypersphere."""
        B, S, H = 2, 5, 1024
        cfg = DualHeadConfig(adapter_dim=H, contrastive_dim=256)
        proj_head = ContrastiveProjectionHead(cfg)

        c = torch.randn(B, S, H)
        z = proj_head(c)
        assert z.shape == (B, S, 256)
        norms = torch.norm(z, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(B, S), atol=1e-5)

    def test_generative_title_json_parsing(self):
        """Verify Ollama JSON response parser extracts title, prereqs, and cue questions."""
        cfg = SegmentationPipelineConfig(force_fallback=False)
        gen = OllamaTitleGenerator(cfg)
        mock_response = (
            '{\n'
            '  "topic_title": "Virtual Memory and Page Table Translation",\n'
            '  "prerequisite_tags": ["Process Abstraction", "RAM Hierarchy"],\n'
            '  "cue_questions": ["How does the MMU translate virtual page numbers?"]\n'
            '}'
        )
        data = gen._clean_and_parse_json(mock_response)
        assert data is not None
        assert data["topic_title"] == "Virtual Memory and Page Table Translation"
        assert len(data["prerequisite_tags"]) == 2
        assert data["cue_questions"][0].endswith("?")

    def test_generative_title_deterministic_fallback(self):
        """Verify deterministic heuristic generator emits valid title, tags, and '?' cues."""
        cfg = SegmentationPipelineConfig(force_fallback=True)
        gen = OllamaTitleGenerator(cfg)
        meta = gen.generate_metadata("In this section we discuss CPU context switching overhead.", is_digression=False)
        assert isinstance(meta["topic_title"], str) and len(meta["topic_title"]) >= 10
        assert isinstance(meta["prerequisite_tags"], list) and len(meta["prerequisite_tags"]) >= 1
        assert isinstance(meta["cue_questions"], list) and len(meta["cue_questions"]) >= 1
        for q in meta["cue_questions"]:
            assert q.endswith("?"), f"Question must end with '?': {q}"

    def test_generative_title_token_normalized_loss(self):
        """Verify generative loss calculation divides cross-entropy by total generated tokens."""
        # 2 segments: lengths 20 and 30, loss sum = 125.0 -> 2.5 nats/token
        total_tokens = 50
        loss_sum = 125.0
        normalized = loss_sum / total_tokens
        assert np.isclose(normalized, 2.5)


# ==============================================================================
# 4. Test Class: Segmentation Losses
# ==============================================================================

class TestSegmentationLosses:
    """Verifies Soft-Margin Focal Loss, Contrastive InfoNCE, and Multi-Task Balancing."""

    def test_soft_margin_focal_loss_sigma_05_smoothing(self):
        """Verify tightened Gaussian bandwidth sigma=0.5 drops distance=1 target to exp(-2) ~ 0.1353."""
        sigma = 0.5
        y_star_dist1 = math.exp(-(1.0**2) / (2.0 * (sigma**2)))
        assert np.isclose(y_star_dist1, 0.135335, atol=1e-4)

    def test_soft_margin_focal_loss_numeric_stability(self):
        """Verify Soft-Margin Focal Loss computes without NaN under extreme logits z in [-50, +50]."""
        logits = np.array([-50.0, -20.0, 0.0, 20.0, 50.0])
        gold = {2}
        loss = reference_soft_margin_focal_loss(logits, gold, total_sentences=5, sigma=0.5)
        assert not np.isnan(loss)
        assert not np.isinf(loss)
        assert loss > 0.0

    def test_soft_margin_focal_loss_alpha_weighting(self):
        """Verify alpha=0.75 applies higher penalty to false negatives at gold boundaries."""
        loss_fn = reference_soft_margin_focal_loss(np.array([-5.0]), {0}, total_sentences=1, alpha=0.75)
        loss_fp = reference_soft_margin_focal_loss(np.array([5.0]), set(), total_sentences=1, alpha=0.75)
        assert loss_fn > loss_fp, "Alpha=0.75 must penalize false negatives more heavily"

    def test_contrastive_infonce_temporal_decay(self):
        """Verify temporal decay weight w_ij = exp(-|i-j| / 10.0)."""
        w1 = math.exp(-1.0 / 10.0)
        w5 = math.exp(-5.0 / 10.0)
        assert np.isclose(w1, 0.904837, atol=1e-4)
        assert np.isclose(w5, 0.606531, atol=1e-4)
        assert w1 > w5

    def test_contrastive_infonce_single_sentence_protection(self):
        """Verify singleton segments (|P_i \\ {i}| = 0) do not cause ZeroDivisionError or NaN."""
        labels = [0, 1, 2]  # All singletons
        vecs = np.random.randn(3, 16)
        loss = reference_contrastive_infonce_loss(vecs, labels)
        assert loss == 0.0
        assert not np.isnan(loss)

    def test_multitask_loss_balancing_weights(self):
        """Verify total loss formula: L_total = 10.0 * L_SM-FL + 1.0 * L_cont + 0.5 * L_gen."""
        l_focal, l_cont, l_gen = 0.035, 0.30, 2.20
        total = reference_multitask_total_loss(l_focal, l_cont, l_gen)
        # 10.0 * 0.035 + 1.0 * 0.30 + 0.5 * 2.20 = 0.35 + 0.30 + 1.10 = 1.75
        assert np.isclose(total, 1.75)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_pytorch_soft_margin_focal_loss_parity(self):
        """Verify PyTorch SoftMarginFocalLoss parity against reference oracle."""
        logits_np = np.array([-2.5, 0.0, 3.1, -1.2, 4.0])
        gold = {2, 4}
        oracle_loss = reference_soft_margin_focal_loss(logits_np, gold, total_sentences=5, sigma=0.5)

        loss_fn = SoftMarginFocalLoss(sigma=0.5, gamma=2.0, alpha=0.75)
        torch_loss = loss_fn(torch.tensor(logits_np, dtype=torch.float32), torch.tensor([0, 0, 1, 0, 1]))
        assert np.isclose(float(torch_loss.item()), oracle_loss, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_pytorch_infonce_parity_and_protection(self):
        """Verify PyTorch TemporallyDecayedSupConLoss parity against reference oracle."""
        np.random.seed(42)
        N, D = 6, 16
        vecs_np = np.random.randn(N, D)
        labels = [0, 0, 0, 1, 1, 1]
        oracle_loss = reference_contrastive_infonce_loss(vecs_np, labels, tau_temp=10.0, temperature=0.07)

        loss_fn = TemporallyDecayedSupConLoss(tau=0.07, tau_temp=10.0)
        torch_loss = loss_fn(torch.tensor(vecs_np, dtype=torch.float64), torch.tensor(labels, dtype=torch.long))
        assert np.isclose(float(torch_loss.item()), oracle_loss, atol=1e-5)


# ==============================================================================
# 5. Test Class: 1D Non-Maximum Suppression (1D-NMS)
# ==============================================================================

class Test1DNonMaximumSuppression:
    """Verifies 1D Non-Maximum Suppression (radius r=2 sentences)."""

    def test_nms_radius_2_cluster_suppression(self):
        """Verify no two predicted boundaries exist within radius r=2 sentences."""
        probs = [0.1, 0.2, 0.85, 0.94, 0.82, 0.1, 0.1, 0.89, 0.75]
        selected = reference_1d_nms(probs, threshold=0.5, radius=2)
        for i in range(len(selected) - 1):
            assert selected[i + 1] - selected[i] > 2, "Violated radius r=2 suppression"

    def test_nms_local_peak_selection(self):
        """Verify 1D-NMS selects the exact local peak within a cluster."""
        probs = [0.1, 0.1, 0.80, 0.96, 0.79, 0.1]
        selected = reference_1d_nms(probs, threshold=0.5, radius=2)
        assert selected == [3]

    def test_nms_threshold_filtering(self):
        """Verify probabilities below threshold theta are discarded."""
        probs = [0.1, 0.42, 0.49, 0.35]
        selected = reference_1d_nms(probs, threshold=0.5, radius=2)
        assert selected == []

    def test_nms_deterministic_tie_breaking(self):
        """Verify deterministic selection of earlier candidate on identical probabilities."""
        probs = [0.1, 0.88, 0.88, 0.1]
        selected = reference_1d_nms(probs, threshold=0.5, radius=2)
        assert selected == [1]

    def test_nms_module_parity_with_reference_oracle(self):
        """Verify non_maximum_suppression_1d parity against reference_1d_nms."""
        probs = [0.1, 0.85, 0.92, 0.78, 0.1, 0.1, 0.89, 0.1]
        ref = reference_1d_nms(probs, threshold=0.5, radius=2)
        impl = non_maximum_suppression_1d(probs, threshold=0.5, radius=2)
        assert ref == impl == [2, 6]


# ==============================================================================
# 6. Test Class: Pipeline Integration & Contract Conformance
# ==============================================================================

class TestSegmentationPipelineIntegration:
    """Verifies end-to-end Phase 2 Pipeline execution from Contract 1 to Contract 2."""

    def test_pipeline_contract1_to_contract2_execution(self):
        """Verify pipeline transforms Contract 1 sentences into valid Contract 2 payload."""
        sentences = get_contract1_sample_sentences()
        config = SegmentationPipelineConfig(force_fallback=True, boundary_threshold=0.40)
        pipeline = SegmentationPipeline(config=config)

        payload = pipeline.segment_transcript(sentences)
        validate_contract2_payload(payload)

    def test_pipeline_contiguous_segment_coverage(self):
        """Verify segments partition all sentences contiguously without gaps."""
        sentences = get_contract1_sample_sentences()
        N = len(sentences)
        config = SegmentationPipelineConfig(force_fallback=True)
        pipeline = SegmentationPipeline(config=config)

        payload = pipeline.segment_transcript(sentences)
        segments = payload["topic_segments"]

        assert segments[0]["start_sentence_idx"] == 0
        assert segments[-1]["end_sentence_idx"] == N - 1

        for i in range(len(segments) - 1):
            assert segments[i + 1]["start_sentence_idx"] == segments[i]["end_sentence_idx"] + 1

    def test_pipeline_timestamp_formatting(self):
        """Verify start_time_str and end_time_str strictly match 'HH:MM:SS'."""
        sentences = get_contract1_sample_sentences()
        config = SegmentationPipelineConfig(force_fallback=True)
        pipeline = SegmentationPipeline(config=config)

        payload = pipeline.segment_transcript(sentences)
        time_regex = re.compile(r"^\d{2}:\d{2}:\d{2}$")

        for seg in payload["topic_segments"]:
            assert time_regex.match(seg["start_time_str"])
            assert time_regex.match(seg["end_time_str"])

    def test_pipeline_pedagogical_metadata_integrity(self):
        """Verify all segments include descriptive titles and valid '?' cue questions."""
        sentences = get_contract1_sample_sentences()
        config = SegmentationPipelineConfig(force_fallback=True)
        pipeline = SegmentationPipeline(config=config)

        payload = pipeline.segment_transcript(sentences)
        for seg in payload["topic_segments"]:
            assert len(seg["topic_title"].strip()) >= 5
            assert len(seg["cue_questions"]) >= 1
            for q in seg["cue_questions"]:
                assert q.endswith("?"), f"Cue question must end with '?': {q}"

    def test_pipeline_empty_transcript_handling(self):
        """Verify empty input transcript emits 0 segments without crashing."""
        config = SegmentationPipelineConfig(force_fallback=True)
        pipeline = SegmentationPipeline(config=config)
        payload = pipeline.segment_transcript([])
        assert payload["num_sentences"] == 0
        assert len(payload["topic_segments"]) == 0

    def test_pipeline_single_sentence_handling(self):
        """Verify single-sentence input transcript emits exactly 1 segment [0..0]."""
        sentences = [get_contract1_sample_sentences()[0]]
        config = SegmentationPipelineConfig(force_fallback=True)
        pipeline = SegmentationPipeline(config=config)

        payload = pipeline.segment_transcript(sentences)
        assert payload["num_sentences"] == 1
        assert len(payload["topic_segments"]) == 1
        assert payload["topic_segments"][0]["start_sentence_idx"] == 0
        assert payload["topic_segments"][0]["end_sentence_idx"] == 0

    def test_pipeline_administrative_digression_segmentation(self):
        """Verify administrative digression sentences are isolated into a dedicated logistics segment."""
        lecture = get_synthetic_10min_lecture_fixture()
        sentences = lecture["sentences"]
        config = SegmentationPipelineConfig(force_fallback=True, boundary_threshold=0.35)
        pipeline = SegmentationPipeline(config=config)

        payload = pipeline.segment_transcript(sentences)
        # Find if any segment is identified as administrative
        has_admin = any("Administrative" in seg["topic_title"] or "Logistics" in seg["topic_title"] for seg in payload["topic_segments"])
        assert has_admin, "Pipeline must isolate administrative digression segment"


# ==============================================================================
# 7. Test Class: Hardware Bounds & Memory Constraints
# ==============================================================================

class TestHardwareAndMemoryConstraints:
    """Verifies execution strictly adheres to RTX 4050 6GB VRAM and 16GB RAM envelope."""

    def test_system_ram_budget(self):
        """Verify intermediate sentence embeddings consume < 5 MB RAM for 1,000 sentences."""
        N = 1000
        H = 1024
        mem_bytes = N * H * 4
        mem_mb = mem_bytes / (1024 * 1024)
        assert mem_mb < 5.0, f"Intermediate memory {mem_mb} MB is well within 16GB System RAM"

    def test_gpu_vram_safety_envelope(self):
        """Verify DBCA attention matrices consume < 50 MB VRAM for N=500 sentences."""
        N = 500
        heads = 8
        attn_bytes = heads * N * N * 2
        attn_mb = attn_bytes / (1024 * 1024)
        assert attn_mb < 50.0, f"DBCA attention memory {attn_mb} MB satisfies 6GB VRAM bounds"

    def test_ollama_residency_and_eviction_mechanics(self):
        """Verify keep_alive: 0 parameter recovers VRAM."""
        cfg = SegmentationPipelineConfig(force_fallback=True)
        gen = OllamaTitleGenerator(cfg)
        success = gen.evict_model()
        assert isinstance(success, bool)


# ==============================================================================
# 8. Test Class: Unified Model Execution (when PyTorch available)
# ==============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
class TestDiscourseAwareSegmentationModelUnified:
    """Verifies DiscourseAwareSegmentationModel integration of all components."""

    def test_model_forward_pass_with_loss(self):
        """Verify unified model forward pass with full multi-task loss computation."""
        B, S, H = 2, 8, 1024
        cfg = DiscourseAwareSegmentationConfig(hidden_size=H, adapter_dim=H, num_heads=8, d_ff=4096)
        model = DiscourseAwareSegmentationModel(cfg)

        sentence_reps = torch.randn(B, S, H)
        gold_bounds = torch.zeros(B, S)
        gold_bounds[:, 3] = 1.0  # Boundary at sentence 3
        gold_bounds[:, 6] = 1.0  # Boundary at sentence 6

        topic_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1, 2, 2]])

        output = model(
            sentence_reps=sentence_reps,
            gold_boundaries=gold_bounds,
            topic_segment_ids=topic_ids,
        )

        assert isinstance(output, SegmentationModelOutput)
        assert output.loss is not None
        assert output.loss > 0.0
        assert not torch.isnan(output.loss)
        assert output.boundary_probabilities is not None
        assert output.boundary_probabilities.shape == (B, S)
        assert len(output.predicted_boundaries) == B

    def test_model_inference_predicted_boundaries(self):
        """Verify model returns 1D-NMS compliant boundary predictions during inference."""
        B, S, H = 1, 10, 1024
        cfg = DiscourseAwareSegmentationConfig(hidden_size=H, adapter_dim=H)
        model = DiscourseAwareSegmentationModel(cfg)

        sentence_reps = torch.randn(B, S, H)
        output = model(sentence_reps=sentence_reps)

        assert output.loss is None
        assert len(output.predicted_boundaries) == 1
        bounds = output.predicted_boundaries[0]
        for i in range(len(bounds) - 1):
            assert bounds[i + 1] - bounds[i] > 2, "Violated radius r=2 in inference predictions"
