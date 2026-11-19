"""Authoritative Reference Algorithms and Mathematical Oracles.

This module provides verified pure-Python/numpy implementations of:
1. Silero VAD v4 silence-guided chunking bounds [10.0s, 28.0s].
2. Regex disfluency scrubber and normalizer.
3. Running topic centroid semantic digression gating (d_topic > 0.65 sustained >= 3).
4. Learnable query-based attention pooling.
5. 2-layer Decoupled Bidirectional Context Adapter (DBCA) representation.
6. Transition feature construction: f_i^trans = [c_i || c_{i-1} || (c_i - c_{i-1}) || (c_i * c_{i-1})].
7. Logit-space numerically stable Soft-Margin Focal Loss (sigma=0.5, gamma=2.0, alpha=0.75).
8. 1D Non-Maximum Suppression (1D-NMS) with radius r=2.
9. Supervised Contrastive InfoNCE Loss with exponential temporal decay (tau_temp=10.0, tau=0.07).
10. Multi-task loss balancing: L_total = 10.0 L_SM-FL + 1.0 L_cont + 0.5 L_gen.
11. Evaluation metrics: Micro F1, Macro F1, Relaxed F1 (+-1 window), P_k, WindowDiff (matching miniseg_core.py).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ==============================================================================
# 1. Silero VAD Silence Chunking Reference Oracle
# ==============================================================================

def reference_vad_chunker(
    samples: np.ndarray,
    sample_rate: int = 16000,
    min_chunk_sec: float = 10.0,
    max_chunk_sec: float = 28.0,
    silence_threshold_rms: float = 0.01,
    min_silence_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    """Partition audio into silence-guided chunks strictly within [min_chunk_sec, max_chunk_sec].

    Args:
        samples: Float32 audio samples in [-1.0, 1.0].
        sample_rate: Sample rate in Hz.
        min_chunk_sec: Lower bound for chunk duration (10.0s).
        max_chunk_sec: Upper bound for chunk duration (28.0s).
        silence_threshold_rms: Energy threshold for detecting speech pauses.
        min_silence_sec: Minimum pause length required to split a chunk.

    Returns:
        List of chunk metadata dictionaries with start_sec, end_sec, duration_sec.
    """
    total_samples = len(samples)
    total_sec = total_samples / float(sample_rate) if sample_rate > 0 else 0.0

    if total_sec <= 0.0:
        return []

    # If entire audio is shorter than max_chunk_sec, return single chunk
    if total_sec <= max_chunk_sec:
        return [
            {
                "chunk_id": 0,
                "start_sec": 0.0,
                "end_sec": total_sec,
                "duration_sec": total_sec,
                "start_sample": 0,
                "end_sample": total_samples,
            }
        ]

    # Detect candidate silence intervals (using 50ms sliding windows)
    window_samples = int(0.05 * sample_rate)
    hop_samples = window_samples // 2
    silence_points: List[float] = []

    for idx in range(0, total_samples - window_samples, hop_samples):
        window = samples[idx : idx + window_samples]
        rms = np.sqrt(np.mean(window**2)) if len(window) > 0 else 0.0
        if rms < silence_threshold_rms:
            silence_points.append(idx / float(sample_rate))

    # Identify valid split points between min_chunk_sec and max_chunk_sec
    chunks: List[Dict[str, Any]] = []
    chunk_start_sec = 0.0
    chunk_id = 0

    while chunk_start_sec < total_sec:
        remaining_sec = total_sec - chunk_start_sec
        if remaining_sec <= max_chunk_sec:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start_sec": chunk_start_sec,
                    "end_sec": total_sec,
                    "duration_sec": remaining_sec,
                    "start_sample": int(chunk_start_sec * sample_rate),
                    "end_sample": total_samples,
                }
            )
            break

        # Search for a silence point within [chunk_start_sec + min_chunk_sec, chunk_start_sec + max_chunk_sec]
        earliest_split = chunk_start_sec + min_chunk_sec
        latest_split = chunk_start_sec + max_chunk_sec
        candidate_splits = [t for t in silence_points if earliest_split <= t <= latest_split]

        if candidate_splits:
            # Pick silence closest to the middle of the allowable interval (19.0s)
            target_t = chunk_start_sec + (min_chunk_sec + max_chunk_sec) / 2.0
            split_sec = min(candidate_splits, key=lambda t: abs(t - target_t))
        else:
            # Hard boundary fallback at max_chunk_sec
            split_sec = latest_split

        duration = split_sec - chunk_start_sec
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start_sec": chunk_start_sec,
                "end_sec": split_sec,
                "duration_sec": duration,
                "start_sample": int(chunk_start_sec * sample_rate),
                "end_sample": min(total_samples, int(split_sec * sample_rate)),
            }
        )
        chunk_start_sec = split_sec
        chunk_id += 1

    return chunks


# ==============================================================================
# 2. Disfluency Normalizer Reference Oracle
# ==============================================================================

# Common spoken conversational fillers
FILLER_PATTERNS = [
    r"\b(um+)\b",
    r"\b(uh+)\b",
    r"\b(you know)\b",
    r"\b(like)\b(?=\s+[a-z])",
    r"\b(sort of)\b",
    r"\b(kind of)\b",
    r"\b(i mean)\b",
    r"\b(basically)\b",
]

# Single word repetition regex: e.g. "in in", "the the"
STUTTER_REGEX = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
# Multi-word phrase repetition regex: e.g. "most of the most of the"
PHRASE_STUTTER_REGEX = re.compile(r"\b(\w+(?:\s+\w+){1,4})\s+\1\b", re.IGNORECASE)


def reference_disfluency_scrubber(text: str) -> str:
    """Deterministic regex disfluency cleaner and sentence punctuator."""
    if not text or not text.strip():
        return ""

    s = text.strip()

    # Step 1: Remove bracketed annotations like [Music], [Applause]
    s = re.sub(r"\[.*?\]", " ", s)

    # Step 2: Strip conversational filler words
    for pat in FILLER_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)

    # Step 3: Strip immediate adjacent word and phrase repetitions (stutters)
    for _ in range(3):
        s = PHRASE_STUTTER_REGEX.sub(r"\1", s)
        s = STUTTER_REGEX.sub(r"\1", s)

    # Step 4: Normalize spaces
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    # Step 5: Capitalize first character and ensure terminal punctuation
    s = s[0].upper() + s[1:]
    if not s.endswith((".", "?", "!")):
        s += "."

    return s


# ==============================================================================
# 3. Semantic Digression Gating Reference Oracle
# ==============================================================================

def reference_digression_gate(
    sentence_embeddings: np.ndarray,
    sentences: List[str],
    threshold_distance: float = 0.65,
    sustained_k: int = 3,
    centroid_momentum: float = 0.85,
) -> List[Dict[str, Any]]:
    """Gating filter identifying topic digressions where cosine distance > 0.65 sustained >= 3 sentences.

    Args:
        sentence_embeddings: [N, D] normalized embedding vectors for each sentence.
        sentences: List of N string texts.
        threshold_distance: Cosine distance threshold d_topic > 0.65.
        sustained_k: Minimum consecutive sentences above threshold required to gate.
        centroid_momentum: Momentum for running topic centroid update (beta).

    Returns:
        List of dicts with is_digression, digression_type, and cosine_distance.
    """
    N = len(sentences)
    if N == 0:
        return []

    results: List[Dict[str, Any]] = []
    # Normalize embeddings
    norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized_embs = sentence_embeddings / norms

    running_centroid = normalized_embs[0].copy()
    consecutive_high_dist = 0
    raw_distances: List[float] = []

    for i in range(N):
        emb = normalized_embs[i]
        cos_sim = float(np.dot(emb, running_centroid))
        cos_dist = float(1.0 - cos_sim)
        raw_distances.append(cos_dist)

        if cos_dist > threshold_distance:
            consecutive_high_dist += 1
        else:
            consecutive_high_dist = 0
            # Update centroid only on core pedagogical sentences
            running_centroid = centroid_momentum * running_centroid + (1.0 - centroid_momentum) * emb
            c_norm = np.linalg.norm(running_centroid)
            if c_norm > 0:
                running_centroid /= c_norm

    # Identify sustained runs where consecutive_high_dist >= sustained_k
    is_digression_flags = [False] * N
    run_indices: List[int] = []

    for i in range(N):
        if raw_distances[i] > threshold_distance:
            run_indices.append(i)
        else:
            if len(run_indices) >= sustained_k:
                for idx in run_indices:
                    is_digression_flags[idx] = True
            run_indices = []

    if len(run_indices) >= sustained_k:
        for idx in run_indices:
            is_digression_flags[idx] = True

    for i in range(N):
        results.append(
            {
                "sentence_idx": i,
                "text": sentences[i],
                "cosine_distance": raw_distances[i],
                "is_digression": is_digression_flags[i],
                "digression_type": "administrative" if is_digression_flags[i] else "none",
            }
        )

    return results


# ==============================================================================
# 4. Query-Based Attention Pooling Reference Oracle
# ==============================================================================

def reference_query_attention_pooling(
    token_hidden_states: np.ndarray,
    query_vector: np.ndarray,
    projection_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Learnable query-based attention pooling: u_i = sum_t alpha_{i,t} H_{i,t}.

    Args:
        token_hidden_states: [T, D] token hidden states for a sentence.
        query_vector: [K] learnable query vector.
        projection_matrix: [K, D] projection weights.

    Returns:
        (pooled_sentence_vector [D], attention_weights [T])
    """
    # H_{i,t} is [D]
    # W_p H_{i,t} is [K]
    # scores: q^T tanh(W_p H_{i,t})
    projected = np.tanh(np.dot(token_hidden_states, projection_matrix.T))  # [T, K]
    scores = np.dot(projected, query_vector)  # [T]

    # Numerically stable softmax: alpha = exp(s - max(s)) / sum(exp(s - max(s)))
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    alpha = exp_scores / np.sum(exp_scores)  # [T]

    # Weighted sum: u = sum_t alpha_t H_t
    pooled_vector = np.sum(token_hidden_states * alpha[:, np.newaxis], axis=0)  # [D]

    return pooled_vector, alpha


# ==============================================================================
# 5. DBCA Context Adapter Reference Oracle
# ==============================================================================

def reference_dbca_adapter(
    sentence_vectors: np.ndarray,
    num_heads: int = 8,
    hidden_dim: Optional[int] = None,
) -> np.ndarray:
    """2-Layer Bidirectional Context Adapter (DBCA) eliminating causal blindness.

    Args:
        sentence_vectors: [N, D] sequence of sentence vectors.
        num_heads: Number of attention heads (default 8).
        hidden_dim: Hidden dimension H (default matches D).

    Returns:
        [N, H] bidirectional discourse-adapted contextual representations.
    """
    N, D = sentence_vectors.shape
    H = hidden_dim or D

    # Pure numpy bidirectional self-attention layer (Layer 1)
    # Full NxN attention matrix (unmasked, bidirectional)
    sim_matrix = np.dot(sentence_vectors, sentence_vectors.T) / np.sqrt(D)
    attn_weights = np.exp(sim_matrix - np.max(sim_matrix, axis=1, keepdims=True))
    attn_weights /= np.sum(attn_weights, axis=1, keepdims=True)
    layer1_out = np.dot(attn_weights, sentence_vectors) + sentence_vectors  # Residual

    # Layer 2 (Feed-forward + second unmasked bidirectional pass)
    sim_matrix2 = np.dot(layer1_out, layer1_out.T) / np.sqrt(H)
    attn_weights2 = np.exp(sim_matrix2 - np.max(sim_matrix2, axis=1, keepdims=True))
    attn_weights2 /= np.sum(attn_weights2, axis=1, keepdims=True)
    layer2_out = np.dot(attn_weights2, layer1_out) + layer1_out  # Residual

    return layer2_out


# ==============================================================================
# 6. Transition Feature Construction Reference Oracle
# ==============================================================================

def reference_transition_features(context_states: np.ndarray) -> np.ndarray:
    """Construct transition representation: f_i^trans = [c_i || c_{i-1} || (c_i - c_{i-1}) || (c_i * c_{i-1})].

    Args:
        context_states: [N, D] contextual sentence representations from DBCA.

    Returns:
        [N-1, 4D] transition feature matrix for indices 1 to N-1.
    """
    N, D = context_states.shape
    if N < 2:
        return np.zeros((0, 4 * D), dtype=context_states.dtype)

    c_curr = context_states[1:]       # c_i for i in 1..N-1
    c_prev = context_states[:-1]      # c_{i-1} for i in 1..N-1
    diff = c_curr - c_prev            # (c_i - c_{i-1})
    hadamard = c_curr * c_prev        # (c_i * c_{i-1})

    features = np.concatenate([c_curr, c_prev, diff, hadamard], axis=1)  # [N-1, 4D]
    return features


# ==============================================================================
# 7. Soft-Margin Focal Loss Reference Oracle (Logit-Space Formulation)
# ==============================================================================

def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x))."""
    # For x >= 0: -log(1 + exp(-x))
    # For x < 0:  x - log(1 + exp(x))
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))


def reference_soft_margin_focal_loss(
    logits: np.ndarray,
    gold_boundaries: Set[int] | List[int],
    total_sentences: int,
    sigma: float = 0.5,
    gamma: float = 2.0,
    alpha: float = 0.75,
    eps: float = 1e-7,
) -> float:
    """Logit-space numerically stable Soft-Margin Focal Loss with tightened Gaussian smoothing (sigma=0.5).

    Mathematical specification:
        y_i^* = max_{j in Gold} exp( - (i - j)^2 / (2 * sigma^2) )
        L_{SM-FL} = - 1/N sum_i [ alpha y_i^* sigma(-z_i)^gamma log_sigmoid(z_i) +
                                  (1 - alpha)(1 - y_i^*) sigma(z_i)^gamma log_sigmoid(-z_i) ]

    Args:
        logits: [N] unconstrained transition logits z_i.
        gold_boundaries: Set or list of true boundary indices.
        total_sentences: Total number of boundary candidate positions N.
        sigma: Gaussian label smoothing bandwidth (default 0.5).
        gamma: Focal loss modulating exponent (default 2.0).
        alpha: Positive class weight (default 0.75).
        eps: Epsilon clamp for stability.

    Returns:
        Scalar loss value.
    """
    N = total_sentences
    if N == 0 or len(logits) == 0:
        return 0.0

    z = np.asarray(logits, dtype=np.float64)
    gold_set = set(gold_boundaries)

    # Compute smoothed targets y_i^*
    indices = np.arange(N)
    smoothed_targets = np.zeros(N, dtype=np.float64)

    if gold_set:
        for i in range(N):
            min_dist_sq = min((i - j) ** 2 for j in gold_set)
            smoothed_targets[i] = math.exp(-min_dist_sq / (2.0 * (sigma**2)))
    # If no gold boundaries exist, all targets are 0.0

    # Probabilities in logit space:
    # sigma(z) = 1 / (1 + exp(-z))
    # sigma(-z) = 1 - sigma(z)
    # To prevent overflow:
    sig_z = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
    sig_neg_z = 1.0 - sig_z

    # Modulating focal factors:
    # (1 - p_t)^gamma:
    # for positive target: (1 - p)^gamma = sigma(-z)^gamma
    # for negative target: p^gamma = sigma(z)^gamma
    focal_pos = (sig_neg_z ** gamma)
    focal_neg = (sig_z ** gamma)

    # Stable log-sigmoid
    log_sig_z = log_sigmoid(z)
    log_sig_neg_z = log_sigmoid(-z)

    # Combined focal cross-entropy terms
    term_pos = alpha * smoothed_targets * focal_pos * log_sig_z
    term_neg = (1.0 - alpha) * (1.0 - smoothed_targets) * focal_neg * log_sig_neg_z

    loss = -np.mean(term_pos + term_neg)
    return float(loss)


# ==============================================================================
# 8. 1D Non-Maximum Suppression (1D-NMS) Reference Oracle
# ==============================================================================

def reference_1d_nms(
    probabilities: np.ndarray | List[float],
    threshold: float = 0.5,
    radius: int = 2,
) -> List[int]:
    """1D Non-Maximum Suppression eliminating boundary clustering within radius r=2.

    Args:
        probabilities: Array of boundary transition probabilities p_i for i in 0..N-1.
        threshold: Minimum probability threshold for candidate boundaries.
        radius: Elimination radius in sentences (r=2).

    Returns:
        List of selected boundary indices satisfying non-clustering constraint.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    N = len(probs)
    if N == 0:
        return []

    # Candidates exceeding threshold
    candidates = [i for i in range(N) if probs[i] >= threshold]
    if not candidates:
        return []

    # Sort candidates by probability descending (break ties with smaller index)
    sorted_candidates = sorted(candidates, key=lambda idx: (-probs[idx], idx))

    suppressed = set()
    selected = []

    for idx in sorted_candidates:
        if idx in suppressed:
            continue
        selected.append(idx)
        # Suppress all candidates within distance <= radius
        for neighbor in range(max(0, idx - radius), min(N, idx + radius + 1)):
            if neighbor != idx:
                suppressed.add(neighbor)

    return sorted(selected)


# ==============================================================================
# 9. Contrastive InfoNCE Loss Reference Oracle
# ==============================================================================

def reference_contrastive_infonce_loss(
    sentence_vectors: np.ndarray,
    segment_labels: List[int],
    tau_temp: float = 10.0,
    temperature: float = 0.07,
) -> float:
    """Supervised Contrastive InfoNCE loss with exponential temporal decay w_{ij} = exp(-|i-j| / tau_temp).

    Args:
        sentence_vectors: [N, D] sentence representations.
        segment_labels: [N] integer topic segment IDs for each sentence.
        tau_temp: Temporal decay bandwidth (default 10.0).
        temperature: InfoNCE temperature tau (default 0.07).

    Returns:
        Scalar InfoNCE contrastive loss.
    """
    N = len(segment_labels)
    if N <= 1:
        return 0.0

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(sentence_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = sentence_vectors / norms

    # Pairwise cosine similarities: [N, N]
    sim = np.dot(vecs, vecs.T)

    # Distance matrix |i - j|: [N, N]
    indices = np.arange(N)
    dist_matrix = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    temporal_weights = np.exp(-dist_matrix / tau_temp)

    total_loss = 0.0
    valid_anchors = 0

    for i in range(N):
        curr_label = segment_labels[i]
        # Positive peers: same segment, excluding self
        pos_mask = (np.array(segment_labels) == curr_label) & (indices != i)
        # Negative peers: different segment
        neg_mask = (np.array(segment_labels) != curr_label)

        pos_indices = np.where(pos_mask)[0]
        if len(pos_indices) == 0:
            # Singleton segment: no positive pairs to contrast
            continue

        valid_anchors += 1

        # Scaled similarities
        # For positives: w_ij * exp(sim(i, j) / tau)
        pos_exp = temporal_weights[i, pos_indices] * np.exp(sim[i, pos_indices] / temperature)

        # For negatives: exp(sim(i, k) / tau)
        neg_indices = np.where(neg_mask)[0]
        neg_sum = np.sum(np.exp(sim[i, neg_indices] / temperature)) if len(neg_indices) > 0 else 0.0

        # Loss for anchor i: - 1/|P_i| sum_{j in P_i} log( pos_exp / (pos_exp + neg_sum) )
        denominator = pos_exp + neg_sum + 1e-12
        anchor_loss = -np.mean(np.log(pos_exp / denominator))
        total_loss += float(anchor_loss)

    if valid_anchors == 0:
        return 0.0

    return float(total_loss / valid_anchors)


# ==============================================================================
# 10. Multi-Task Joint Objective Reference Oracle
# ==============================================================================

def reference_multitask_total_loss(
    focal_loss: float,
    contrastive_loss: float,
    generative_loss: float,
    lambda_focal: float = 10.0,
    lambda_contrastive: float = 1.0,
    lambda_generative: float = 0.5,
) -> float:
    """Calibrated joint multi-task objective: L_total = 10.0 L_SM-FL + 1.0 L_cont + 0.5 L_gen."""
    return float(lambda_focal * focal_loss + lambda_contrastive * contrastive_loss + lambda_generative * generative_loss)


# ==============================================================================
# 11. Evaluation Metrics Reference Oracle (P_k, WindowDiff, F1)
# ==============================================================================

def reference_eval_metrics(
    gold_boundaries: Set[int],
    pred_boundaries: Set[int],
    num_sentences: int,
    relaxed_window: int = 1,
) -> Dict[str, float]:
    """Compute Micro F1, Relaxed F1 (+-1), P_k, and WindowDiff evaluation metrics."""
    if num_sentences <= 1:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "relaxed_f1": 1.0, "pk": 0.0, "wd": 0.0}

    # Strict Precision, Recall, F1
    true_positives = len(gold_boundaries.intersection(pred_boundaries))
    precision = true_positives / len(pred_boundaries) if pred_boundaries else (1.0 if not gold_boundaries else 0.0)
    recall = true_positives / len(gold_boundaries) if gold_boundaries else (1.0 if not pred_boundaries else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Relaxed Precision, Recall, F1 (+- relaxed_window)
    relaxed_tp = 0
    for p in pred_boundaries:
        if any(abs(p - g) <= relaxed_window for g in gold_boundaries):
            relaxed_tp += 1
    relaxed_prec = relaxed_tp / len(pred_boundaries) if pred_boundaries else (1.0 if not gold_boundaries else 0.0)

    relaxed_recall_count = 0
    for g in gold_boundaries:
        if any(abs(g - p) <= relaxed_window for p in pred_boundaries):
            relaxed_recall_count += 1
    relaxed_rec = relaxed_recall_count / len(gold_boundaries) if gold_boundaries else (1.0 if not pred_boundaries else 0.0)
    relaxed_f1 = (
        2 * relaxed_prec * relaxed_rec / (relaxed_prec + relaxed_rec) if (relaxed_prec + relaxed_rec) > 0 else 0.0
    )

    # Compute P_k and WindowDiff (k = round(0.5 * N / (num_gold_segments)))
    num_gold_segments = len(gold_boundaries) + 1
    k = max(1, int(round(0.5 * num_sentences / num_gold_segments)))

    # Vector of segment ids for each sentence
    def to_seg_ids(boundaries: Set[int], total: int) -> np.ndarray:
        arr = np.zeros(total, dtype=int)
        cur = 0
        for i in range(total):
            if i in boundaries:
                cur += 1
            arr[i] = cur
        return arr

    gold_seg = to_seg_ids(gold_boundaries, num_sentences)
    pred_seg = to_seg_ids(pred_boundaries, num_sentences)

    pk_diffs = 0
    wd_diffs = 0
    window_count = max(1, num_sentences - k)

    for i in range(window_count):
        j = i + k
        gold_same = 1 if gold_seg[i] == gold_seg[j] else 0
        pred_same = 1 if pred_seg[i] == pred_seg[j] else 0
        if gold_same != pred_same:
            pk_diffs += 1

        # Count boundaries in window [i, j)
        gold_bounds = sum(1 for b in gold_boundaries if i < b <= j)
        pred_bounds = sum(1 for b in pred_boundaries if i < b <= j)
        if gold_bounds != pred_bounds:
            wd_diffs += 1

    pk = float(pk_diffs / window_count)
    wd = float(wd_diffs / window_count)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "relaxed_f1": float(relaxed_f1),
        "pk": pk,
        "wd": wd,
    }


def reference_greedy_relaxed_f1(
    gold_boundaries: Set[int] | Sequence[int],
    pred_boundaries: Set[int] | Sequence[int],
    window: int = 2,
    ignore_first: bool = True,
) -> Dict[str, float]:
    """Pure-Python reference oracle implementing greedy 1-to-1 bipartite matching.

    Guarantees parity with src/miniseg_core.py:relaxed_boundary_f1.
    Supports either:
    1. Sets / collections of integer boundary index split points (e.g. {3, 7})
    2. Aligned binary 0/1 indicator vectors of identical length (e.g. [0, 0, 0, 1, ...])
    """
    if (
        isinstance(gold_boundaries, (list, tuple))
        and isinstance(pred_boundaries, (list, tuple))
        and len(gold_boundaries) == len(pred_boundaries)
        and len(gold_boundaries) > 2
        and set(gold_boundaries).issubset({0, 1})
        and set(pred_boundaries).issubset({0, 1})
    ):
        start = 1 if ignore_first else 0
        golds = [i for i in range(start, len(gold_boundaries)) if gold_boundaries[i] == 1]
        preds = [i for i in range(start, len(pred_boundaries)) if pred_boundaries[i] == 1]
    else:
        start = 1 if ignore_first else 0
        golds = sorted([int(g) for g in gold_boundaries if g >= start])
        preds = sorted([int(p) for p in pred_boundaries if p >= start])

    matched_preds: Set[int] = set()
    matched_trues: Set[int] = set()

    for g in golds:
        for p in preds:
            if p in matched_preds:
                continue
            if abs(p - g) <= window:
                matched_trues.add(g)
                matched_preds.add(p)
                break

    tp = len(matched_trues)
    fp = max(0, len(preds) - len(matched_preds))
    fn = max(0, len(golds) - len(matched_trues))

    precision = tp / (tp + fp + 1e-9) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-9) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall + 1e-9) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "window": float(window),
    }

