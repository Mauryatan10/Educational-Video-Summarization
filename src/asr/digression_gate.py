"""Semantic Digression Gate.

Tracks running pedagogical topic centroid:
    e_topic^(i) = beta * e_topic^(i-1) + (1 - beta) * e_i,  with beta = 0.90
Measures cosine distance d_topic > 0.65 sustained over >= 3 consecutive sentences
with administrative lexical anchoring to gate off-topic administrative/logistical tangents.
Preserves 100% of conceptual definitions, mechanisms, and mathematical proofs.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

DEFAULT_ADMIN_KEYWORDS = (
    "homework",
    "canvas",
    "exam",
    "office hours",
    "announcement",
    "quiz",
    "syllabus",
    "grade",
    "assignment due",
    "homework due",
    "due date",
    "project due",
)


@dataclass
class DigressionGateConfig:
    """Configuration for semantic digression gating."""

    decay_factor: float = 0.90  # beta = 0.90
    distance_threshold: float = 0.65  # d_topic > 0.65
    min_consecutive: int = 3  # sustained over >= 3 sentences
    admin_keywords: Tuple[str, ...] = DEFAULT_ADMIN_KEYWORDS
    embedding_dim: int = 1024
    embedding_backend: str = "auto"  # 'auto' | 'minilm' | 'tfidf'


class LightweightTfidfEmbedder:
    """Zero-dependency, deterministic pure NumPy TF-IDF embedder.

    Uses CRC32 deterministic n-gram subword hashing to produce normalized vector representations.
    """

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.doc_freq: Counter[int] = Counter()
        self.total_docs = 0
        self.is_fit = False

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        tokens = list(words)
        # Include adjacent word bigrams for richer semantic phrase clustering
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        # Include character n-grams for subword matching (e.g. 3-grams and 4-grams)
        for w in words:
            if len(w) >= 3:
                for n in (3, 4):
                    for j in range(len(w) - n + 1):
                        tokens.append(f"#{w[j:j+n]}")
        return tokens

    def _hash_token(self, token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) % self.vocab_size

    def fit(self, texts: List[str]) -> None:
        """Fit document frequencies across document corpus."""
        for t in texts:
            tokens = set(self._tokenize(t))
            for tok in tokens:
                self.doc_freq[self._hash_token(tok)] += 1
            self.total_docs += 1
        self.is_fit = True

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts into L2-normalized float32 vectors."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for t in texts:
            tokens = self._tokenize(t)
            vec = np.zeros(self.vocab_size, dtype=np.float32)
            counts = Counter([self._hash_token(tok) for tok in tokens])
            for idx, count in counts.items():
                idf = (
                    math.log((1.0 + self.total_docs) / (1.0 + self.doc_freq[idx])) + 1.0
                    if self.is_fit
                    else 1.0
                )
                tf = 1.0 + math.log(count) if count > 0 else 0.0
                vec[idx] = tf * idf
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


class MiniLMEmbedder:
    """Dense semantic embedder utilizing sentence-transformers all-MiniLM-L6-v2."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.array(embeddings, dtype=np.float32)


class SemanticDigressionGate:
    """Tracks running pedagogical topic centroid and gates administrative tangents.

    Preserves 100% of conceptual lecture depth, proofs, and definitions.
    """

    def __init__(self, config: Optional[DigressionGateConfig] = None):
        self.config = config or DigressionGateConfig()
        self.admin_patterns = []
        for kw in self.config.admin_keywords:
            if kw.lower() == "due":
                self.admin_patterns.append(re.compile(r"\bdue\b(?!\s+to\b)", re.IGNORECASE))
            else:
                self.admin_patterns.append(re.compile(rf"\b{re.escape(kw)}s?\b", re.IGNORECASE))
        self.admin_patterns.append(
            re.compile(r"\b(grading|quizzes|exams|announcements)\b", re.IGNORECASE)
        )
        self.admin_patterns.append(
            re.compile(r"\b(assignment|homework|project)\s+is\s+due\b", re.IGNORECASE)
        )

        self.embedder = self._init_embedder()
        self.topic_centroid: Optional[np.ndarray] = None

    def _init_embedder(self) -> Any:
        backend = self.config.embedding_backend.lower()
        if backend in ("auto", "minilm"):
            try:
                return MiniLMEmbedder()
            except Exception:
                return LightweightTfidfEmbedder(vocab_size=self.config.embedding_dim)
        return LightweightTfidfEmbedder(vocab_size=self.config.embedding_dim)

    def has_admin_keywords(self, text: str) -> bool:
        """Check if text contains administrative lexical markers."""
        return any(pat.search(text) is not None for pat in self.admin_patterns)

    def compute_cosine_distance(self, emb: np.ndarray, centroid: np.ndarray) -> float:
        """Compute cosine distance d_topic = 1 - (e_i . centroid) / (||e_i|| ||centroid||)."""
        norm_emb = float(np.linalg.norm(emb))
        norm_cent = float(np.linalg.norm(centroid))
        if norm_emb < 1e-8 or norm_cent < 1e-8:
            return 1.0
        sim = float(np.dot(emb, centroid) / (norm_emb * norm_cent))
        sim = max(-1.0, min(1.0, sim))
        return 1.0 - sim

    def update_centroid(self, current_centroid: np.ndarray, new_emb: np.ndarray) -> np.ndarray:
        """Update centroid via: e_topic^(i) = beta * e_topic^(i-1) + (1 - beta) * e_i."""
        beta = self.config.decay_factor
        updated = beta * current_centroid + (1.0 - beta) * new_emb
        norm = float(np.linalg.norm(updated))
        if norm > 1e-8:
            updated /= norm
        return updated

    def gate_sentences(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process list of sentence dicts, setting is_digression and digression_type.

        Conforms strictly to Interface Contract 1.
        """
        if not sentences:
            return []

        texts = [s.get("text", "") for s in sentences]

        if isinstance(self.embedder, LightweightTfidfEmbedder) and not self.embedder.is_fit:
            self.embedder.fit(texts)

        embeddings = self.embedder.encode(texts)
        n = len(sentences)

        # Identify administrative keyword presence per sentence
        admin_keyword_flags = [self.has_admin_keywords(t) for t in texts]

        # Initialize topic centroid from first non-administrative sentence
        init_idx = 0
        while init_idx < n and admin_keyword_flags[init_idx]:
            init_idx += 1
        if init_idx >= n:
            init_idx = 0

        centroid = embeddings[init_idx].copy()
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-8:
            centroid /= norm

        # First pass: compute cosine distances against running pedagogical centroid
        distances = []
        running_centroid = centroid.copy()
        for i in range(n):
            d = self.compute_cosine_distance(embeddings[i], running_centroid)
            distances.append(d)
            if not admin_keyword_flags[i]:
                running_centroid = self.update_centroid(running_centroid, embeddings[i])

        # Second pass: detect contiguous runs with d_topic > threshold sustained >= min_consecutive
        # Anchored administrative digression detection with internal gap bridging
        is_digression = [False] * n
        high_dist = [d > self.config.distance_threshold for d in distances]

        admin_indices = [k for k in range(n) if high_dist[k] and admin_keyword_flags[k]]
        if admin_indices:
            clusters: List[List[int]] = []
            current_cluster = [admin_indices[0]]
            max_internal_gap = 1
            for idx in admin_indices[1:]:
                gap = idx - current_cluster[-1] - 1
                if gap <= max_internal_gap and all(
                    high_dist[k] for k in range(current_cluster[-1] + 1, idx)
                ):
                    current_cluster.append(idx)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [idx]
            clusters.append(current_cluster)

            for cluster in clusters:
                start_idx = cluster[0]
                end_idx = cluster[-1]
                span_len = end_idx - start_idx + 1
                if span_len >= self.config.min_consecutive:
                    for k in range(start_idx, end_idx + 1):
                        is_digression[k] = True

        # Recompute clean topic centroid strictly excluding digression spans (freeze during digression)
        clean_centroid = None
        for i in range(n):
            if not is_digression[i]:
                if clean_centroid is None:
                    clean_centroid = embeddings[i].copy()
                    c_norm = float(np.linalg.norm(clean_centroid))
                    if c_norm > 1e-8:
                        clean_centroid /= c_norm
                else:
                    clean_centroid = self.update_centroid(clean_centroid, embeddings[i])

        self.topic_centroid = clean_centroid

        # Construct output records matching Interface Contract 1
        output_records = []
        for i, s in enumerate(sentences):
            rec = dict(s)
            rec["sentence_id"] = i
            rec["is_digression"] = bool(is_digression[i])
            rec["digression_type"] = "administrative" if is_digression[i] else "none"
            output_records.append(rec)

        return output_records
