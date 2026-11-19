"""src/segmentation/dual_head.py - Discriminative transition head & Generative title head."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DualHeadConfig:
    """Configuration for heterogeneous dual heads and contrastive projection."""

    adapter_dim: int = 1024          # H = 1024 contextual representation dimension
    transition_hidden: int = 512     # Hidden dimension of transition MLP
    contrastive_dim: int = 256       # Normalized contrastive hypersphere dimension
    llm_hidden_size: int = 2048      # Target LLM token embedding dimension
    dropout: float = 0.15
    eps: float = 1e-5
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "hf.co/unsloth/Qwen3.5-4b-GGUF:UD-Q3_K_XL"
    ollama_timeout: float = 10.0
    enable_ollama: bool = True


# ==============================================================================
# Transition Feature Extractor
# ==============================================================================

class TransitionFeatureExtractor:
    """Constructs 4D transition features across sentence boundaries:

    f_i^trans = [c_i || c_{i-1} || (c_i - c_{i-1}) || (c_i * c_{i-1})] in R^{4H}
    """

    @staticmethod
    def extract(context_states: torch.Tensor, pad_first: bool = False) -> torch.Tensor:
        """Extract 4D transition interaction vectors.

        Args:
            context_states: [B, N, H] or [N, H] contextual representations.
            pad_first: If True, prepends a dummy transition for index 0 to return shape [..., N, 4H].
                       If False, returns valid transitions of shape [..., N-1, 4H].

        Returns:
            Transition feature tensor of shape [..., N-1, 4H] or [..., N, 4H].
        """
        is_2d = context_states.dim() == 2
        if is_2d:
            context_states = context_states.unsqueeze(0)  # [1, N, H]

        B, N, H = context_states.shape
        if N < 2:
            empty = context_states.new_zeros((B, 0, 4 * H))
            return empty.squeeze(0) if is_2d else empty

        if pad_first:
            c_prev = torch.cat([context_states[:, :1, :], context_states[:, :-1, :]], dim=1)
            c_curr = context_states
        else:
            c_curr = context_states[:, 1:, :]   # c_i for i in 1..N-1
            c_prev = context_states[:, :-1, :]  # c_{i-1} for i in 1..N-1

        diff = c_curr - c_prev
        hadamard = c_curr * c_prev
        features = torch.cat([c_curr, c_prev, diff, hadamard], dim=-1)

        return features.squeeze(0) if is_2d else features


# ==============================================================================
# Head A: Discriminative Transition Head
# ==============================================================================

class DiscriminativeTransitionHead(nn.Module):
    """Head A: Discriminative boundary classifier MLP.

    Operates on the 4D transition feature vector:
        f_i^trans = [c_i || c_{i-1} || (c_i - c_{i-1}) || (c_i * c_{i-1})] in R^{4H}
    Projecting through LayerNorm, GELU, and Dropout to a scalar boundary logit z_i.
    """

    def __init__(
        self,
        config: Optional[DualHeadConfig] = None,
        hidden_dim: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.config = config or DualHeadConfig()
        h_dim = hidden_dim or self.config.adapter_dim
        inter_dim = intermediate_dim or self.config.transition_hidden
        drop = dropout if dropout is not None else self.config.dropout

        self.hidden_dim = h_dim
        self.intermediate_dim = inter_dim
        self.feature_dim = 4 * h_dim

        self.input_norm = nn.LayerNorm(self.feature_dim, eps=self.config.eps)
        self.dropout1 = nn.Dropout(drop)
        self.fc1 = nn.Linear(self.feature_dim, inter_dim)
        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(drop)
        self.fc2 = nn.Linear(inter_dim, max(inter_dim // 2, 64))
        self.act2 = nn.GELU()
        self.dropout3 = nn.Dropout(drop)
        self.fc_out = nn.Linear(max(inter_dim // 2, 64), 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.fc1, self.fc2]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.fc_out, nn.Linear):
            nn.init.xavier_uniform_(self.fc_out.weight)
            if self.fc_out.bias is not None:
                # Prior probability initialization: bias ~ -2.5 yields sigmoid(-2.5) ~ 0.076
                # preventing the model from defaulting to predicting 0.50 probability on rare boundaries
                nn.init.constant_(self.fc_out.bias, -2.5)

    def construct_transition_features(
        self,
        context_states: torch.Tensor,
        pad_first: bool = False,
    ) -> torch.Tensor:
        """Helper to construct 4D transition feature vectors."""
        return TransitionFeatureExtractor.extract(context_states, pad_first=pad_first)

    def forward(
        self,
        context_states: torch.Tensor,
        pad_first: bool = False,
    ) -> torch.Tensor:
        """Compute boundary transition logits z_i.

        Args:
            context_states: Context representations of shape [B, N, H] or [N, H].
            pad_first: If True, returns logits of shape [..., N] with logit 0 = -1e4.
                       If False, returns logits of shape [..., N-1].

        Returns:
            Raw transition logits z_i.
        """
        is_2d = context_states.dim() == 2
        if is_2d:
            context_states = context_states.unsqueeze(0)

        B, N, H = context_states.shape
        if N < 2:
            out = context_states.new_full((B, N), -1e4) if pad_first else context_states.new_zeros((B, 0))
            return out.squeeze(0) if is_2d else out

        features = self.construct_transition_features(context_states, pad_first=pad_first)
        h = self.input_norm(features)
        h = self.dropout1(h)
        h = self.act1(self.fc1(h))
        h = self.dropout2(h)
        h = self.act2(self.fc2(h))
        h = self.dropout3(h)
        logits = self.fc_out(h).squeeze(-1)  # [B, N-1] or [B, N]

        if pad_first and N >= 1:
            logits[:, 0] = -1e4

        return logits.squeeze(0) if is_2d else logits

    def predict_probabilities(
        self,
        context_states: torch.Tensor,
        pad_first: bool = False,
    ) -> torch.Tensor:
        """Compute calibrated boundary probabilities p_i = sigmoid(z_i)."""
        logits = self.forward(context_states, pad_first=pad_first)
        return torch.sigmoid(logits)


# Alias for backward compatibility
TransitionMLP = DiscriminativeTransitionHead


# ==============================================================================
# Contrastive Projection Head
# ==============================================================================

class ContrastiveProjectionHead(nn.Module):
    """Projects contextual sentence states into L2-normalized hypersphere for InfoNCE."""

    def __init__(self, config: Optional[DualHeadConfig] = None):
        super().__init__()
        self.config = config or DualHeadConfig()
        self.proj = nn.Sequential(
            nn.LayerNorm(self.config.adapter_dim, eps=self.config.eps),
            nn.Linear(self.config.adapter_dim, self.config.contrastive_dim),
        )

    def forward(self, contextual_states: torch.Tensor) -> torch.Tensor:
        """Project and normalize representations.

        Args:
            contextual_states: [..., H] sentence vectors.

        Returns:
            normalized_reps: [..., contrastive_dim] with unit L2 norm.
        """
        projected = self.proj(contextual_states)
        return F.normalize(projected.float(), p=2, dim=-1).to(contextual_states.dtype)


# ==============================================================================
# Head B: Generative Pedagogical Title Head
# ==============================================================================

@dataclass
class TopicMetadata:
    """Structured pedagogical metadata conforming to Interface Contract 2."""

    topic_title: str
    prerequisite_tags: List[str]
    cue_questions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_title": self.topic_title,
            "prerequisite_tags": self.prerequisite_tags,
            "cue_questions": self.cue_questions,
        }


class GenerativeTitleHead(nn.Module):
    """Head B: Generative head emitting structured pedagogical titles, tags, and Cornell cue questions.

    Features:
    - Primary: HTTP interface to local Ollama (hf.co/unsloth/Qwen3.5-4b-GGUF:UD-Q3_K_XL) using format='json'
      and keep_alive='0' to release GPU memory immediately.
    - Fallback: Deterministic offline pedagogical synthesizer guaranteeing zero-crash operation.
    - Training: Token-normalized causal language modeling cross-entropy loss computation.
    """

    def __init__(
        self,
        config: Optional[DualHeadConfig] = None,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        enable_ollama: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.config = config or DualHeadConfig()
        self.ollama_url = ollama_url or self.config.ollama_url
        self.model_name = model_name or self.config.ollama_model
        self.timeout_seconds = timeout_seconds or self.config.ollama_timeout
        self.enable_ollama = enable_ollama if enable_ollama is not None else self.config.enable_ollama

        # Projection layer mapping segment representation to LLM token embedding space
        self.proj = nn.Sequential(
            nn.LayerNorm(self.config.adapter_dim, eps=self.config.eps),
            nn.Linear(self.config.adapter_dim, self.config.llm_hidden_size),
        )

    def forward(
        self,
        lm_logits: Optional[torch.Tensor] = None,
        lm_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute token-normalized causal language modeling loss during training.

        L_{gen} = - 1 / (sum |T_k|) sum log P_LM(w_t | w_{<t})
        """
        if lm_logits is None or lm_labels is None or lm_labels.numel() == 0:
            return torch.tensor(0.0, device=lm_logits.device if lm_logits is not None else None)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()
        V = shift_logits.size(-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",  # Token-normalized
        )
        return loss

    def project_segment_context(self, segment_representations: torch.Tensor) -> torch.Tensor:
        """Project segment pooled representation to LLM token embedding space."""
        return self.proj(segment_representations).unsqueeze(-2)

    def generate_metadata(
        self,
        segment_text: str,
        is_digression: bool = False,
    ) -> TopicMetadata:
        """Synthesize structured pedagogical metadata for a topic segment.

        Tries local Ollama first; automatically falls back to deterministic heuristic if unavailable.
        """
        clean_text = " ".join(segment_text.split()).strip()
        if not clean_text:
            return self._offline_fallback("Introductory Lecture Concepts", is_digression=is_digression)

        if self.enable_ollama and not is_digression:
            try:
                result = self._query_ollama(clean_text)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug(f"Ollama generation fallback triggered: {e}")

        return self._offline_fallback(clean_text, is_digression=is_digression)

    def _query_ollama(self, segment_text: str) -> Optional[TopicMetadata]:
        """Query local Ollama server with format='json' to bypass thinking token prefixes."""
        snippet = segment_text[:1500]
        prompt = (
            f"Segment text: {snippet}\n"
            "Provide educational metadata in JSON with fields:\n"
            "- topic_title: descriptive pedagogical topic title (>= 10 chars)\n"
            "- prerequisite_tags: list of 2-4 foundational concept tags\n"
            "- cue_questions: list of 1-3 study questions, each ending in '?'\n"
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 300,
            },
            "keep_alive": "0",  # Release GPU VRAM immediately
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_response = body.get("response", "")
            parsed = json.loads(raw_response)

            title = str(parsed.get("topic_title", "")).strip()
            prereqs = [str(p).strip() for p in parsed.get("prerequisite_tags", []) if str(p).strip()]
            cues = [str(q).strip() for q in parsed.get("cue_questions", []) if str(q).strip()]

            # Validate constraints
            if len(title) >= 10 and len(prereqs) >= 1 and len(cues) >= 1:
                formatted_cues = [q if q.endswith("?") else f"{q}?" for q in cues]
                return TopicMetadata(
                    topic_title=title,
                    prerequisite_tags=prereqs,
                    cue_questions=formatted_cues,
                )
        return None

    def _offline_fallback(
        self,
        segment_text: str,
        is_digression: bool = False,
    ) -> TopicMetadata:
        """Deterministic offline rule-based pedagogical metadata generator."""
        if is_digression:
            return TopicMetadata(
                topic_title="Administrative Announcements: Course Logistics & Schedule",
                prerequisite_tags=[],
                cue_questions=[
                    "What assignment deadlines and logistical guidelines were announced?",
                    "What resources or preparation are required for upcoming course milestones?",
                ],
            )

        clean_text = " ".join(segment_text.split())
        words = re.findall(r"[a-zA-Z0-9_\-\+]+", clean_text)
        stopwords = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
            "is", "are", "was", "were", "this", "that", "with", "by", "as", "we",
            "can", "from", "it", "its", "be", "so", "if", "not", "but", "have",
            "has", "had", "will", "would", "should", "our", "you", "your", "they",
            "their", "then", "into", "also", "here", "there", "what", "how", "when",
        }
        content_words = [w for w in words if w.lower() not in stopwords and len(w) > 2]

        # 1. Descriptive Title (>= 10 chars)
        if content_words:
            top_words = content_words[:4]
            title_core = " ".join(w.capitalize() for w in top_words)
            title = f"Overview of {title_core}"
        else:
            title = "Topic Overview and Core Concepts"
        if len(title) < 10:
            title = f"{title} - Detailed Analysis"

        # 2. Prerequisite Tags (>= 1)
        text_lower = clean_text.lower()
        STEM_PREREQ_MAP = [
            ("process", "Process Concepts & Execution Model"),
            ("thread", "Concurrency & Thread Management"),
            ("cpu", "Computer Architecture & Registers"),
            ("memory", "Memory Hierarchy & Virtual Memory"),
            ("matrix", "Linear Algebra & Matrix Operations"),
            ("vector", "Vector Spaces & Inner Products"),
            ("probability", "Probability Theory & Random Variables"),
            ("gradient", "Multivariate Calculus & Optimization"),
            ("network", "Distributed Systems & Networking"),
            ("graph", "Data Structures & Graph Algorithms"),
            ("cache", "Computer Architecture & Caching"),
            ("compiler", "Formal Languages & Compilers"),
            ("paging", "Paging & Memory Management"),
        ]
        prerequisites = [tag for key, tag in STEM_PREREQ_MAP if key in text_lower]
        if not prerequisites:
            if len(content_words) >= 2:
                prerequisites = [
                    f"Foundations of {content_words[0].capitalize()}",
                    f"Principles of {content_words[1].capitalize()}",
                ]
            else:
                prerequisites = ["Foundational Mathematics", "Core Subject Principles"]

        # 3. Cornell Cue Questions (each ending strictly with '?')
        lead_concept = content_words[0].capitalize() if content_words else "the primary topic"
        cue_questions = [
            f"What is the fundamental theoretical objective of {lead_concept}?",
            f"How do the mechanisms discussed affect system performance and trade-offs?",
        ]

        return TopicMetadata(
            topic_title=title,
            prerequisite_tags=prerequisites,
            cue_questions=cue_questions,
        )


# Alias for backward compatibility
GenerativeTopicHead = GenerativeTitleHead
