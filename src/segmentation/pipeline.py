"""src/segmentation/pipeline.py - Unified Phase 2 Discourse-Aware Topic Segmentation Pipeline.

Orchestrates:
1. Ingestion and invariant validation of Interface Contract 1 sentence records.
2. Boundary classification via neural model or deterministic semantic cohesion fallback.
3. 1D Non-Maximum Suppression (1D-NMS) with radius r=2 sentences.
4. Contiguous segment partitioning and timestamp formatting (HH:MM:SS).
5. Pedagogical topic titling and Cornell cue question synthesis via local Ollama
   (hf.co/unsloth/Qwen3.5-4b-GGUF:UD-Q3_K_XL) with deterministic heuristic fallback.
6. Emission of validated Interface Contract 2 payloads.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/Qwen3.5-4b-GGUF:UD-Q3_K_XL"
DEFAULT_NMS_RADIUS = 2
DEFAULT_NMS_THRESHOLD = 0.50
DEFAULT_CHECKPOINT_PATH = "checkpoints/segmentation_qwen3_embed_multimodal/best_model.pt"
DEFAULT_BACKBONE_TYPE = "qwen3_embed"
DEFAULT_BACKBONE_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


# ==============================================================================
# Interface Contract 2 Dataclasses
# ==============================================================================

@dataclass
class TopicSegment:
    """Strongly typed representation of a topic segment in Interface Contract 2."""

    segment_id: int
    start_sentence_idx: int
    end_sentence_idx: int
    start_time_str: str  # "HH:MM:SS"
    end_time_str: str    # "HH:MM:SS"
    topic_title: str
    prerequisite_tags: List[str]
    cue_questions: List[str]
    sentences: List[Dict[str, Any]]
    raw_text: str

    def to_contract_dict(self) -> Dict[str, Any]:
        """Serialize to exact Contract 2 dictionary schema."""
        return {
            "segment_id": int(self.segment_id),
            "start_sentence_idx": int(self.start_sentence_idx),
            "end_sentence_idx": int(self.end_sentence_idx),
            "start_time_str": str(self.start_time_str),
            "end_time_str": str(self.end_time_str),
            "topic_title": str(self.topic_title),
            "prerequisite_tags": list(self.prerequisite_tags),
            "cue_questions": list(self.cue_questions),
            "sentences": list(self.sentences),
            "raw_text": str(self.raw_text),
        }


@dataclass
class MacroTopic:
    """Strongly typed representation of a Macro Lecture Module containing micro-segments."""

    module_id: int
    module_title: str
    start_time_str: str  # "HH:MM:SS"
    end_time_str: str    # "HH:MM:SS"
    pedagogical_focus: str
    segment_ids: List[int]

    def to_contract_dict(self) -> Dict[str, Any]:
        """Serialize to Contract 2 dictionary schema."""
        return {
            "module_id": int(self.module_id),
            "module_title": str(self.module_title),
            "start_time_str": str(self.start_time_str),
            "end_time_str": str(self.end_time_str),
            "pedagogical_focus": str(self.pedagogical_focus),
            "segment_ids": [int(sid) for sid in self.segment_ids],
        }


@dataclass
class SegmentationPayload:
    """Strongly typed top-level container for Interface Contract 2."""

    num_sentences: int
    topic_segments: List[TopicSegment]
    macro_topics: List[MacroTopic] = field(default_factory=list)

    def to_contract_dict(self) -> Dict[str, Any]:
        """Serialize to exact Contract 2 top-level dictionary."""
        d: Dict[str, Any] = {
            "num_sentences": int(self.num_sentences),
            "topic_segments": [seg.to_contract_dict() for seg in self.topic_segments],
        }
        if self.macro_topics:
            d["num_modules"] = len(self.macro_topics)
            d["macro_topics"] = [m.to_contract_dict() for m in self.macro_topics]
        return d


# ==============================================================================
# Pipeline Configuration
# ==============================================================================

@dataclass
class SegmentationPipelineConfig:
    """Comprehensive configuration for Phase 2 Topic Segmentation Pipeline."""

    # Decoding Hyperparameters
    nms_radius: int = DEFAULT_NMS_RADIUS  # Radius r=6 sentences
    boundary_threshold: float = DEFAULT_NMS_THRESHOLD  # Probability threshold theta=0.24
    min_segment_length: int = 1  # Minimum sentences per segment
    enable_macro_grouping: bool = True  # Enable 2-tier hierarchical grouping

    # Model & Architecture Options
    use_neural_model: bool = True  # Attempt PyTorch model forward pass if available
    checkpoint_path: Optional[str] = DEFAULT_CHECKPOINT_PATH
    backbone_type: str = DEFAULT_BACKBONE_TYPE
    backbone_model_name: str = DEFAULT_BACKBONE_MODEL_NAME
    load_in_4bit: bool = True
    device: Literal["cpu", "cuda"] = "cuda"
    lazy_load: bool = True  # Defer neural checkpoint loading until segment_transcript() to prevent VRAM collision
    isolate_neural_process: bool = True  # Run neural inference in isolated subprocess to immediately free bitsandbytes VRAM
    d_model: int = 1024
    num_heads: int = 8
    dbca_layers: int = 2

    # Generative Pedagogical Titling (Ollama)
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_sec: float = 30.0
    ollama_keep_alive: str = "5m"  # Pin model in VRAM across segments; eliminates reload thrashing
    ollama_num_ctx: int = 8192  # Shared 8K context across Phase 2 & Phase 3 (zero KV reload, 100% GPU)
    ollama_num_gpu: int = 99  # Offload all layers to GPU VRAM
    force_fallback: bool = False  # Force deterministic offline heuristics (useful in CI)

    # Heuristic Fallback Options
    max_title_words: int = 10
    default_cue_count: int = 2


# ==============================================================================
# Ollama Client for Generative Titling
# ==============================================================================

class OllamaTitleGenerator:
    """Client for querying local Ollama instance for pedagogical topic titles and metadata."""

    def __init__(self, config: Optional[SegmentationPipelineConfig] = None):
        self.config = config or SegmentationPipelineConfig()

    def generate_metadata(
        self,
        segment_text: Union[str, Sequence[str]],
        is_digression: bool = False,
    ) -> Dict[str, Any]:
        """Generate pedagogical topic title, prerequisite tags, and Cornell cue questions.

        Returns:
            Dict with 'topic_title', 'prerequisite_tags', 'cue_questions'.
        """
        if isinstance(segment_text, (list, tuple)):
            segment_text = " ".join(str(s).strip() for s in segment_text if str(s).strip())
        elif not isinstance(segment_text, str):
            segment_text = str(segment_text)

        if self.config.force_fallback or is_digression:
            return self._heuristic_fallback(segment_text, is_digression)

        system_prompt = (
            "Synthesize an educational topic title and Cornell study notes metadata for a lecture excerpt.\n"
            "Respond ONLY with a JSON object adhering to this schema:\n"
            "{\n"
            '  "topic_title": "concise descriptive title (>= 10 chars)",\n'
            '  "prerequisite_tags": ["concept1", "concept2"],\n'
            '  "cue_questions": ["question1?", "question2?"]\n'
            "}"
        )
        prompt = f"Transcript: {segment_text[:1000]}"

        request_body = {
            "model": self.config.ollama_model,
            "system": system_prompt,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 160,
                "num_ctx": self.config.ollama_num_ctx,
                "num_gpu": self.config.ollama_num_gpu,
            },
            "keep_alive": self.config.ollama_keep_alive,
        }

        try:
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/generate",
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.ollama_timeout_sec) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    raw_text = resp_json.get("response", "").strip()
                    parsed = self._clean_and_parse_json(raw_text)
                    if parsed:
                        return parsed
        except Exception as e:
            logger.debug("Ollama query failed (%s); switching to deterministic fallback.", e)

        return self._heuristic_fallback(segment_text, is_digression)

    def _clean_and_parse_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response, extracting cleanly even if wrapped in markdown code blocks."""
        if not raw_text:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
            title = str(data.get("topic_title", "")).strip()
            prereqs = [str(t).strip() for t in data.get("prerequisite_tags", []) if str(t).strip()]
            cues = [str(q).strip() for q in data.get("cue_questions", []) if str(q).strip()]

            # Enforce that every cue question ends with '?'
            cues = [q if q.endswith("?") else f"{q}?" for q in cues]
            if not cues:
                cues = ["What core educational concepts are established in this section?"]

            if len(title) >= 10 and len(prereqs) >= 1:
                return {
                    "topic_title": title,
                    "prerequisite_tags": prereqs,
                    "cue_questions": cues,
                }
        except Exception:
            pass
        return None

    def _heuristic_fallback(
        self,
        segment_text: Union[str, Sequence[str]],
        is_digression: bool = False,
    ) -> Dict[str, Any]:
        """Deterministic pedagogical heuristic generator for offline / test environments."""
        if isinstance(segment_text, (list, tuple)):
            segment_text = " ".join(str(s).strip() for s in segment_text if str(s).strip())
        elif not isinstance(segment_text, str):
            segment_text = str(segment_text)

        if not segment_text.strip():
            return {
                "topic_title": "General Conceptual Introduction",
                "prerequisite_tags": ["Foundational Subject Concepts"],
                "cue_questions": ["What fundamental concepts are discussed in this section?"],
            }

        if is_digression:
            return {
                "topic_title": "Administrative Announcements and Course Logistics",
                "prerequisite_tags": [],
                "cue_questions": [
                    "What assignment deadlines and logistical guidelines were announced?",
                ],
            }

        # Deterministic extraction of primary conceptual noun phrases
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", segment_text) if s.strip()]
        first_sentence = sentences[0] if sentences else segment_text

        # Strip introductory conversational noise
        cleaned_first = re.sub(
            r"^(?:Returning to our core topic|Let us now examine|In this section we discuss|Today we cover)\s*,?\s*",
            "",
            first_sentence,
            flags=re.IGNORECASE,
        ).strip()

        words = cleaned_first.rstrip(".?!").split()
        title_words = words[: self.config.max_title_words]
        topic_title = " ".join(title_words).strip()
        if topic_title:
            topic_title = topic_title[0].upper() + topic_title[1:]
        else:
            topic_title = "Discourse Analysis and Conceptual Foundations"

        if len(topic_title) < 10:
            topic_title = f"{topic_title} - Detailed Analysis"

        # Heuristic prerequisite and cue question extraction
        key_terms = [w.strip(",.;:") for w in words if len(w) > 4 and w.isalpha()]
        prereqs = list(dict.fromkeys(key_terms[:2])) if key_terms else ["Core Subject Fundamentals"]

        primary_term = prereqs[0] if prereqs else "this topic"
        cue_questions = [
            f"What is the theoretical definition and functional mechanism of {primary_term}?",
            f"How does {primary_term} operate within the broader course curriculum?",
        ]

        return {
            "topic_title": topic_title,
            "prerequisite_tags": prereqs,
            "cue_questions": cue_questions,
        }

    def group_into_macro_topics(self, segments: List[TopicSegment]) -> List[MacroTopic]:
        """Group contiguous micro-segments into cohesive Macro Lecture Modules (Macro-Topics)."""
        if not segments:
            return []

        if len(segments) == 1:
            seg = segments[0]
            return [
                MacroTopic(
                    module_id=1,
                    module_title=seg.topic_title,
                    start_time_str=seg.start_time_str,
                    end_time_str=seg.end_time_str,
                    pedagogical_focus=f"Foundational concepts in {seg.topic_title}.",
                    segment_ids=[int(seg.segment_id)],
                )
            ]

        if self.config.force_fallback:
            return self._heuristic_macro_grouping(segments)

        roadmap_lines = [
            f"- Segment {s.segment_id} [{s.start_time_str} - {s.end_time_str}]: {s.topic_title}"
            for s in segments
        ]
        roadmap_str = "\n".join(roadmap_lines)

        prompt = (
            "You are an expert curriculum designer and university professor.\n"
            "Below is the sequential outline of discourse segments from an educational lecture.\n"
            "Group these contiguous segments into 4 to 7 cohesive, high-level pedagogical lecture modules (Macro-Topics).\n"
            "Distinct pedagogical paradigms (e.g., Course Logistics, Supervised Learning, Practical ML Engineering, Unsupervised Learning, Reinforcement Learning) should each form their own module.\n"
            "Every segment ID from 0 to N-1 must appear in exactly one module, in sequential order without gaps.\n\n"
            f"Lecture Outline:\n{roadmap_str}\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "modules": [\n'
            "    {\n"
            '      "module_id": 1,\n'
            '      "module_title": "Descriptive Academic Title for Module",\n'
            '      "pedagogical_focus": "1-2 sentence description of what this module teaches",\n'
            '      "segment_ids": [0, 1, 2]\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        request_body = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
                "num_ctx": self.config.ollama_num_ctx,
                "num_gpu": self.config.ollama_num_gpu,
            },
            "keep_alive": self.config.ollama_keep_alive,
        }

        try:
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/generate",
                data=json.dumps(request_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.ollama_timeout_sec) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    raw_text = resp_json.get("response", "").strip()
                    macro_topics = self._parse_macro_modules_json(raw_text, segments)
                    if macro_topics:
                        return macro_topics
        except Exception as e:
            logger.warning("Ollama macro-grouping query failed (%s); reverting to heuristic grouping.", e)

        return self._heuristic_macro_grouping(segments)

    def _parse_macro_modules_json(
        self, raw_text: str, segments: List[TopicSegment]
    ) -> Optional[List[MacroTopic]]:
        """Parse and strictly validate LLM macro-module grouping response."""
        if not raw_text:
            return None
        text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
            mod_list = data.get("modules", [])
            if not isinstance(mod_list, list) or len(mod_list) == 0:
                return None

            seg_map = {s.segment_id: s for s in segments}
            macro_topics: List[MacroTopic] = []
            covered_seg_ids = set()

            for idx, m in enumerate(mod_list, 1):
                title = str(m.get("module_title", f"Module {idx}")).strip()
                focus = str(m.get("pedagogical_focus", "")).strip()
                raw_sids = m.get("segment_ids", [])
                valid_sids = [int(sid) for sid in raw_sids if int(sid) in seg_map]

                if not valid_sids:
                    continue

                valid_sids = sorted(valid_sids)
                covered_seg_ids.update(valid_sids)

                start_str = seg_map[valid_sids[0]].start_time_str
                end_str = seg_map[valid_sids[-1]].end_time_str

                macro_topics.append(
                    MacroTopic(
                        module_id=idx,
                        module_title=title,
                        start_time_str=start_str,
                        end_time_str=end_str,
                        pedagogical_focus=focus or f"Thematic study module covering {title}.",
                        segment_ids=valid_sids,
                    )
                )

            # Ensure all segments are covered sequentially
            missing = [s.segment_id for s in segments if s.segment_id not in covered_seg_ids]
            if missing and macro_topics:
                macro_topics[-1].segment_ids.extend(missing)
                macro_topics[-1].segment_ids.sort()
                macro_topics[-1].end_time_str = seg_map[macro_topics[-1].segment_ids[-1]].end_time_str

            return macro_topics if macro_topics else None
        except Exception as e:
            logger.debug("Failed to parse macro modules JSON: %s", e)
            return None

    def _heuristic_macro_grouping(self, segments: List[TopicSegment]) -> List[MacroTopic]:
        """Contiguous pedagogical fallback grouping when LLM is unavailable."""
        N = len(segments)
        if N == 0:
            return []

        if N <= 3:
            chunk_size = 1
        elif N <= 10:
            chunk_size = 3
        elif N <= 20:
            chunk_size = 4
        else:
            chunk_size = 5

        macro_topics: List[MacroTopic] = []
        mod_idx = 1
        for i in range(0, N, chunk_size):
            chunk = segments[i : i + chunk_size]
            sids = [s.segment_id for s in chunk]
            start_str = chunk[0].start_time_str
            end_str = chunk[-1].end_time_str
            first_title = chunk[0].topic_title
            title = f"{first_title} & Thematic Overview" if len(chunk) > 1 else first_title

            macro_topics.append(
                MacroTopic(
                    module_id=mod_idx,
                    module_title=title,
                    start_time_str=start_str,
                    end_time_str=end_str,
                    pedagogical_focus=f"Thematic lecture module covering {', '.join(s.topic_title for s in chunk[:3])}.",
                    segment_ids=sids,
                )
            )
            mod_idx += 1

        return macro_topics

    def evict_model(self) -> bool:
        """Evict model from VRAM immediately via keep_alive: 0."""
        try:
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/generate",
                data=json.dumps({"model": self.config.ollama_model, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False


def _neural_inference_worker(
    config_dict: Dict[str, Any],
    sentences: List[Dict[str, Any]],
    q_out: Any,
    video_path: Optional[str] = None,
) -> None:
    """Isolated subprocess worker executing neural model inference without polluting parent CUDA state."""
    try:
        from pathlib import Path
        cfg = SegmentationPipelineConfig(**config_dict)
        cfg.isolate_neural_process = False  # Prevent recursion
        pipe = SegmentationPipeline(cfg)
        visual_embeddings = None
        visual_shifts = None
        if video_path and Path(video_path).exists():
            visual_embeddings, visual_shifts = pipe._extract_visual_features(
                Path(video_path), sentences
            )
        if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
            pipe._load_neural_checkpoint(Path(cfg.checkpoint_path))
        probs = pipe._forward_neural_model(
            sentences,
            visual_embeddings=visual_embeddings,
            visual_shifts=visual_shifts,
        )
        pipe.unload_neural_model()
        q_out.put({"success": True, "probabilities": probs})
    except Exception as e:
        q_out.put({"success": False, "error": str(e)})


# ==============================================================================
# High-Level Segmentation Pipeline
# ==============================================================================

class SegmentationPipeline:
    """High-Level Inference & Topic Segmentation Pipeline.

    Converts Interface Contract 1 sentence sequences into structured, contiguous
    topic segments conforming strictly to Interface Contract 2.
    """

    def __init__(
        self,
        config: Optional[SegmentationPipelineConfig] = None,
        model: Optional[Any] = None,
        title_generator: Optional[OllamaTitleGenerator] = None,
    ):
        self.config = config or SegmentationPipelineConfig()
        self.model = model
        self.extractor = None
        self.title_generator = title_generator or OllamaTitleGenerator(self.config)

        if not self.config.lazy_load and self.model is None and self.config.use_neural_model and not self.config.force_fallback and self.config.checkpoint_path:
            ckpt_path = Path(self.config.checkpoint_path)
            if ckpt_path.exists():
                try:
                    self._load_neural_checkpoint(ckpt_path)
                except Exception as e:
                    logger.warning("Could not load neural checkpoint from %s (%s); will use semantic fallback.", ckpt_path, e)
            else:
                logger.info("Neural checkpoint %s not found; will use semantic fallback.", ckpt_path)

    def _load_neural_checkpoint(self, checkpoint_path: Path) -> None:
        """Load trained DiscourseAwareSegmentationModel and Qwen sentence extractor.

        Defaults to the 3-stage multimodal Qwen3-Embedding-0.6B + DBCA checkpoint
        (including Stage 3 LoRA weights) while retaining backward compatibility
        with legacy Qwen3.5-0.8B checkpoints. Never leaves a half-initialized model
        on failure.
        """
        import torch
        from src.segmentation.model import DiscourseAwareSegmentationModel, DiscourseAwareSegmentationConfig
        from src.segmentation.qwen_extractor import (
            QWEN3_EMBED_MODEL,
            DEFAULT_QWEN_MODEL,
            Qwen3EmbeddingSentenceRepresentationExtractor,
            QwenSentenceRepresentationExtractor,
        )

        device = self.config.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.config.device = device

        # Reset before loading so failures never leave stale CPU weights behind
        self.model = None
        self.extractor = None

        logger.info("Loading trained segmentation checkpoint: %s on %s...", checkpoint_path, device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_cfg = ckpt.get("model_config")
        if isinstance(model_cfg, dict):
            cfg = DiscourseAwareSegmentationConfig(**model_cfg)
        elif isinstance(model_cfg, DiscourseAwareSegmentationConfig):
            cfg = model_cfg
        else:
            cfg = DiscourseAwareSegmentationConfig(
                hidden_size=self.config.d_model,
                adapter_dim=self.config.d_model,
                nms_radius=self.config.nms_radius,
                nms_threshold=self.config.boundary_threshold,
            )

        model = DiscourseAwareSegmentationModel(cfg)
        state_dict = ckpt["model_state_dict"]
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                "Loaded checkpoint with %d missing optional keys (e.g. %s).",
                len(incompatible.missing_keys),
                incompatible.missing_keys[:3],
            )
        if incompatible.unexpected_keys:
            logger.info(
                "Loaded checkpoint with %d unexpected keys (e.g. %s).",
                len(incompatible.unexpected_keys),
                incompatible.unexpected_keys[:3],
            )

        # Multimodal checkpoints train trimodal_fusion.acoustic_fusion while leaving standalone
        # acoustic_fusion untouched; sync them so audio/transcript-only inference uses trained weights.
        has_trimodal_weights = any(k.startswith("trimodal_fusion.") for k in state_dict.keys())
        if has_trimodal_weights and getattr(cfg, "use_vision", False):
            model.acoustic_fusion.load_state_dict(model.trimodal_fusion.acoustic_fusion.state_dict())
            logger.debug("Synced trained trimodal_fusion.acoustic_fusion weights to acoustic_fusion for unimodal fallback.")

        model.to(device)
        model.eval()

        # Auto-detect extractor backbone from checkpoint contents/metadata
        ckpt_meta_cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
        has_lora_sd = bool(ckpt.get("extractor_lora_state_dict"))
        path_lower = str(checkpoint_path).lower()

        if has_lora_sd or "qwen3_embed" in path_lower or "embedding" in str(self.config.backbone_model_name).lower():
            resolved_backbone_type = "qwen3_embed"
            resolved_model_name = QWEN3_EMBED_MODEL
        elif ckpt_meta_cfg.get("backbone_type") == "qwen_08b" or "qwen_hq" in path_lower:
            resolved_backbone_type = "qwen_08b"
            resolved_model_name = ckpt_meta_cfg.get("backbone_model_name", DEFAULT_QWEN_MODEL)
        else:
            resolved_backbone_type = self.config.backbone_type
            resolved_model_name = self.config.backbone_model_name

        if resolved_backbone_type == "qwen3_embed":
            extractor = Qwen3EmbeddingSentenceRepresentationExtractor(
                model_name_or_path=resolved_model_name,
                load_in_4bit=self.config.load_in_4bit,
                device=device,
            )
            lora_sd = ckpt.get("extractor_lora_state_dict")
            if lora_sd:
                extractor.apply_lora(r=16, alpha=32)
                lora_incompat = extractor.backbone.load_state_dict(lora_sd, strict=False)
                logger.info(
                    "Loaded %d Stage-3 Qwen3-Embedding LoRA tensors (unexpected=%d).",
                    len(lora_sd),
                    len(lora_incompat.unexpected_keys),
                )
            extractor.eval()
        elif resolved_backbone_type == "qwen_08b":
            extractor = QwenSentenceRepresentationExtractor(
                model_name_or_path=resolved_model_name,
                load_in_4bit=self.config.load_in_4bit,
                device=device,
            )
            if "sentence_extractor_state_dict" in ckpt:
                try:
                    extractor.load_state_dict(ckpt["sentence_extractor_state_dict"], strict=False)
                except Exception as e:
                    logger.debug("Extractor partial load note: %s", e)
            extractor.eval()
        else:
            extractor = None

        self.model = model
        self.extractor = extractor
        logger.info(
            "Neural topic segmentation model successfully loaded (backbone=%s, model=%s, multimodal=%s).",
            resolved_backbone_type,
            resolved_model_name,
            has_trimodal_weights,
        )

    def _extract_visual_features(
        self,
        video_path: Path,
        sentences: List[Dict[str, Any]],
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """Extract aligned slide visual embeddings [1, N, 1024] and shifts [1, N] from video.

        Executes keyframe extraction (CPU FFmpeg) followed by Qwen 3.5 Vision Tower encoding,
        then immediately unloads the vision encoder to keep peak VRAM safely within 6 GB.
        """
        import gc
        import tempfile
        import torch

        try:
            from src.vision.keyframe_extractor import LectureKeyframeExtractor
            from src.vision.slide_encoder import QwenSlideRepresentationEncoder

            sentence_timestamps = [
                (float(s["start_ms"]) / 1000.0, float(s["end_ms"]) / 1000.0)
                for s in sentences
            ]
            if not sentence_timestamps:
                return None, None

            logger.info("Extracting multimodal slide keyframes from video: %s", video_path)
            kf_extractor = LectureKeyframeExtractor()
            with tempfile.TemporaryDirectory(prefix="lecture_keyframes_") as tmp_dir:
                keyframes = kf_extractor.extract_keyframes(str(video_path), output_dir=tmp_dir)
                if not keyframes:
                    logger.info("No slide keyframes extracted from %s; proceeding with text-only DBCA.", video_path)
                    return None, None

                shifts = kf_extractor.compute_sentence_visual_shifts(sentence_timestamps, keyframes)
                device = self.config.device
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"

                slide_encoder = QwenSlideRepresentationEncoder(device=device)
                slide_embs = slide_encoder.encode_keyframes(keyframes, batch_size=4)
                aligned_embs = slide_encoder.align_slides_to_sentences(
                    sentence_timestamps, keyframes, slide_embs
                )

                # Immediately release vision tower VRAM before Qwen3-Embedding + DBCA loads
                del slide_encoder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            v_emb = aligned_embs.unsqueeze(0)  # [1, N, 1024] on CPU
            v_shifts = torch.tensor(shifts, dtype=torch.float32).unsqueeze(0)  # [1, N] on CPU
            logger.info(
                "Extracted %d canonical slide keyframes -> aligned visual features %s.",
                len(keyframes),
                tuple(v_emb.shape),
            )
            return v_emb, v_shifts
        except Exception as e:
            logger.warning("Multimodal visual extraction skipped (%s); using trained text+DBCA path.", e)
            return None, None

    def unload_neural_model(self) -> None:
        """Unload neural model and extractor from memory, clearing CUDA cache for downstream LLMs."""
        import gc
        import torch

        if self.extractor is not None:
            del self.extractor
            self.extractor = None
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded Phase 2 neural segmentation model and cleared CUDA cache.")

    def segment_transcript(
        self,
        sentences_input: Union[Sequence[Dict[str, Any]], Sequence[Any]],
        return_payload_object: bool = False,
        video_path: Optional[Union[str, Path]] = None,
    ) -> Union[Dict[str, Any], SegmentationPayload]:
        """Execute Phase 2 topic segmentation.

        Args:
            sentences_input: List of sentence dictionaries (Contract 1) or SentenceRecord objects.
            return_payload_object: If True, returns SegmentationPayload; else Contract 2 dict.
            video_path: Optional path to lecture video file for multimodal slide visual fusion.

        Returns:
            Structured topic segmentation payload conforming to Interface Contract 2.
        """
        # Step 1: Normalize & Validate Interface Contract 1 Ingestion
        sentences = self._normalize_contract1_input(sentences_input)
        N = len(sentences)

        if N == 0:
            payload = SegmentationPayload(num_sentences=0, topic_segments=[])
            return payload if return_payload_object else payload.to_contract_dict()

        if N == 1:
            # Single-sentence document: emits exactly 1 segment [0..0]
            seg = self._build_single_segment(0, 0, 0, sentences)
            macro = MacroTopic(
                module_id=1,
                module_title=seg.topic_title,
                start_time_str=seg.start_time_str,
                end_time_str=seg.end_time_str,
                pedagogical_focus=f"Foundational concepts in {seg.topic_title}.",
                segment_ids=[0],
            )
            payload = SegmentationPayload(num_sentences=1, topic_segments=[seg], macro_topics=[macro])
            return payload if return_payload_object else payload.to_contract_dict()

        # Ensure Ollama is evicted before neural model loads into VRAM
        if hasattr(self, "title_generator") and hasattr(self.title_generator, "evict_model"):
            self.title_generator.evict_model()

        # Step 2: Compute Boundary Transition Probabilities
        probabilities = self._predict_transition_probabilities(sentences, video_path=video_path)

        # Step 3: Apply 1D-NMS with Radius r and Threshold theta
        boundary_indices = self._apply_1d_nms(
            probabilities=probabilities,
            radius=self.config.nms_radius,
            threshold=self.config.boundary_threshold,
        )

        # Explicitly unload PyTorch neural segmentation model from VRAM before LLM titling
        # so PyTorch (1.4 GB) and Ollama (4.1 GB) NEVER overlap in GPU VRAM
        self.unload_neural_model()

        # Step 4: Partition Sentences into Contiguous Segments [0..N-1]
        segments = self._partition_into_segments(sentences, boundary_indices)

        # Step 5: 2-Tier Hierarchical Grouping (Macro-Topics)
        macro_topics: List[MacroTopic] = []
        if getattr(self.config, "enable_macro_grouping", True) and segments:
            macro_topics = self.title_generator.group_into_macro_topics(segments)

        payload = SegmentationPayload(
            num_sentences=N,
            topic_segments=segments,
            macro_topics=macro_topics,
        )
        return payload if return_payload_object else payload.to_contract_dict()

    def _normalize_contract1_input(
        self, sentences_input: Sequence[Any]
    ) -> List[Dict[str, Any]]:
        """Validate and normalize Contract 1 sentence records."""
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(sentences_input):
            if hasattr(item, "to_contract_dict"):
                d = item.to_contract_dict()
            elif isinstance(item, dict):
                d = dict(item)
            elif hasattr(item, "__dict__"):
                d = dict(vars(item))
            else:
                raise TypeError(f"Unsupported sentence item type: {type(item)}")

            # Invariant assertions
            assert "sentence_id" in d, f"Missing sentence_id at index {idx}"
            assert "text" in d, f"Missing text at index {idx}"
            assert "start_ms" in d and d["start_ms"] >= 0, f"Invalid start_ms at index {idx}"
            assert "end_ms" in d and d["end_ms"] >= d["start_ms"], f"Invalid end_ms at index {idx}"
            assert "pause_before_ms" in d and d["pause_before_ms"] >= 0.0, f"Invalid pause_before_ms at {idx}"

            # Standardize defaults
            d["pitch_reset"] = float(d.get("pitch_reset", 0.0))
            d["speaker_tag"] = str(d.get("speaker_tag", "Lecturer"))
            d["is_digression"] = bool(d.get("is_digression", False))
            d["digression_type"] = str(d.get("digression_type", "administrative" if d["is_digression"] else "none"))
            normalized.append(d)

        return normalized

    def _predict_transition_probabilities(
        self,
        sentences: List[Dict[str, Any]],
        video_path: Optional[Union[str, Path]] = None,
    ) -> List[float]:
        """Compute boundary transition probabilities for adjacent sentences (indices 0..N-2)."""
        N = len(sentences)
        video_path_str = str(video_path) if video_path is not None else None

        # Isolated Subprocess Neural Inference: liberates 100% of bitsandbytes/PyTorch VRAM upon completion
        if (
            self.config.use_neural_model
            and not self.config.force_fallback
            and self.config.checkpoint_path
            and Path(self.config.checkpoint_path).exists()
            and self.config.device == "cuda"
            and getattr(self.config, "isolate_neural_process", True)
        ):
            try:
                import multiprocessing as mp
                ctx = mp.get_context("spawn")
                q_out = ctx.Queue()
                cfg_dict = asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else dict(vars(self.config))
                p = ctx.Process(
                    target=_neural_inference_worker,
                    args=(cfg_dict, sentences, q_out, video_path_str),
                )
                p.start()
                res = q_out.get(timeout=300)
                p.join(timeout=15)
                if p.is_alive():
                    p.terminate()
                if res.get("success"):
                    logger.info("Isolated neural segmentation inference completed successfully (0 bytes VRAM leak).")
                    return res["probabilities"]
                else:
                    logger.warning("Isolated neural prediction error: %s; falling back to in-process execution.", res.get("error"))
            except Exception as e:
                logger.warning("Isolated neural subprocess failed (%s); reverting to semantic fallback.", e)

        # Optional multimodal visual feature extraction before loading Qwen3-Embedding + DBCA
        visual_embeddings = None
        visual_shifts = None
        if (
            self.config.use_neural_model
            and not self.config.force_fallback
            and video_path_str
            and Path(video_path_str).exists()
        ):
            visual_embeddings, visual_shifts = self._extract_visual_features(
                Path(video_path_str), sentences
            )

        # Lazy load neural model if deferred from __init__ (in-process fallback)
        if self.config.use_neural_model and self.model is None and not self.config.force_fallback and self.config.checkpoint_path:
            ckpt_path = Path(self.config.checkpoint_path)
            if ckpt_path.exists():
                try:
                    self._load_neural_checkpoint(ckpt_path)
                except Exception as e:
                    self.model = None
                    self.extractor = None
                    logger.warning("Could not load neural checkpoint from %s (%s); will use semantic fallback.", ckpt_path, e)

        # Check if PyTorch neural model is available
        if self.config.use_neural_model and self.model is not None:
            try:
                return self._forward_neural_model(
                    sentences,
                    visual_embeddings=visual_embeddings,
                    visual_shifts=visual_shifts,
                )
            except Exception as e:
                logger.warning("Neural forward pass failed (%s); reverting to semantic fallback.", e)

        # Deterministic Offline Semantic Fallback (Cosine Cohesion Depth Scoring)
        return self._compute_semantic_fallback_probabilities(sentences)

    def _compute_semantic_fallback_probabilities(
        self, sentences: List[Dict[str, Any]]
    ) -> List[float]:
        """Deterministic lexical semantic fallback computing boundary probabilities."""
        N = len(sentences)
        if N <= 1:
            return []

        # Simple term frequency vectorization
        vocab: Dict[str, int] = {}
        tf_vectors: List[Dict[str, int]] = []

        for s in sentences:
            words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", s["text"])]
            counts: Dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
                if w not in vocab:
                    vocab[w] = len(vocab)
            tf_vectors.append(counts)

        # Construct dense normalized vectors
        D = max(1, len(vocab))
        dense_embs = np.zeros((N, D), dtype=np.float32)
        for i, counts in enumerate(tf_vectors):
            for w, c in counts.items():
                dense_embs[i, vocab[w]] = float(c)
            norm = np.linalg.norm(dense_embs[i])
            if norm > 0:
                dense_embs[i] /= norm

        # Compute adjacent cosine similarities and pause weights
        probs: List[float] = []
        for i in range(N - 1):
            curr_emb = dense_embs[i]
            next_emb = dense_embs[i + 1]
            cos_sim = float(np.dot(curr_emb, next_emb))
            cos_dist = 1.0 - max(0.0, cos_sim)

            # Pause boost: pause_before_ms at i+1
            pause_ms = sentences[i + 1].get("pause_before_ms", 0.0)
            pause_factor = min(0.3, pause_ms / 5000.0)

            # Digression state change boost
            digress_shift = 0.4 if (sentences[i].get("is_digression") != sentences[i + 1].get("is_digression")) else 0.0

            prob = float(np.clip(0.35 * cos_dist + pause_factor + digress_shift, 0.01, 0.99))
            probs.append(prob)

        return probs

    def _forward_neural_model(
        self,
        sentences: List[Dict[str, Any]],
        visual_embeddings: Optional[Any] = None,
        visual_shifts: Optional[Any] = None,
    ) -> List[float]:
        """Forward pass through PyTorch DiscourseAwareSegmentationModel with Qwen3-Embedding/Qwen extractor."""
        import torch

        if self.model is None:
            return self._compute_semantic_fallback_probabilities(sentences)

        texts = [s["text"] for s in sentences]
        N = len(texts)
        device = self.config.device

        if self.extractor is not None:
            max_len = int(getattr(self.model.config, "max_sentences", 1024))
            with torch.no_grad():
                sentence_reps = self.extractor(texts)  # [1, N, D]
                if N <= max_len:
                    mask = torch.ones(1, N, device=device)
                    v_emb = visual_embeddings.to(device) if visual_embeddings is not None else None
                    v_sh = visual_shifts.to(device) if visual_shifts is not None else None
                    output = self.model(
                        sentence_reps=sentence_reps,
                        sentence_mask=mask,
                        visual_embeddings=v_emb,
                        visual_shifts=v_sh,
                    )
                    probs_tensor = output.boundary_probabilities[0].cpu()
                    p_list = probs_tensor.tolist()
                    return p_list[1:] if len(p_list) == N else p_list
                else:
                    # Sliding window inference for very long lectures exceeding DBCA positional limit
                    stride = max(1, max_len // 2)
                    accum = np.zeros(N, dtype=np.float64)
                    counts = np.zeros(N, dtype=np.float64)
                    for start in range(0, N, stride):
                        end = min(N, start + max_len)
                        sub_len = end - start
                        sub_reps = sentence_reps[:, start:end, :]
                        sub_mask = torch.ones(1, sub_len, device=device)
                        sub_v_emb = visual_embeddings[:, start:end, :].to(device) if visual_embeddings is not None else None
                        sub_v_sh = visual_shifts[:, start:end].to(device) if visual_shifts is not None else None
                        out = self.model(
                            sentence_reps=sub_reps,
                            sentence_mask=sub_mask,
                            visual_embeddings=sub_v_emb,
                            visual_shifts=sub_v_sh,
                        )
                        sub_probs = out.boundary_probabilities[0].cpu().numpy()
                        # Ignore position 0 of each window except the very first window
                        w_start = 0 if start == 0 else 1
                        accum[start + w_start : end] += sub_probs[w_start:]
                        counts[start + w_start : end] += 1.0
                        if end == N:
                            break
                    merged = np.divide(accum, np.maximum(counts, 1.0))
                    return merged[1:].tolist()

        if hasattr(self.model, "predict_boundary_probabilities"):
            H = getattr(self.model.config, "hidden_size", 1024)
            dummy_reps = torch.randn(1, N, H, device=device)
            probs = self.model.predict_boundary_probabilities(dummy_reps)
            if isinstance(probs, torch.Tensor):
                p_list = probs.squeeze(0).cpu().tolist()
                return p_list[1:] if len(p_list) == N else p_list

        return self._compute_semantic_fallback_probabilities(sentences)

    def _apply_1d_nms(
        self,
        probabilities: Sequence[float],
        radius: int = 2,
        threshold: float = 0.50,
    ) -> List[int]:
        """Apply 1D Non-Maximum Suppression with radius r=2.

        Note on index semantics:
        probabilities[i] for i in 0..N-2 is the boundary between sentence i and sentence i+1.
        Selecting candidate i means splitting after sentence i, so the next segment starts at i+1.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        M = len(probs)
        if M == 0:
            return []

        candidates = [i for i in range(M) if probs[i] >= threshold]
        if not candidates:
            return []

        # Sort descending by probability with deterministic tie-breaking (earlier index first)
        sorted_cands = sorted(candidates, key=lambda idx: (-probs[idx], idx))

        suppressed = set()
        selected = []

        for idx in sorted_cands:
            if idx in suppressed:
                continue
            selected.append(idx)
            for neighbor in range(max(0, idx - radius), min(M, idx + radius + 1)):
                if neighbor != idx:
                    suppressed.add(neighbor)

        # Transition candidate index i corresponds to sentence boundary split at sentence i+1
        selected_boundaries = sorted([idx + 1 for idx in selected])
        return selected_boundaries

    def _partition_into_segments(
        self,
        sentences: List[Dict[str, Any]],
        boundary_indices: List[int],
    ) -> List[TopicSegment]:
        """Partition sentences into contiguous topic segments [0..N-1]."""
        N = len(sentences)
        # Boundaries define split points: e.g. [5, 9] creates [0..4], [5..8], [9..N-1]
        valid_bounds = [b for b in boundary_indices if 0 < b < N]

        # Enforce min_segment_length when document has sufficient sentences
        min_len = max(1, self.config.min_segment_length)
        if min_len > 1 and len(valid_bounds) > 0 and N >= min_len * 2:
            filtered_bounds = []
            last_b = 0
            for b in valid_bounds:
                # Always preserve administrative digression transitions
                is_digress_edge = False
                if 0 < b < N:
                    is_digress_edge = (sentences[b - 1].get("is_digression") != sentences[b].get("is_digression"))
                if is_digress_edge or ((b - last_b) >= min_len and (N - b) >= min_len):
                    filtered_bounds.append(b)
                    last_b = b
            valid_bounds = filtered_bounds

        splits = [0] + valid_bounds + [N]

        segments: List[TopicSegment] = []
        for seg_idx in range(len(splits) - 1):
            start_idx = splits[seg_idx]
            end_idx = splits[seg_idx + 1] - 1

            seg = self._build_single_segment(seg_idx, start_idx, end_idx, sentences)
            segments.append(seg)

        return segments

    def _build_single_segment(
        self,
        segment_id: int,
        start_idx: int,
        end_idx: int,
        sentences: List[Dict[str, Any]],
    ) -> TopicSegment:
        """Construct a single TopicSegment and synthesize metadata."""
        seg_sentences = sentences[start_idx : end_idx + 1]
        raw_text = " ".join(s["text"] for s in seg_sentences).strip()

        # Timestamp strings "HH:MM:SS"
        start_ms = seg_sentences[0]["start_ms"]
        end_ms = seg_sentences[-1]["end_ms"]
        start_time_str = self._format_timestamp_str(start_ms)
        end_time_str = self._format_timestamp_str(end_ms)

        # Check if this segment represents an administrative digression
        is_digression = all(s.get("is_digression", False) for s in seg_sentences)

        # Synthesize pedagogical topic title, prerequisite tags, and cue questions
        metadata = self.title_generator.generate_metadata(raw_text, is_digression=is_digression)

        return TopicSegment(
            segment_id=segment_id,
            start_sentence_idx=start_idx,
            end_sentence_idx=end_idx,
            start_time_str=start_time_str,
            end_time_str=end_time_str,
            topic_title=metadata["topic_title"],
            prerequisite_tags=metadata["prerequisite_tags"],
            cue_questions=metadata["cue_questions"],
            sentences=seg_sentences,
            raw_text=raw_text,
        )

    @staticmethod
    def _format_timestamp_str(time_ms: int) -> str:
        """Format millisecond timestamp into 'HH:MM:SS'."""
        total_sec = max(0, int(round(time_ms / 1000.0)))
        hours = total_sec // 3600
        mins = (total_sec % 3600) // 60
        secs = total_sec % 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
