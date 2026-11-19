"""src/pipeline/run_e2e_lecture.py - Unified End-to-End Educational Lecture Pipeline.

Connects:
- Phase 1: Acoustic Grounding, Disfluency Scrubbing & Digression Gating (`src/asr/pipeline.py`)
- Phase 2: Discourse-Aware Topic Segmentation (`src/segmentation/pipeline.py`)
- Phase 3: Pedagogical Cornell Study Note Generation (`src/pipeline/note_generator.py`)

Adheres strictly to RTX 4050 (6GB VRAM) and 16GB System RAM hardware constraints:
- Sequential memory staging (CPU/INT8 ASR by default, 0 VRAM; DBCA + Ollama in Phase 2).
- Zero-thrashing eviction (`keep_alive: 0`) with explicit model unload upon stage completion.
- Polymorphic input ingestion: raw 16 kHz WAV, pre-computed JSON, plain text TXT.
- Dual output generation: publication-grade Markdown and draft-07 compliant JSON.
- Exports all 5 artifacts:
    1. phase1_transcript.json
    2. phase2_segmentation.json
    3. lecture_study_notes.md
    4. lecture_study_notes.json
    5. execution_summary.json
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import re
import resource
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

local_bin = str(Path.home() / ".local" / "bin")
if local_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"

# Phase 1 & 2 Modules
from src.asr.pipeline import ASRPipeline, ASRPipelineConfig, SentenceRecord
from src.segmentation.pipeline import (
    SegmentationPayload,
    SegmentationPipeline,
    SegmentationPipelineConfig,
    TopicSegment,
)
from src.pipeline.note_generator import (
    NoteGeneratorConfig,
    PedagogicalNoteGenerator,
    LectureStudyNotes,
)
from src.pipeline.note_translator import (
    NoteTranslator,
    NoteTranslatorConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/Qwen3.5-4b-GGUF:UD-Q3_K_XL"


# ==============================================================================
# 1. Hardware Memory Manager
# ==============================================================================

class HardwareMemoryManager:
    """Manages VRAM and System RAM allocations, stage isolation, and Ollama eviction."""

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def get_memory_stats(self) -> Dict[str, Any]:
        """Capture current process RSS and GPU VRAM footprint."""
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        stats: Dict[str, Any] = {
            "peak_rss_mb": round(rusage.ru_maxrss / 1024.0, 2),
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
        }
        try:
            import torch
            if torch.cuda.is_available():
                stats["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 2)
                stats["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024.0 * 1024.0), 2)
        except Exception:
            pass
        return stats

    def cleanup(self) -> None:
        """Sequential stage isolation: collect garbage and release PyTorch CUDA caches."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def evict_ollama(self) -> bool:
        """Unload Ollama model from GPU VRAM using zero-thrashing keep_alive: 0."""
        req_data = json.dumps({
            "model": self.ollama_model,
            "keep_alive": 0,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug("Ollama eviction call skipped or server not reachable: %s", e)
            return False

    @contextlib.contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager tracking stage latency and memory deltas."""
        start_time = time.perf_counter()
        initial_stats = self.get_memory_stats()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            final_stats = self.get_memory_stats()
            logger.info(
                "Stage [%s] finished in %.3fs (Peak RSS: %.1f MB, VRAM: %.1f MB)",
                stage_name,
                elapsed,
                final_stats["peak_rss_mb"],
                final_stats["vram_allocated_mb"],
            )


# ==============================================================================
# 2. Transcript Loader (Polymorphic Ingestion)
# ==============================================================================

class TranscriptLoader:
    """Loads and normalizes transcripts from JSON files, TXT files, or in-memory fixtures."""

    @classmethod
    def load(cls, input_source: Union[str, Path, Sequence[Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize any input source into Interface Contract 1 sentence dictionaries."""
        if isinstance(input_source, (str, Path)):
            path = Path(input_source)
            if not path.exists():
                raise FileNotFoundError(f"Transcript source not found: {path.resolve()}")
            if path.suffix.lower() == ".json":
                return cls.load_from_json(path)
            elif path.suffix.lower() in (".txt", ".text"):
                return cls.load_from_text(path)
            else:
                # Default attempt JSON then text
                try:
                    return cls.load_from_json(path)
                except Exception:
                    return cls.load_from_text(path)

        if isinstance(input_source, dict):
            return cls.load_from_json(input_source)

        if isinstance(input_source, (list, tuple)):
            normalized = []
            for idx, item in enumerate(input_source):
                normalized.append(cls._normalize_sentence_dict(item, idx))
            return normalized

        raise TypeError(f"Unsupported transcript input source type: {type(input_source)}")

    @classmethod
    def load_from_json(cls, path_or_dict: Union[str, Path, Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
        """Load Contract 1 sentences from JSON file path or dictionary."""
        if isinstance(path_or_dict, (str, Path)):
            with open(path_or_dict, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = path_or_dict

        if isinstance(data, list):
            raw_sents = data
        elif isinstance(data, dict):
            raw_sents = data.get("sentences", data.get("transcript", [data]))
        else:
            raise ValueError(f"Unexpected JSON data format: {type(data)}")

        normalized = []
        for idx, s in enumerate(raw_sents):
            normalized.append(cls._normalize_sentence_dict(s, idx))
        return normalized

    @classmethod
    def load_from_text(cls, path_or_str: Union[str, Path, str]) -> List[Dict[str, Any]]:
        """Convert raw plain text into pedagogical Contract 1 sentence records."""
        if isinstance(path_or_str, (str, Path)) and Path(path_or_str).is_file():
            text = Path(path_or_str).read_text(encoding="utf-8")
        else:
            text = str(path_or_str)

        # Split sentences preserving punctuation
        sentences_raw = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences_raw = [s.strip() for s in sentences_raw if s.strip()]

        normalized = []
        cur_ms = 1000
        words_per_sec = 2.5  # ~150 words per minute typical lecture rate
        for idx, sent_text in enumerate(sentences_raw):
            words_count = max(1, len(sent_text.split()))
            dur_ms = int(round((words_count / words_per_sec) * 1000.0))
            pause_ms = 500.0 if idx > 0 else 0.0

            normalized.append({
                "sentence_id": idx,
                "text": sent_text,
                "start_ms": cur_ms,
                "end_ms": cur_ms + dur_ms,
                "pause_before_ms": pause_ms,
                "pitch_reset": 0.0,
                "speaker_tag": "Lecturer",
                "is_digression": False,
                "digression_type": "none",
            })
            cur_ms += dur_ms + int(pause_ms)

        return normalized

    @classmethod
    def _normalize_sentence_dict(cls, item: Any, default_id: int) -> Dict[str, Any]:
        """Ensure dictionary strictly satisfies Interface Contract 1 schema."""
        if hasattr(item, "to_contract_dict"):
            return item.to_contract_dict()
        if hasattr(item, "__dict__"):
            d = dict(item.__dict__)
        elif isinstance(item, dict):
            d = dict(item)
        else:
            d = {"text": str(item)}

        s_id = int(d.get("sentence_id", default_id))
        text = str(d.get("text", "")).strip()
        start_ms = int(d.get("start_ms", default_id * 3000))
        end_ms = int(d.get("end_ms", start_ms + max(1000, len(text.split()) * 300)))
        pause_before_ms = float(d.get("pause_before_ms", 0.0))
        pitch_reset = float(d.get("pitch_reset", 0.0))
        speaker_tag = str(d.get("speaker_tag", "Lecturer"))
        is_dig = bool(d.get("is_digression", False))
        dig_type = str(d.get("digression_type", "administrative" if is_dig else "none"))

        if is_dig and dig_type == "none":
            dig_type = "administrative"
        elif not is_dig:
            dig_type = "none"

        return {
            "sentence_id": s_id,
            "text": text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "pause_before_ms": pause_before_ms,
            "pitch_reset": pitch_reset,
            "speaker_tag": speaker_tag,
            "is_digression": is_dig,
            "digression_type": dig_type,
        }


# ==============================================================================
# 3. Pipeline Configuration & Result Model
# ==============================================================================

@dataclass
class E2EPipelineConfig:
    """Unified configuration for the End-to-End Educational Lecture Pipeline."""

    # Hardware & Stage Isolation
    device: Literal["cpu", "cuda"] = "cpu"
    whisper_compute_type: str = "int8_float16"
    whisper_model_size: str = "large-v3-turbo"
    whisper_beam_size: int = 1
    whisper_batch_size: int = 8

    # Acoustic VAD Parameters
    vad_min_chunk_duration: float = 10.0
    vad_max_chunk_duration: float = 28.0

    # Discourse Segmentation Parameters
    # Calibrated defaults: 0.24 / r=6 for Qwen3-Embedding + DBCA multimodal champion (51.37% Relaxed F1);
    # automatically switches to 0.50 / r=2 when force_fallback=True unless explicitly overridden.
    boundary_threshold: Optional[float] = None
    nms_radius: Optional[int] = None
    min_segment_length: int = 1
    segmentation_checkpoint: Optional[str] = "checkpoints/segmentation_qwen3_embed_multimodal/best_model.pt"

    # LLM & Titling Parameters
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_keep_alive: str = "5m"
    ollama_num_ctx: int = 8192
    ollama_num_gpu: int = 99
    ollama_num_predict: int = 4096
    ollama_reduce_num_predict: int = 4096

    # Determinism & Fallback
    force_fallback: bool = False
    offline: bool = False

    # Output & Persistence
    notes_markdown_filename: str = "lecture_study_notes.md"
    notes_json_filename: str = "lecture_study_notes.json"
    transcript_filename: str = "phase1_transcript.json"
    segmentation_filename: str = "phase2_segmentation.json"
    execution_summary_filename: str = "execution_summary.json"
    output_dir: Optional[Union[str, Path]] = None
    save_intermediate: bool = True

    # Phase 4: Hindi Study Notes Translation
    translate_hindi: bool = False
    translation_backend: str = "ollama"  # "ollama" (merged Q4_K_M GGUF) with auto-fallback to "pytorch"
    translation_ollama_model: str = "sarvam1-hi-step200"
    translation_adapter_path: Optional[str] = "models/sarvam1_subtopic_script_constrained/best_checkpoint"
    translation_base_model_id: str = "sarvamai/sarvam-1"
    translation_device: str = "cuda"
    translation_load_in_4bit: bool = True
    notes_hindi_markdown_filename: str = "lecture_study_notes_hi.md"
    notes_hindi_json_filename: str = "lecture_study_notes_hi.json"

    # Lecture Metadata Defaults
    course: str = "Computer Science"
    title: str = "Educational Lecture Study Notes"

    def __post_init__(self):
        if self.offline:
            self.force_fallback = True
        if self.device == "cpu" and self.whisper_compute_type in ("float16", "int8_float16"):
            self.whisper_compute_type = "int8"
        if self.boundary_threshold is None:
            self.boundary_threshold = 0.50 if self.force_fallback else 0.24
        if self.nms_radius is None:
            self.nms_radius = 2 if self.force_fallback else 6
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class PipelineResult:
    """Execution output container supporting both attribute and dict-key access."""

    status: str
    num_sentences: int
    num_segments: int
    phase1_transcript: List[Dict[str, Any]]
    phase2_segmentation: Dict[str, Any]
    study_notes_markdown: str
    study_notes_json: Dict[str, Any]
    execution_summary: Dict[str, Any]
    study_notes_hindi_markdown: Optional[str] = None
    study_notes_hindi_json: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Path] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return asdict(self).keys()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 4. End-to-End Lecture Pipeline Orchestrator
# ==============================================================================

class E2ELecturePipeline:
    """Unified Orchestrator executing Phase 1 -> Phase 2 -> Pedagogical Study Notes."""

    def __init__(self, config: Optional[E2EPipelineConfig] = None):
        self.config = config or E2EPipelineConfig()
        if self.config.offline:
            self.config.force_fallback = True

        self.memory_manager = HardwareMemoryManager(
            ollama_url=self.config.ollama_url,
            ollama_model=self.config.ollama_model,
        )

        # Initialize Phase 1 Pipeline
        self.asr_pipeline = ASRPipeline(
            ASRPipelineConfig(
                device=self.config.device,
                whisper_compute_type=self.config.whisper_compute_type,
                whisper_model_size=self.config.whisper_model_size,
                whisper_beam_size=self.config.whisper_beam_size,
                whisper_batch_size=self.config.whisper_batch_size,
                vad_min_chunk_duration=self.config.vad_min_chunk_duration,
                vad_max_chunk_duration=self.config.vad_max_chunk_duration,
                force_fallback=self.config.force_fallback,
            )
        )

        # Initialize Phase 2 Pipeline
        self.segmentation_pipeline = SegmentationPipeline(
            SegmentationPipelineConfig(
                device=self.config.device,
                boundary_threshold=self.config.boundary_threshold,
                nms_radius=self.config.nms_radius,
                min_segment_length=self.config.min_segment_length,
                checkpoint_path=self.config.segmentation_checkpoint,
                ollama_url=self.config.ollama_url,
                ollama_model=self.config.ollama_model,
                ollama_keep_alive=self.config.ollama_keep_alive,
                ollama_num_ctx=self.config.ollama_num_ctx,
                ollama_num_gpu=self.config.ollama_num_gpu,
                force_fallback=self.config.force_fallback,
            )
        )

        # Initialize Phase 3 Note Generator
        self.note_generator = PedagogicalNoteGenerator(
            NoteGeneratorConfig(
                use_ollama=not self.config.force_fallback,
                ollama_url=self.config.ollama_url,
                ollama_model=self.config.ollama_model,
                ollama_keep_alive=self.config.ollama_keep_alive,
                ollama_num_ctx=self.config.ollama_num_ctx,
                ollama_num_gpu=self.config.ollama_num_gpu,
                ollama_num_predict=self.config.ollama_num_predict,
                ollama_reduce_num_predict=self.config.ollama_reduce_num_predict,
                force_fallback=self.config.force_fallback,
                default_course=self.config.course,
                default_lecture_title=self.config.title,
            )
        )

        # Initialize Phase 4 Note Translator
        self.note_translator = NoteTranslator(
            NoteTranslatorConfig(
                backend=self.config.translation_backend,
                ollama_url=self.config.ollama_url,
                ollama_model=self.config.translation_ollama_model,
                base_model_id=self.config.translation_base_model_id,
                adapter_path=self.config.translation_adapter_path,
                device=self.config.translation_device or self.config.device,
                load_in_4bit=self.config.translation_load_in_4bit,
                force_fallback=self.config.force_fallback,
                offline=self.config.offline,
            )
        )

        self.last_execution_stats: Dict[str, Any] = {}

    def _notify_stage(
        self,
        stage_callback: Optional[Any],
        stage_id: str,
        event: str,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        """Invoke optional UI/caller stage callback without letting callback errors abort execution."""
        if stage_callback is None:
            return
        try:
            stage_callback(stage_id, event, elapsed_sec)
        except Exception as e:
            logger.debug("Stage callback (%s, %s) raised non-fatal error: %s", stage_id, event, e)

    def run(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        audio_array: Optional[np.ndarray] = None,
        transcript_input: Optional[Union[str, Path, Sequence[Any], Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        video_path: Optional[Union[str, Path]] = None,
        stage_callback: Optional[Any] = None,
    ) -> PipelineResult:
        """Execute end-to-end processing pipeline from audio or transcript source.

        Args:
            audio_path: File path to WAV audio.
            audio_array: In-memory 1D float32 audio waveform.
            transcript_input: Pre-computed transcript (file path, list, or dict).
            metadata: Optional lecture metadata dictionary.
            video_path: Optional path to lecture video file for multimodal slide visual fusion.
            stage_callback: Optional callable(stage_id, event, elapsed_sec) invoked on stage start/done.

        Returns:
            PipelineResult container.
        """
        if audio_path is not None:
            return self.run_from_audio(
                audio_path, metadata=metadata, video_path=video_path, stage_callback=stage_callback
            )
        elif audio_array is not None:
            return self.run_from_audio(
                audio_array, metadata=metadata, video_path=video_path, stage_callback=stage_callback
            )
        elif transcript_input is not None:
            return self.run_from_transcript(
                transcript_input, metadata=metadata, video_path=video_path, stage_callback=stage_callback
            )
        else:
            raise ValueError("Must provide either audio_path, audio_array, or transcript_input.")

    def run_from_audio(
        self,
        audio_input: Union[str, Path, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        video_path: Optional[Union[str, Path]] = None,
        stage_callback: Optional[Any] = None,
    ) -> PipelineResult:
        """Execute full audio workflow: Audio -> Phase 1 ASR -> Phase 2 DBCA -> Cornell Notes."""
        # Validate existence if file path
        if isinstance(audio_input, (str, Path)):
            p = Path(audio_input)
            if not p.exists():
                raise FileNotFoundError(f"Audio file not found: {p.resolve()}")
            if p.stat().st_size == 0:
                raise ValueError(f"Audio file is empty (0 bytes): {p.resolve()}")

        t_start = time.perf_counter()
        asr_time = 0.0
        seg_time = 0.0
        notes_time = 0.0

        try:
            # Stage 1: Phase 1 Acoustic Grounding & Preprocessing
            self._notify_stage(stage_callback, "asr", "start", None)
            t0 = time.perf_counter()
            with self.memory_manager.profile_stage("Phase 1: ASR Transcription"):
                c1_sentences = self.asr_pipeline.process_audio(audio_input)
            asr_time = max(0.0001, time.perf_counter() - t0)
            self._notify_stage(stage_callback, "asr", "done", asr_time)

            # Stage Memory Cleanup: explicitly offload Whisper ASR model from GPU VRAM
            if hasattr(self.asr_pipeline, "unload"):
                self.asr_pipeline.unload()
            self.memory_manager.cleanup()

            # Handle Empty Audio or Pure Silence
            if not c1_sentences:
                logger.info("ASR yielded 0 sentences (audio was silent or sub-second). Emitting empty result.")
                total_time = max(0.0001, time.perf_counter() - t_start)
                empty_seg_payload = {"num_sentences": 0, "topic_segments": []}
                md_notes, json_notes = self.note_generator.generate(empty_seg_payload, lecture_metadata=metadata)

                summary = {
                    "status": "EMPTY_AUDIO",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stage_latencies": {
                        "asr_time_sec": round(asr_time, 4),
                        "segmentation_time_sec": 0.0,
                        "notes_time_sec": 0.0,
                        "total_time_sec": round(total_time, 4),
                    },
                    "num_sentences": 0,
                    "num_segments": 0,
                    "device": self.config.device,
                    "memory_stats": self.memory_manager.get_memory_stats(),
                }
                self.last_execution_stats = summary

                result = PipelineResult(
                    status="EMPTY_AUDIO",
                    num_sentences=0,
                    num_segments=0,
                    phase1_transcript=[],
                    phase2_segmentation=empty_seg_payload,
                    study_notes_markdown=md_notes,
                    study_notes_json=json_notes,
                    execution_summary=summary,
                )
                if self.config.output_dir:
                    result.artifacts = self.export_artifacts(result, self.config.output_dir)
                return result

            # Stage 2: Phase 2 Discourse-Aware Topic Segmentation
            self._notify_stage(stage_callback, "segmentation", "start", None)
            t1 = time.perf_counter()
            with self.memory_manager.profile_stage("Phase 2: Topic Segmentation"):
                c2_payload = self.segmentation_pipeline.segment_transcript(
                    c1_sentences, video_path=video_path
                )
            seg_time = max(0.0001, time.perf_counter() - t1)
            self._notify_stage(stage_callback, "segmentation", "done", seg_time)

            # Stage Memory Cleanup: free neural segmentation weights before Phase 3 Ollama
            if hasattr(self.segmentation_pipeline, "unload_neural_model"):
                self.segmentation_pipeline.unload_neural_model()
            self.memory_manager.cleanup()

            # Stage 3: Phase 3 Cornell Study Notes Synthesis
            self._notify_stage(stage_callback, "notes", "start", None)
            t2 = time.perf_counter()
            with self.memory_manager.profile_stage("Phase 3: Cornell Study Notes Generation"):
                md_notes, json_notes = self.note_generator.generate(c2_payload, lecture_metadata=metadata)
            notes_time = max(0.0001, time.perf_counter() - t2)
            self._notify_stage(stage_callback, "notes", "done", notes_time)

            # Stage 4: Phase 4 Hindi Study Notes Translation (Optional)
            hindi_md_notes = None
            hindi_json_notes = None
            trans_time = 0.0
            if self.config.translate_hindi:
                self._notify_stage(stage_callback, "translation", "start", None)
                # Sequential VRAM Handshake: Evict Ollama before loading PyTorch into VRAM (keep Ollama if translating via GGUF)
                if self.config.translation_backend != "ollama":
                    self.memory_manager.evict_ollama()
                    self.memory_manager.cleanup()

                t3 = time.perf_counter()
                with self.memory_manager.profile_stage("Phase 4: Hindi Study Notes Translation"):
                    notes_obj = getattr(self.note_generator, "last_notes_obj", None) or json_notes
                    hindi_md_notes, hindi_json_notes = self.note_translator.translate_study_notes(
                        notes_obj, auto_unload=True
                    )
                trans_time = max(0.0001, time.perf_counter() - t3)
                self._notify_stage(stage_callback, "translation", "done", trans_time)

            total_time = max(0.0001, time.perf_counter() - t_start)

            latencies = {
                "asr_time_sec": round(asr_time, 4),
                "segmentation_time_sec": round(seg_time, 4),
                "notes_time_sec": round(notes_time, 4),
            }
            if self.config.translate_hindi:
                latencies["translation_time_sec"] = round(trans_time, 4)
            latencies["total_time_sec"] = round(total_time, 4)

            summary = {
                "status": "SUCCESS",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage_latencies": latencies,
                "num_sentences": len(c1_sentences),
                "num_segments": len(c2_payload.get("topic_segments", [])),
                "device": self.config.device,
                "memory_stats": self.memory_manager.get_memory_stats(),
            }
            self.last_execution_stats = summary

            result = PipelineResult(
                status="SUCCESS",
                num_sentences=len(c1_sentences),
                num_segments=len(c2_payload.get("topic_segments", [])),
                phase1_transcript=c1_sentences,
                phase2_segmentation=c2_payload,
                study_notes_markdown=md_notes,
                study_notes_json=json_notes,
                study_notes_hindi_markdown=hindi_md_notes,
                study_notes_hindi_json=hindi_json_notes,
                execution_summary=summary,
            )

            if self.config.output_dir:
                result.artifacts = self.export_artifacts(result, self.config.output_dir)

            return result

        finally:
            # Enforce zero-thrashing eviction upon completion
            try:
                self.segmentation_pipeline.title_generator.evict_model()
            except Exception:
                pass
            try:
                self.note_translator.unload_model()
            except Exception:
                pass
            self.memory_manager.cleanup()

    def run_from_transcript(
        self,
        transcript_input: Union[str, Path, Sequence[Any], Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        video_path: Optional[Union[str, Path]] = None,
        stage_callback: Optional[Any] = None,
    ) -> PipelineResult:
        """Fast-path execution: Pre-computed Transcript -> Phase 2 DBCA -> Cornell Notes."""
        t_start = time.perf_counter()

        try:
            # Stage 1: Load and validate Contract 1 sentences
            self._notify_stage(stage_callback, "asr", "start", None)
            t0 = time.perf_counter()
            with self.memory_manager.profile_stage("Transcript Loading"):
                c1_sentences = TranscriptLoader.load(transcript_input)
            asr_time = max(0.0001, time.perf_counter() - t0)
            self._notify_stage(stage_callback, "asr", "done", asr_time)

            # Stage Memory Cleanup: explicitly offload Whisper ASR model from GPU VRAM
            if hasattr(self.asr_pipeline, "unload"):
                self.asr_pipeline.unload()
            self.memory_manager.cleanup()

            # Handle Empty Input
            if not c1_sentences:
                total_time = max(0.0001, time.perf_counter() - t_start)
                empty_seg_payload = {"num_sentences": 0, "topic_segments": []}
                md_notes, json_notes = self.note_generator.generate(empty_seg_payload, lecture_metadata=metadata)

                summary = {
                    "status": "EMPTY_AUDIO",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "stage_latencies": {
                        "asr_time_sec": round(asr_time, 4),
                        "segmentation_time_sec": 0.0,
                        "notes_time_sec": 0.0,
                        "total_time_sec": round(total_time, 4),
                    },
                    "num_sentences": 0,
                    "num_segments": 0,
                    "device": self.config.device,
                    "memory_stats": self.memory_manager.get_memory_stats(),
                }
                self.last_execution_stats = summary

                result = PipelineResult(
                    status="EMPTY_AUDIO",
                    num_sentences=0,
                    num_segments=0,
                    phase1_transcript=[],
                    phase2_segmentation=empty_seg_payload,
                    study_notes_markdown=md_notes,
                    study_notes_json=json_notes,
                    execution_summary=summary,
                )
                if self.config.output_dir:
                    result.artifacts = self.export_artifacts(result, self.config.output_dir)
                return result

            # Stage 2: Phase 2 Topic Segmentation
            self._notify_stage(stage_callback, "segmentation", "start", None)
            t1 = time.perf_counter()
            with self.memory_manager.profile_stage("Phase 2: Topic Segmentation"):
                c2_payload = self.segmentation_pipeline.segment_transcript(
                    c1_sentences, video_path=video_path
                )
            seg_time = max(0.0001, time.perf_counter() - t1)
            self._notify_stage(stage_callback, "segmentation", "done", seg_time)

            # Stage Memory Cleanup: free neural segmentation weights before Phase 3 Ollama
            if hasattr(self.segmentation_pipeline, "unload_neural_model"):
                self.segmentation_pipeline.unload_neural_model()
            self.memory_manager.cleanup()

            # Stage 3: Phase 3 Cornell Study Notes Synthesis
            self._notify_stage(stage_callback, "notes", "start", None)
            t2 = time.perf_counter()
            with self.memory_manager.profile_stage("Phase 3: Cornell Study Notes Generation"):
                md_notes, json_notes = self.note_generator.generate(c2_payload, lecture_metadata=metadata)
            notes_time = max(0.0001, time.perf_counter() - t2)
            self._notify_stage(stage_callback, "notes", "done", notes_time)

            # Stage 4: Phase 4 Hindi Study Notes Translation (Optional)
            hindi_md_notes = None
            hindi_json_notes = None
            trans_time = 0.0
            if self.config.translate_hindi:
                self._notify_stage(stage_callback, "translation", "start", None)
                # Sequential VRAM Handshake: Evict Ollama before loading PyTorch into VRAM (keep Ollama if translating via GGUF)
                if self.config.translation_backend != "ollama":
                    self.memory_manager.evict_ollama()
                    self.memory_manager.cleanup()

                t3 = time.perf_counter()
                with self.memory_manager.profile_stage("Phase 4: Hindi Study Notes Translation"):
                    notes_obj = getattr(self.note_generator, "last_notes_obj", None) or json_notes
                    hindi_md_notes, hindi_json_notes = self.note_translator.translate_study_notes(
                        notes_obj, auto_unload=True
                    )
                trans_time = max(0.0001, time.perf_counter() - t3)
                self._notify_stage(stage_callback, "translation", "done", trans_time)

            total_time = max(0.0001, time.perf_counter() - t_start)

            latencies = {
                "asr_time_sec": round(asr_time, 4),
                "segmentation_time_sec": round(seg_time, 4),
                "notes_time_sec": round(notes_time, 4),
            }
            if self.config.translate_hindi:
                latencies["translation_time_sec"] = round(trans_time, 4)
            latencies["total_time_sec"] = round(total_time, 4)

            summary = {
                "status": "SUCCESS",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stage_latencies": latencies,
                "num_sentences": len(c1_sentences),
                "num_segments": len(c2_payload.get("topic_segments", [])),
                "device": self.config.device,
                "memory_stats": self.memory_manager.get_memory_stats(),
            }
            self.last_execution_stats = summary

            result = PipelineResult(
                status="SUCCESS",
                num_sentences=len(c1_sentences),
                num_segments=len(c2_payload.get("topic_segments", [])),
                phase1_transcript=c1_sentences,
                phase2_segmentation=c2_payload,
                study_notes_markdown=md_notes,
                study_notes_json=json_notes,
                study_notes_hindi_markdown=hindi_md_notes,
                study_notes_hindi_json=hindi_json_notes,
                execution_summary=summary,
            )

            if self.config.output_dir:
                result.artifacts = self.export_artifacts(result, self.config.output_dir)

            return result

        finally:
            try:
                self.segmentation_pipeline.title_generator.evict_model()
            except Exception:
                pass
            try:
                self.note_translator.unload_model()
            except Exception:
                pass
            self.memory_manager.cleanup()

    def export_artifacts(
        self,
        result: Union[PipelineResult, Dict[str, Any]],
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """Export all 5 pipeline artifacts to output directory.

        Writes:
        1. phase1_transcript.json
        2. phase2_segmentation.json
        3. lecture_study_notes.md
        4. lecture_study_notes.json
        5. execution_summary.json
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        res_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)

        artifacts: Dict[str, Path] = {}

        # 1. phase1_transcript.json
        p1_file = out_path / self.config.transcript_filename
        p1_file.write_text(
            json.dumps(res_dict.get("phase1_transcript", []), indent=2),
            encoding="utf-8",
        )
        artifacts["phase1_transcript"] = p1_file

        # 2. phase2_segmentation.json
        p2_file = out_path / self.config.segmentation_filename
        p2_file.write_text(
            json.dumps(res_dict.get("phase2_segmentation", {}), indent=2),
            encoding="utf-8",
        )
        artifacts["phase2_segmentation"] = p2_file

        # 3. lecture_study_notes.md
        md_file = out_path / self.config.notes_markdown_filename
        md_file.write_text(
            res_dict.get("study_notes_markdown", ""),
            encoding="utf-8",
        )
        artifacts["study_notes_markdown"] = md_file

        # 4. lecture_study_notes.json
        json_file = out_path / self.config.notes_json_filename
        json_file.write_text(
            json.dumps(res_dict.get("study_notes_json", {}), indent=2),
            encoding="utf-8",
        )
        artifacts["study_notes_json"] = json_file

        # 5. lecture_study_notes_hi.md (if generated)
        if res_dict.get("study_notes_hindi_markdown"):
            hi_md_file = out_path / self.config.notes_hindi_markdown_filename
            hi_md_file.write_text(
                res_dict["study_notes_hindi_markdown"],
                encoding="utf-8",
            )
            artifacts["study_notes_hindi_markdown"] = hi_md_file

        # 6. lecture_study_notes_hi.json (if generated)
        if res_dict.get("study_notes_hindi_json"):
            hi_json_file = out_path / self.config.notes_hindi_json_filename
            hi_json_file.write_text(
                json.dumps(res_dict["study_notes_hindi_json"], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            artifacts["study_notes_hindi_json"] = hi_json_file

        # 7. execution_summary.json
        summary_file = out_path / self.config.execution_summary_filename
        summary_file.write_text(
            json.dumps(res_dict.get("execution_summary", {}), indent=2),
            encoding="utf-8",
        )
        artifacts["execution_summary"] = summary_file

        return artifacts


# ==============================================================================
# 5. CLI Interface & Entry Point
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build comprehensive command-line argument parser for E2E lecture pipeline."""
    parser = argparse.ArgumentParser(
        description="End-to-End Educational Lecture Pipeline: Audio/Transcript -> Phase 1 -> Phase 2 -> Cornell Study Notes"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to input audio file (16 kHz mono WAV)",
    )
    input_group.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (.mp4, .mkv, .webm)",
    )
    input_group.add_argument(
        "--url",
        "--youtube-url",
        type=str,
        default=None,
        dest="youtube_url",
        help="YouTube lecture URL to download and process end-to-end",
    )
    input_group.add_argument(
        "--transcript-json",
        "--transcript",
        type=str,
        default=None,
        dest="transcript_json",
        help="Path to pre-computed Interface Contract 1 JSON transcript file",
    )
    input_group.add_argument(
        "--fixture",
        type=str,
        choices=["synthetic_10min", "mod01lec01"],
        default=None,
        help="Pre-loaded test fixture name",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save exported artifacts",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Execute offline without downloading models or querying Ollama",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        default=False,
        help="Force deterministic offline fallback for fast execution",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Compute device for ASR and Segmentation (default: cuda)",
    )
    parser.add_argument(
        "--whisper-compute-type",
        type=str,
        default="int8_float16",
        help="Quantization compute type for faster-whisper (default: int8_float16 on GPU, int8 on CPU)",
    )
    parser.add_argument(
        "--whisper-model-size",
        type=str,
        default="large-v3-turbo",
        help="Whisper model size (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--whisper-beam-size",
        type=int,
        default=1,
        help="Whisper decoding beam size (default: 1 for greedy decoding)",
    )
    parser.add_argument(
        "--whisper-batch-size",
        type=int,
        default=8,
        help="BatchedInferencePipeline parallel GPU chunk batch size (default: 8)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Educational Lecture Study Notes",
        help="Lecture title for note generation metadata",
    )
    parser.add_argument(
        "--course",
        type=str,
        default="Computer Science",
        help="Course name for note generation metadata",
    )
    parser.add_argument(
        "--segmentation-checkpoint",
        type=str,
        default="checkpoints/segmentation_qwen3_embed_multimodal/best_model.pt",
        help="Path to trained Phase 2 topic segmentation checkpoint (default: multimodal Qwen3-Embedding + DBCA champion)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.24,
        help="Boundary probability threshold for 1D-NMS (default: 0.24 for Qwen3-Embedding + DBCA champion)",
    )
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=6,
        help="Radius for 1D Non-Maximum Suppression in sentences (default: 6)",
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=15,
        help="Minimum sentences per topic segment (default: 15)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model name (default: {DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        type=str,
        default="5m",
        help="Ollama keep-alive duration (e.g. '5m' to pin model in VRAM across segments, '0' for immediate eviction)",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=8192,
        help="Context window size for Ollama (default: 8192 for 100%% 6GB GPU residency)",
    )
    parser.add_argument(
        "--ollama-num-gpu",
        type=int,
        default=99,
        help="Number of layers to offload to GPU (default: 99 for all layers)",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=4096,
        help="Max output tokens for Ollama note generation (default: 4096)",
    )
    parser.add_argument(
        "--ollama-reduce-num-predict",
        type=int,
        default=4096,
        help="Max output tokens for Ollama global synthesis (default: 4096)",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        default=False,
        help="Disable saving intermediate JSON artifacts",
    )
    parser.add_argument(
        "--translate-hindi",
        action="store_true",
        default=False,
        help="Enable Phase 4: Translate study notes to Hindi (Devanagari) using fine-tuned Sarvam-1",
    )
    parser.add_argument(
        "--translation-backend",
        type=str,
        choices=["ollama", "pytorch"],
        default="ollama",
        help="Phase 4 translation engine (default: ollama merged Q4_K_M GGUF with pytorch fallback)",
    )
    parser.add_argument(
        "--translation-ollama-model",
        type=str,
        default="sarvam1-hi-step200",
        help="Ollama model name for merged Sarvam-1 GGUF (default: sarvam1-hi-step200)",
    )
    parser.add_argument(
        "--translation-adapter-path",
        type=str,
        default="models/sarvam1_subtopic_script_constrained/best_checkpoint",
        help="Path to fine-tuned Sarvam-1 LoRA adapter checkpoint (default: models/sarvam1_subtopic_script_constrained/best_checkpoint)",
    )
    parser.add_argument(
        "--translation-device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Compute device for translation inference (default: cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point for E2E lecture processing."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.output_dir) if args.output_dir else None

    config = E2EPipelineConfig(
        device=args.device,
        whisper_compute_type=args.whisper_compute_type,
        whisper_model_size=args.whisper_model_size,
        whisper_beam_size=args.whisper_beam_size,
        whisper_batch_size=args.whisper_batch_size,
        boundary_threshold=args.threshold,
        nms_radius=args.nms_radius,
        min_segment_length=args.min_segment_length,
        segmentation_checkpoint=args.segmentation_checkpoint,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_keep_alive=args.ollama_keep_alive,
        ollama_num_ctx=args.ollama_num_ctx,
        ollama_num_gpu=args.ollama_num_gpu,
        ollama_num_predict=args.ollama_num_predict,
        ollama_reduce_num_predict=args.ollama_reduce_num_predict,
        force_fallback=args.force_fallback or args.offline,
        offline=args.offline,
        output_dir=out_dir,
        save_intermediate=not args.no_save_intermediate,
        translate_hindi=args.translate_hindi,
        translation_backend=args.translation_backend,
        translation_ollama_model=args.translation_ollama_model,
        translation_adapter_path=args.translation_adapter_path,
        translation_device=args.translation_device,
        course=args.course,
        title=args.title,
    )

    pipeline = E2ELecturePipeline(config=config)

    meta = {
        "course": args.course,
        "lecture_title": args.title,
    }

    if args.audio:
        logger.info("Running pipeline from audio file: %s", args.audio)
        result = pipeline.run(audio_path=args.audio, metadata=meta)
    elif args.video:
        logger.info("Extracting audio from video file: %s", args.video)
        v_path = Path(args.video)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")
        audio_dir = Path("audios")
        audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audio_dir / f"{v_path.stem}_16k.wav"
        import subprocess
        cmd = [
            "ffmpeg", "-i", str(v_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path), "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Extracted 16 kHz WAV: %s", wav_path)
        result = pipeline.run(audio_path=wav_path, video_path=v_path, metadata=meta)
    elif args.youtube_url:
        logger.info("Downloading YouTube lecture video: %s", args.youtube_url)
        from src.video_downloader import download_video
        video_dir = Path("videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = download_video(args.youtube_url, output_dir=str(video_dir))
        logger.info("Downloaded video to: %s", video_path)
        audio_dir = Path("audios")
        audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audio_dir / f"{Path(video_path).stem}_16k.wav"
        import subprocess
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(wav_path), "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Extracted 16 kHz WAV: %s", wav_path)
        result = pipeline.run(audio_path=wav_path, video_path=video_path, metadata=meta)
    elif args.transcript_json:
        logger.info("Running pipeline from transcript JSON: %s", args.transcript_json)
        result = pipeline.run(transcript_input=args.transcript_json, metadata=meta)
    elif args.fixture:
        logger.info("Running pipeline from fixture: %s", args.fixture)
        if args.fixture == "synthetic_10min":
            from tests.fixtures.audio_fixture import get_synthetic_10min_lecture_fixture
            fixture = get_synthetic_10min_lecture_fixture()
            result = pipeline.run(
                transcript_input=fixture["sentences"],
                metadata=fixture.get("metadata", meta),
            )
        elif args.fixture == "mod01lec01":
            from tests.fixtures.audio_fixture import get_mod01lec01_sample_sentences
            raw_sents = get_mod01lec01_sample_sentences(count=25)
            result = pipeline.run(transcript_input=raw_sents, metadata=meta)
        else:
            raise ValueError(f"Unknown fixture name: {args.fixture}")
    else:
        logger.error("No valid input specified.")
        return 1

    logger.info(
        "Pipeline finished with status [%s]. Sentences: %d, Segments: %d",
        result.status,
        result.num_sentences,
        result.num_segments,
    )

    if out_dir:
        logger.info("Artifacts saved to: %s", out_dir.resolve())
        for name, path in result.artifacts.items():
            logger.info(" - %s: %s", name, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
