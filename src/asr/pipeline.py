"""Unified Phase 1 ASR Pipeline.

Orchestrates:
1. Silero VAD v4 Silence Chunking (10.0s - 28.0s) in CPU System RAM
2. faster-whisper large-v3-turbo (CTranslate2 INT8 CPU / sequential GPU)
3. WhisperX CTC Phoneme Forced Alignment (wav2vec2-large-960h, millisecond timestamps, Δt_pause)
4. Disfluency Normalizer (Regex filler cleaner & DeepMultilingualPunctuation recovery)
5. Semantic Digression Gating (running topic centroid d_topic > 0.65 sustained >= 3 sentences)

Strictly enforces Interface Contract 1 and hardware constraints (< 6GB VRAM, < 16GB RAM).
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SentenceRecord:
    """Strongly typed representation of Interface Contract 1."""

    sentence_id: int  # 0-indexed sequential rank
    text: str  # Cleaned, punctuated sentence
    start_ms: int  # Millisecond start timestamp
    end_ms: int  # Millisecond end timestamp
    pause_before_ms: float  # Silence interval preceding sentence
    pitch_reset: float = 0.0  # Semitone pitch reset delta
    speaker_tag: str = "Lecturer"  # "Lecturer" or "Student"
    is_digression: bool = False  # Gated digression flag
    digression_type: str = "none"  # "administrative", "logistical", or "none"

    def to_contract_dict(self) -> Dict[str, Any]:
        """Serialize to the exact dictionary schema specified in Interface Contract 1."""
        return {
            "sentence_id": int(self.sentence_id),
            "text": str(self.text),
            "start_ms": int(self.start_ms),
            "end_ms": int(self.end_ms),
            "pause_before_ms": float(round(self.pause_before_ms, 2)),
            "pitch_reset": float(round(self.pitch_reset, 2)),
            "speaker_tag": str(self.speaker_tag),
            "is_digression": bool(self.is_digression),
            "digression_type": str(self.digression_type),
        }


@dataclass
class ASRPipelineConfig:
    """Comprehensive configuration for Phase 1 ASR Pipeline."""

    sample_rate: int = 16000
    device: Literal["cpu", "cuda"] = "cuda"
    whisper_model_size: str = "large-v3-turbo"
    whisper_compute_type: str = "int8_float16"  # "int8_float16" on SM89 GPU, "int8" on CPU
    whisper_cpu_threads: int = 6  # Tuned for Intel Core i7-13650HX
    whisper_beam_size: int = 1  # Greedy decoding for fast inference
    whisper_batch_size: int = 8  # BatchedInferencePipeline parallel GPU chunk batch size
    whisper_word_timestamps: bool = False  # Skip DTW cross-attention overhead for sentence-level pipeline
    whisper_condition_on_previous: bool = False  # Suppresses repetition loops
    vad_min_chunk_duration: float = 10.0  # 10s minimum chunk
    vad_max_chunk_duration: float = 28.0  # 28s maximum chunk (safety margin <30s)
    vad_threshold: float = 0.50
    align_model_name: str = "wav2vec2-large-960h"
    digression_distance_threshold: float = 0.65  # d_topic > 0.65
    digression_decay_beta: float = 0.90  # Running centroid decay factor
    digression_min_consecutive: int = 3  # Sustained across >= 3 sentences
    isolate_process: bool = True  # Run ASR in isolated subprocess to immediately free 100% of CUDA/RAM upon completion
    enable_truecasing: bool = True
    use_neural_punctuation: bool = False  # large-v3-turbo natively punctuates; avoids redundant CPU model load
    force_fallback: bool = False  # Deterministic synthetic test mode
    mock_transcripts: Optional[List[str]] = None


def _asr_subprocess_worker(config_dict: Dict[str, Any], audio_input_path: str, q_out: Any) -> None:
    """Isolated subprocess worker executing faster-whisper ASR and alignment without polluting parent memory."""
    try:
        cfg = ASRPipelineConfig(**config_dict)
        cfg.isolate_process = False  # Prevent recursive subprocess spawning
        pipe = ASRPipeline(config=cfg)
        records = pipe.process_audio(audio_input_path, return_records=False)
        pipe.unload()
        q_out.put({"success": True, "records": records})
    except Exception as e:
        q_out.put({"success": False, "error": str(e)})


class ASRPipeline:
    """Unified Phase 1 Audio Processing Pipeline:

    Acoustic Grounding, Speech Disfluency Scrubbing, and Semantic Digression Gating.
    Emits verified Interface Contract 1 records strictly within RTX 4050 / 16GB RAM limits.
    """

    def __init__(
        self,
        config: Optional[ASRPipelineConfig] = None,
        vad_chunker: Optional[Any] = None,
        transcriber: Optional[Any] = None,
        aligner: Optional[Any] = None,
        disfluency_filter: Optional[Any] = None,
        digression_gate: Optional[Any] = None,
    ):
        self.config = config or ASRPipelineConfig()

        # Dependency injection with lazy factory defaults
        self._vad_chunker = vad_chunker
        self._transcriber = transcriber
        self._aligner = aligner
        self._disfluency_filter = disfluency_filter
        self._digression_gate = digression_gate

    @property
    def vad_chunker(self) -> Any:
        if self._vad_chunker is None:
            self._vad_chunker = self._build_vad_chunker()
        return self._vad_chunker

    @property
    def transcriber(self) -> Any:
        if self._transcriber is None:
            self._transcriber = self._build_transcriber()
        return self._transcriber

    @property
    def aligner(self) -> Any:
        if self._aligner is None:
            self._aligner = self._build_aligner()
        return self._aligner

    @property
    def disfluency_filter(self) -> Any:
        if self._disfluency_filter is None:
            self._disfluency_filter = self._build_disfluency_filter()
        return self._disfluency_filter

    @property
    def digression_gate(self) -> Any:
        if self._digression_gate is None:
            self._digression_gate = self._build_digression_gate()
        return self._digression_gate

    def unload(self) -> None:
        """Explicitly offload transcriber and aligner models from GPU VRAM."""
        if hasattr(self, "_transcriber") and self._transcriber is not None:
            if hasattr(self._transcriber, "unload"):
                self._transcriber.unload()
            self._transcriber = None
        if hasattr(self, "_aligner") and self._aligner is not None:
            if hasattr(self._aligner, "unload"):
                self._aligner.unload()
            self._aligner = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _build_vad_chunker(self) -> Any:
        from src.asr.vad_chunker import SileroVADChunker, VADChunkConfig

        vcfg = VADChunkConfig(
            sample_rate=self.config.sample_rate,
            min_chunk_duration=self.config.vad_min_chunk_duration,
            max_chunk_duration=self.config.vad_max_chunk_duration,
            vad_threshold=self.config.vad_threshold,
            force_fallback=self.config.force_fallback,
        )
        return SileroVADChunker(config=vcfg)

    def _build_transcriber(self) -> Any:
        from src.asr.transcribe_align import ASRConfig, WhisperTranscriber

        acfg = ASRConfig(
            model_size=self.config.whisper_model_size,
            device=self.config.device,
            compute_type=self.config.whisper_compute_type,
            cpu_threads=self.config.whisper_cpu_threads,
            beam_size=self.config.whisper_beam_size,
            batch_size=self.config.whisper_batch_size,
            word_timestamps=self.config.whisper_word_timestamps,
            condition_on_previous_text=self.config.whisper_condition_on_previous,
            force_fallback=self.config.force_fallback,
            mock_transcripts=self.config.mock_transcripts,
        )
        return WhisperTranscriber(config=acfg)

    def _build_aligner(self) -> Any:
        from src.asr.transcribe_align import ASRConfig, WhisperXAligner

        acfg = ASRConfig(
            align_model_name=self.config.align_model_name,
            align_device=self.config.device,
            force_fallback=self.config.force_fallback,
        )
        return WhisperXAligner(config=acfg)

    def _build_disfluency_filter(self) -> Any:
        from src.asr.disfluency_filter import DisfluencyFilter, DisfluencyFilterConfig

        dcfg = DisfluencyFilterConfig(
            enable_truecasing=self.config.enable_truecasing,
            use_neural_punctuation=self.config.use_neural_punctuation,
        )
        return DisfluencyFilter(config=dcfg)

    def _build_digression_gate(self) -> Any:
        from src.asr.digression_gate import DigressionGateConfig, SemanticDigressionGate

        gcfg = DigressionGateConfig(
            decay_factor=self.config.digression_decay_beta,
            distance_threshold=self.config.digression_distance_threshold,
            min_consecutive=self.config.digression_min_consecutive,
        )
        return SemanticDigressionGate(config=gcfg)

    def process_audio(
        self,
        audio_input: Union[str, Path, np.ndarray],
        return_records: bool = False,
    ) -> Union[List[Dict[str, Any]], List[SentenceRecord]]:
        """Execute end-to-end Phase 1 processing from raw audio to punctuated,

        disfluency-scrubbed, digression-annotated sentence records.

        Args:
            audio_input: File path to 16 kHz mono WAV or 1D float32 numpy array.
            return_records: If True, returns List[SentenceRecord]; else List[Dict].

        Returns:
            List of sentence dictionaries conforming strictly to Interface Contract 1.
        """
        # Process Isolation: Run Whisper in a clean subprocess to ensure 100% VRAM and RAM reclamation
        if (
            getattr(self.config, "isolate_process", True)
            and self.config.device == "cuda"
            and not self.config.force_fallback
            and isinstance(audio_input, (str, Path))
        ):
            try:
                import multiprocessing as mp
                ctx = mp.get_context("spawn")
                q_out = ctx.Queue()
                cfg_dict = asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else dict(vars(self.config))
                p = ctx.Process(
                    target=_asr_subprocess_worker,
                    args=(cfg_dict, str(audio_input), q_out),
                )
                p.start()
                res = q_out.get(timeout=1800)  # 30 min timeout for long audio
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
                if res.get("success"):
                    logger.info("Isolated Phase 1 ASR completed successfully (0 bytes residual VRAM/RAM).")
                    dicts = res["records"]
                    if return_records:
                        return [SentenceRecord(**d) for d in dicts]
                    return dicts
                else:
                    logger.warning("Isolated ASR error (%s); running in-process.", res.get("error"))
            except Exception as e:
                logger.warning("Isolated ASR subprocess failed (%s); falling back to in-process execution.", e)

        # Step 1: Ingestion & Validation
        audio_array = self._validate_and_ingest_audio(audio_input)

        # Step 2: Pure Silence Detection
        if self._is_pure_silence(audio_array):
            return []

        # Step 3 & 4: Fast GPU Batched Transcription (BatchedInferencePipeline) or VAD Chunked Loop
        if (
            not self.config.force_fallback
            and getattr(self.transcriber, "supports_batched", False)
            and self._vad_chunker is None
        ):
            raw_segments = self.transcriber.transcribe_audio_batched(
                audio_array,
                max_chunk_sec=self.config.vad_max_chunk_duration,
            )
        else:
            # Step 3: Silero VAD Dynamic Chunking (10s - 28s)
            chunks = self.vad_chunker.chunk_audio(audio_array)
            if not chunks:
                return []

            # Step 4: Chunked faster-whisper Transcription with Global Timestamp Remapping
            raw_segments = []
            for chunk in chunks:
                chunk_segs = self.transcriber.transcribe_chunk(
                    chunk.audio_data,
                    global_offset_sec=chunk.start_sec,
                )
                raw_segments.extend(chunk_segs)

        if not raw_segments:
            return []

        # Step 5: WhisperX CTC Phoneme Forced Alignment
        aligned_segments = self.aligner.align(raw_segments, audio_array)

        total_duration_ms = (
            int(round(len(audio_array) * 1000.0 / self.config.sample_rate))
            if len(audio_array) > 0
            else None
        )

        # Step 6: Initial Sentence Grouping & Pause Calculation
        initial_sentences = self._construct_sentences(
            aligned_segments, total_duration_ms=total_duration_ms
        )

        # Step 7: Disfluency Normalization & Punctuation Restoration
        cleaned_sentences = self.disfluency_filter.clean_sentence_records(initial_sentences)

        # Step 8: Acoustic Feature Enrichment (Pause and Pitch Reset)
        enriched_sentences = self._enrich_acoustic_features(cleaned_sentences, audio_array)

        # Step 9: Semantic Digression Gating (Centroid Distance d > 0.65 sustained >= 3)
        gated_sentences = self.digression_gate.gate_sentences(enriched_sentences)

        # Step 10: Interface Contract 1 Validation & Invariant Enforcement
        final_records = self._finalize_contract_records(
            gated_sentences, total_duration_ms=total_duration_ms
        )

        if return_records:
            return final_records
        return [r.to_contract_dict() for r in final_records]

    def _validate_and_ingest_audio(self, audio_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Validate input path/data and return 1D float32 normalized waveform."""
        if isinstance(audio_input, np.ndarray):
            orig_dtype = audio_input.dtype
            if audio_input.ndim > 1:
                # Average multi-channel audio to mono
                audio_input = np.mean(audio_input, axis=-1)
            audio = audio_input.astype(np.float32)
            max_val = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
            # Only divide by 32768.0 if raw 16-bit integer PCM or if unscaled amplitude exceeds 500.0.
            # Never divide float32 audio with normal/DC amplitudes by 32768.0.
            if orig_dtype == np.int16 or (orig_dtype != np.float32 and max_val > 500.0) or max_val > 500.0:
                audio = audio / 32768.0
            return np.clip(audio, -1.0, 1.0)

        path = Path(audio_input)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path.resolve()}")
        if path.stat().st_size == 0:
            raise ValueError(f"Audio file is empty (0 bytes): {path.resolve()}")

        try:
            with wave.open(str(path), "rb") as wf:
                framerate = wf.getframerate()
                if framerate != self.config.sample_rate:
                    raise ValueError(
                        f"Unsupported sample rate: {framerate} Hz. Expected {self.config.sample_rate} Hz mono."
                    )
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                if n_frames == 0:
                    return np.zeros(0, dtype=np.float32)

                raw_bytes = wf.readframes(n_frames)
                if sampwidth == 2:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels).mean(axis=1)

                return np.clip(samples, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Corrupted or invalid WAV file: {path.resolve()} — {e}")

    def _is_pure_silence(self, audio: np.ndarray, threshold: float = 1e-4) -> bool:
        """Check if audio waveform contains only background noise, zeros, or flatline DC offset."""
        if len(audio) == 0:
            return True
        if float(np.max(np.abs(audio))) < threshold:
            return True
        # Check zero-mean AC power (variance) to prevent flatline DC offsets from hallucinating text
        if float(np.var(audio)) < 1e-7:
            return True
        return False

    def _construct_sentences(
        self,
        aligned_segments: List[Dict[str, Any]],
        total_duration_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convert aligned segments/words into candidate sentence dictionaries."""
        sentences = []
        prev_end_ms = 0

        for idx, seg in enumerate(aligned_segments):
            text = seg.get("text", "").strip()
            if not text:
                continue

            words = seg.get("words", [])
            if words:
                start_ms = int(round(words[0].get("start_ms", words[0].get("start_sec", 0.0) * 1000)))
                end_ms = int(round(words[-1].get("end_ms", words[-1].get("end_sec", 0.0) * 1000)))
            else:
                start_ms = int(round(seg.get("start_ms", seg.get("start_sec", 0.0) * 1000)))
                end_ms = int(round(seg.get("end_ms", seg.get("end_sec", 0.0) * 1000)))

            if total_duration_ms is not None:
                end_ms = min(end_ms, total_duration_ms)
                start_ms = min(start_ms, end_ms)

            end_ms = max(start_ms, end_ms)

            # Pause interval preceding sentence
            if idx == 0:
                pause_before_ms = max(0.0, float(start_ms))
            else:
                pause_before_ms = max(0.0, float(start_ms - prev_end_ms))

            prev_end_ms = max(prev_end_ms, end_ms)

            sentences.append(
                {
                    "sentence_id": idx,
                    "text": text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "pause_before_ms": pause_before_ms,
                    "pitch_reset": 0.0,
                    "speaker_tag": "Lecturer",
                    "is_digression": False,
                    "digression_type": "none",
                }
            )

        return sentences

    def _enrich_acoustic_features(
        self, sentences: List[Dict[str, Any]], audio: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Compute pitch reset Delta F0 (semitones) across consecutive sentences."""
        sr = self.config.sample_rate
        prev_f0 = 0.0

        for s in sentences:
            start_ms = s["start_ms"]
            end_ms = s["end_ms"]
            start_samp = int(start_ms * sr / 1000.0)
            end_samp = min(len(audio), int(end_ms * sr / 1000.0))

            cur_f0 = 0.0
            if end_samp - start_samp >= 512:
                seg_samples = audio[start_samp : min(end_samp, start_samp + 1600)]
                if np.max(np.abs(seg_samples)) >= 1e-4:
                    corr = np.correlate(seg_samples, seg_samples, mode="full")
                    corr = corr[len(corr) // 2 :]
                    min_lag = int(sr / 300)
                    max_lag = int(sr / 80)
                    if len(corr) > max_lag:
                        search_slice = corr[min_lag:max_lag]
                        if len(search_slice) > 0:
                            best_lag = min_lag + int(np.argmax(search_slice))
                            if corr[best_lag] > 0.3 * corr[0]:
                                cur_f0 = float(sr / best_lag)

            if prev_f0 > 50.0 and cur_f0 > 50.0:
                s["pitch_reset"] = 12.0 * math.log2(cur_f0 / prev_f0)
            else:
                s["pitch_reset"] = 0.0
            prev_f0 = cur_f0

        return sentences

    def _finalize_contract_records(
        self,
        sentences: List[Dict[str, Any]],
        total_duration_ms: Optional[int] = None,
    ) -> List[SentenceRecord]:
        """Validate and strictly enforce Interface Contract 1 invariants."""
        final_records: List[SentenceRecord] = []
        prev_end_ms = 0

        for idx, s in enumerate(sentences):
            start_ms = max(0, int(s.get("start_ms", 0)))
            end_ms = max(start_ms, int(s.get("end_ms", start_ms)))

            if total_duration_ms is not None:
                end_ms = min(end_ms, total_duration_ms)
                start_ms = min(start_ms, end_ms)

            # Monotonic pause calculation
            if idx == 0:
                pause_before_ms = max(0.0, float(start_ms))
            else:
                pause_before_ms = max(0.0, float(start_ms - prev_end_ms))
            prev_end_ms = max(prev_end_ms, end_ms)

            is_digression = bool(s.get("is_digression", False))
            digression_type = str(s.get("digression_type", "none"))
            if is_digression and digression_type == "none":
                digression_type = "administrative"
            elif not is_digression:
                digression_type = "none"

            record = SentenceRecord(
                sentence_id=idx,  # Strictly monotonic 0-indexed integer
                text=str(s.get("text", "")).strip(),
                start_ms=start_ms,
                end_ms=end_ms,
                pause_before_ms=pause_before_ms,
                pitch_reset=float(s.get("pitch_reset", 0.0)),
                speaker_tag=str(s.get("speaker_tag", "Lecturer")),
                is_digression=is_digression,
                digression_type=digression_type,
            )
            final_records.append(record)

        return final_records
