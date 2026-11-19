"""Silero VAD v4 Silence-Guided Audio Chunker.

Implements dynamic silence-guided audio chunking bounded between 10.0s and 28.0s,
preventing Whisper 30s context window clipping while avoiding short fragmented segments.
Provides zero-dependency adaptive RMS energy and zero-crossing VAD fallback.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np


@dataclass
class AudioChunk:
    """Represents a bounded speech chunk extracted by dynamic VAD."""

    chunk_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_sample: int
    end_sample: int
    audio_data: np.ndarray  # 1D float32 array, 16 kHz mono
    split_reason: Literal["silence_pause", "min_energy", "hard_limit", "audio_end"]
    silence_before_sec: float = 0.0
    silence_after_sec: float = 0.0


@dataclass
class VADChunkConfig:
    """Configuration parameters for dynamic silence-guided chunking."""

    sample_rate: int = 16000
    min_chunk_duration: float = 10.0  # Lower bound: 10.0s
    max_chunk_duration: float = 28.0  # Upper bound: 28.0s (safety margin < 30.0s)
    vad_threshold: float = 0.50  # Speech probability threshold
    min_speech_duration_ms: int = 250  # Min continuous speech duration
    min_silence_duration_ms: int = 200  # Min continuous silence duration
    speech_pad_ms: int = 100  # Padding for speech boundaries
    engine_preference: Literal["onnx", "torch", "energy", "auto"] = "auto"
    onnx_model_path: Optional[str] = None
    force_fallback: bool = False  # Deterministic test mode toggle


class SileroVADChunker:
    """Dynamic silence-guided audio chunker utilizing Silero VAD v4 with

    adaptive energy fallback. Operates strictly in CPU System RAM.
    """

    def __init__(self, config: Optional[VADChunkConfig] = None):
        self.config = config or VADChunkConfig()
        self._engine: str = "none"
        self._model: Any = None
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize VAD engine: ONNX runtime, PyTorch, or synthetic energy fallback."""
        if self.config.force_fallback:
            self._engine = "energy"
            return

        pref = self.config.engine_preference

        # Attempt 1: ONNX Runtime
        if pref in ("auto", "onnx"):
            try:
                import onnxruntime as ort

                model_path = self.config.onnx_model_path or self._resolve_onnx_model()
                if model_path and Path(model_path).exists():
                    opts = ort.SessionOptions()
                    opts.inter_op_num_threads = 1
                    opts.intra_op_num_threads = 2
                    self._model = ort.InferenceSession(
                        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
                    )
                    self._engine = "onnx"
                    return
            except Exception:
                pass

        # Attempt 2: PyTorch Hub (Silero VAD v4)
        if pref in ("auto", "torch"):
            try:
                import torch

                torch.set_num_threads(2)
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                )
                self._model = (model, utils)
                self._engine = "torch"
                return
            except Exception:
                pass

        # Attempt 3: Robust Energy-Based Synthetic Fallback (zero external dependencies)
        self._engine = "energy"

    def _resolve_onnx_model(self) -> Optional[str]:
        """Locate cached silero_vad.onnx in standard cache locations."""
        candidates = [
            Path.home() / ".cache" / "silero_vad.onnx",
            Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master" / "src" / "silero_vad" / "data" / "silero_vad.onnx",
            Path("/tmp/silero_vad.onnx"),
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def ingest_audio(self, audio_or_path: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Ingest audio from file path or array, validating and converting to 16 kHz float32 mono [-1.0, 1.0]."""
        if isinstance(audio_or_path, (str, Path)):
            path = Path(audio_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Audio file does not exist: {path.resolve()}")
            if path.stat().st_size == 0:
                return np.zeros(0, dtype=np.float32)

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

        elif isinstance(audio_or_path, np.ndarray):
            orig_dtype = audio_or_path.dtype
            if audio_or_path.ndim > 1:
                audio_or_path = np.mean(audio_or_path, axis=-1)
            audio = audio_or_path.astype(np.float32)
            max_val = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
            # Only divide by 32768.0 if raw 16-bit integer PCM or if unscaled amplitude exceeds 500.0.
            # Never divide float32 audio with normal/DC amplitudes by 32768.0.
            if orig_dtype == np.int16 or (orig_dtype != np.float32 and max_val > 500.0) or max_val > 500.0:
                audio = audio / 32768.0
            return np.clip(audio, -1.0, 1.0).astype(np.float32)

        raise TypeError(f"Unsupported audio input type: {type(audio_or_path)}")

    def get_speech_probabilities(self, audio: np.ndarray, frame_size: int = 512) -> np.ndarray:
        """Compute frame-level speech probabilities across the waveform.

        Returns 1D array of float32 probabilities in [0.0, 1.0] at 31.25 Hz (512 samples/frame).
        """
        if len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        if self._engine == "onnx" and self._model is not None:
            try:
                probs = []
                h = np.zeros((2, 1, 64), dtype=np.float32)
                c = np.zeros((2, 1, 64), dtype=np.float32)
                sr = np.array([self.config.sample_rate], dtype=np.int64)

                n_frames = len(audio) // frame_size
                for i in range(n_frames):
                    chunk = audio[i * frame_size : (i + 1) * frame_size].reshape(1, -1)
                    inputs = {
                        "input": chunk,
                        "sr": sr,
                        "h": h,
                        "c": c,
                    }
                    out, h, c = self._model.run(None, inputs)
                    probs.append(float(out[0, 0]))
                return np.array(probs, dtype=np.float32)
            except Exception:
                pass  # Fall back to adaptive energy

        if self._engine == "torch" and self._model is not None:
            try:
                import torch

                model, utils = self._model
                get_speech_timestamps = utils[0]
                wav_tensor = torch.from_numpy(audio)
                n_frames = len(audio) // frame_size
                probs = []
                for i in range(n_frames):
                    chunk = wav_tensor[i * frame_size : (i + 1) * frame_size]
                    prob = model(chunk, self.config.sample_rate).item()
                    probs.append(prob)
                return np.array(probs, dtype=np.float32)
            except Exception:
                pass

        # Adaptive Energy & Zero Crossing Rate Fallback
        return self._compute_adaptive_energy_vad(audio, frame_size=frame_size)

    def _compute_adaptive_energy_vad(self, audio: np.ndarray, frame_size: int = 512) -> np.ndarray:
        """Adaptive RMS energy + Zero-Crossing Rate fallback VAD.

        Accurately differentiates voiced/unvoiced educational speech from silence/noise floors.
        """
        n_samples = len(audio)
        n_frames = n_samples // frame_size
        if n_frames == 0:
            return np.zeros(0, dtype=np.float32)

        # Truncate to whole frames
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)

        # 1. RMS Energy per frame
        energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)

        # Check pure silence / flatline DC offset (zero AC power / zero variance)
        max_e = np.max(energy)
        if max_e < 1e-4 or np.var(audio) < 1e-7:
            return np.zeros(n_frames, dtype=np.float32)

        # 2. Zero Crossing Rate (helps detect fricatives/unvoiced speech)
        zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

        # 3. Dynamic Noise Floor Estimation (15th percentile of lowest 50% energy)
        sorted_e = np.sort(energy)
        noise_floor = sorted_e[int(0.15 * len(sorted_e))]
        noise_floor = max(noise_floor, 1e-5)

        # Peak energy level (95th percentile)
        peak_level = sorted_e[int(0.95 * (len(sorted_e) - 1))]
        dynamic_range = max(peak_level - noise_floor, 1e-4)

        # High energy invariant: if signal has high RMS (>0.05) with collapsed dynamic range,
        # it is a steady voiced sound/clipped wave, NOT silence.
        if peak_level >= 0.05 and dynamic_range < 0.01:
            probs = np.ones(n_frames, dtype=np.float32)
            probs[energy < 5e-4] = 0.0
            return probs

        # Normalized energy metric
        norm_e = (energy - noise_floor) / dynamic_range
        norm_e = np.clip(norm_e, 0.0, 1.0)

        # 4. Sigmoidal Mapping
        # Center decision around 20% of dynamic range above noise floor
        # Voiced frames typically exhibit energy > noise_floor + 0.15 * dynamic_range
        logits = 12.0 * (norm_e - 0.18) + 2.0 * (zcr - 0.05)
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Absolute silence clamp
        probs[energy < 5e-4] = 0.0

        return probs.astype(np.float32)

    def extract_silence_intervals(
        self, audio: np.ndarray, frame_size: int = 512
    ) -> List[Tuple[float, float]]:
        """Extract silence time intervals (start_sec, end_sec) where speech is absent.

        Enforces minimum silence duration thresholding.
        """
        probs = self.get_speech_probabilities(audio, frame_size=frame_size)
        if len(probs) == 0:
            return []

        frame_duration = frame_size / self.config.sample_rate  # 0.032s (32ms)
        is_speech = probs >= self.config.vad_threshold

        # Morphological smoothing: bridge small speech dropouts (< min_speech_duration)
        min_speech_frames = max(1, int(self.config.min_speech_duration_ms / 1000.0 / frame_duration))
        min_silence_frames = max(1, int(self.config.min_silence_duration_ms / 1000.0 / frame_duration))

        # Identify raw silence frames
        is_silence = ~is_speech

        # Group contiguous silence intervals
        silence_intervals: List[Tuple[float, float]] = []
        in_silence = False
        start_frame = 0

        for idx, sil in enumerate(is_silence):
            if sil and not in_silence:
                in_silence = True
                start_frame = idx
            elif not sil and in_silence:
                in_silence = False
                sil_len = idx - start_frame
                if sil_len >= min_silence_frames:
                    silence_intervals.append(
                        (start_frame * frame_duration, idx * frame_duration)
                    )

        if in_silence:
            sil_len = len(is_silence) - start_frame
            if sil_len >= min_silence_frames:
                silence_intervals.append(
                    (start_frame * frame_duration, len(is_silence) * frame_duration)
                )

        return silence_intervals

    def chunk_audio(self, audio_or_path: Union[str, Path, np.ndarray]) -> List[AudioChunk]:
        """Execute dynamic silence-guided chunking between 10s and 28s.

        Guarantees:
        1. 10.0s <= chunk.duration <= 28.0s (except final remainder chunk or if total audio < 10s).
        2. Splits occur at the center of natural pauses where available.
        3. Global sample and second offsets are strictly continuous and preserved.
        """
        audio = self.ingest_audio(audio_or_path)
        total_samples = len(audio)
        if total_samples == 0:
            return []

        sr = self.config.sample_rate
        total_duration = total_samples / float(sr)

        # Short audio edge case (< min_chunk_duration)
        if total_duration <= self.config.max_chunk_duration:
            return [
                AudioChunk(
                    chunk_id=0,
                    start_sec=0.0,
                    end_sec=total_duration,
                    duration_sec=total_duration,
                    start_sample=0,
                    end_sample=total_samples,
                    audio_data=audio,
                    split_reason="audio_end",
                    silence_before_sec=0.0,
                    silence_after_sec=0.0,
                )
            ]

        # Extract silence intervals and speech probabilities
        silence_intervals = self.extract_silence_intervals(audio)
        speech_probs = self.get_speech_probabilities(audio)
        frame_duration = 512 / sr

        chunks: List[AudioChunk] = []
        t_curr = 0.0
        chunk_idx = 0

        min_dur = self.config.min_chunk_duration  # 10.0
        max_dur = self.config.max_chunk_duration  # 28.0

        while t_curr < total_duration:
            t_rem = total_duration - t_curr

            # If remaining duration fits in max_chunk_duration, make it the final chunk
            if t_rem <= max_dur:
                start_samp = int(round(t_curr * sr))
                end_samp = total_samples
                chunk_audio_slice = audio[start_samp:end_samp]
                chunks.append(
                    AudioChunk(
                        chunk_id=chunk_idx,
                        start_sec=t_curr,
                        end_sec=total_duration,
                        duration_sec=round(t_rem, 4),
                        start_sample=start_samp,
                        end_sample=end_samp,
                        audio_data=chunk_audio_slice,
                        split_reason="audio_end",
                        silence_before_sec=0.0,
                        silence_after_sec=0.0,
                    )
                )
                break

            # Lookahead constraint:
            # 1. Minimum chunk duration: current chunk must be >= min_dur
            # 2. Maximum chunk duration: current chunk must be <= max_dur
            # 3. Remainder lookahead: remainder after split MUST be >= min_dur
            w_start = t_curr + min_dur
            w_end = min(t_curr + max_dur, total_duration - min_dur)

            # If remainder fits in 2 * max_dur (terminal 2-chunk horizon),
            # balance the last two chunks and ensure Chunk 2 <= max_dur
            if t_rem <= 2.0 * max_dur:
                w_start = max(w_start, total_duration - max_dur)
                t_target = t_curr + t_rem / 2.0
            else:
                t_target = t_curr + (min_dur + max_dur) / 2.0

            # Find overlapping silence intervals
            best_pause_midpoint: Optional[float] = None
            best_pause_dur: float = 0.0
            best_pause_dist: float = float("inf")

            minor_pause_midpoint: Optional[float] = None
            minor_pause_dur: float = 0.0
            minor_pause_dist: float = float("inf")

            for s_start, s_end in silence_intervals:
                overlap_start = max(s_start, w_start)
                overlap_end = min(s_end, w_end)
                if overlap_end > overlap_start:
                    dur = overlap_end - overlap_start
                    mid = (overlap_start + overlap_end) / 2.0
                    dist = abs(mid - t_target)
                    # Major pause >= 300ms
                    if dur >= 0.30:
                        if dur > best_pause_dur + 0.05 or (abs(dur - best_pause_dur) <= 0.05 and dist < best_pause_dist):
                            best_pause_dur = dur
                            best_pause_midpoint = mid
                            best_pause_dist = dist
                    # Minor pause >= 100ms
                    elif dur >= 0.10:
                        if dur > minor_pause_dur + 0.05 or (abs(dur - minor_pause_dur) <= 0.05 and dist < minor_pause_dist):
                            minor_pause_dur = dur
                            minor_pause_midpoint = mid
                            minor_pause_dist = dist

            # Decision Logic
            t_split: float
            split_reason: Literal["silence_pause", "min_energy", "hard_limit", "audio_end"]

            if best_pause_midpoint is not None:
                t_split = best_pause_midpoint
                split_reason = "silence_pause"
            elif minor_pause_midpoint is not None:
                t_split = minor_pause_midpoint
                split_reason = "silence_pause"
            else:
                # Priority 3: Local energy minimum in window
                idx_w_start = int(w_start / frame_duration)
                idx_w_end = int(w_end / frame_duration)
                if idx_w_end > idx_w_start and len(speech_probs) > idx_w_start:
                    sub_probs = speech_probs[idx_w_start:idx_w_end]
                    if len(sub_probs) > 0:
                        min_offset = int(np.argmin(sub_probs))
                        t_split = (idx_w_start + min_offset) * frame_duration
                        split_reason = "min_energy"
                    else:
                        t_split = t_target
                        split_reason = "hard_limit"
                else:
                    t_split = t_target
                    split_reason = "hard_limit"

            # Enforce strict boundary clamp
            t_split = max(w_start, min(w_end, t_split))
            t_split = round(t_split, 4)
            chunk_duration = round(t_split - t_curr, 4)

            start_samp = int(round(t_curr * sr))
            end_samp = min(total_samples, int(round(t_split * sr)))
            chunk_audio_slice = audio[start_samp:end_samp]

            chunks.append(
                AudioChunk(
                    chunk_id=chunk_idx,
                    start_sec=t_curr,
                    end_sec=t_split,
                    duration_sec=chunk_duration,
                    start_sample=start_samp,
                    end_sample=end_samp,
                    audio_data=chunk_audio_slice,
                    split_reason=split_reason,
                    silence_before_sec=0.0,
                    silence_after_sec=best_pause_dur if split_reason == "silence_pause" else 0.0,
                )
            )

            chunk_idx += 1
            t_curr = t_split

        return chunks
