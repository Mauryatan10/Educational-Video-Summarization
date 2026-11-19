"""Synthetic 16 kHz Mono WAV Generator & Educational Transcript Fixtures.

This module provides:
1. Pure Python (wave + numpy) 16 kHz mono WAV audio generators for testing ASR,
   VAD, phoneme alignment, and acoustic feature extractors without external assets.
2. Fixture loaders for real educational transcripts (texts/mod01lec01.txt) and
   synthetic educational lecture scenarios.
3. Schema validators and mock generators for Interface Contracts 1 & 2.
"""

from __future__ import annotations

import math
import os
import re
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Default audio constants
DEFAULT_SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
MAX_INT16 = 32767
MIN_INT16 = -32768


def generate_synthetic_wav(
    path: str | Path,
    duration_sec: float = 5.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    silence_intervals: Optional[List[Tuple[float, float]]] = None,
    noise_level: float = 0.0,
) -> Path:
    """Generate a synthetic 16 kHz mono 16-bit PCM WAV file.

    Args:
        path: Output path for WAV file.
        duration_sec: Total duration in seconds.
        sample_rate: Audio sampling frequency in Hz (default 16000).
        frequency: Base tone frequency in Hz (default 440.0).
        amplitude: Peak amplitude in [0.0, 1.0].
        silence_intervals: Optional list of (start_sec, end_sec) tuples where
            signal is muted.
        noise_level: Standard deviation of additive white Gaussian noise.

    Returns:
        Path to the generated WAV file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = max(0, int(round(duration_sec * sample_rate)))
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    # Base harmonic sinusoidal tone
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t)

    # Add Gaussian noise if requested
    if noise_level > 0.0:
        noise = np.random.normal(0.0, noise_level, n_samples).astype(np.float32)
        signal += noise

    # Apply silence intervals
    if silence_intervals:
        for start_s, end_s in silence_intervals:
            idx_start = max(0, int(round(start_s * sample_rate)))
            idx_end = min(n_samples, int(round(end_s * sample_rate)))
            if idx_start < idx_end:
                signal[idx_start:idx_end] = 0.0

    # Clip to valid [-1.0, 1.0] float range and convert to int16
    signal = np.clip(signal, -1.0, 1.0)
    int16_samples = (signal * MAX_INT16).astype(np.int16)

    # Write WAV file
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())

    return path


def generate_silence_wav(
    path: str | Path,
    duration_sec: float = 5.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Generate a pure silence 16 kHz mono WAV file."""
    return generate_synthetic_wav(
        path=path,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        amplitude=0.0,
        noise_level=0.0,
    )


def generate_speech_like_wav(
    path: str | Path,
    duration_sec: float = 10.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    speech_intervals: Optional[List[Tuple[float, float]]] = None,
    f0: float = 130.0,
) -> Path:
    """Generate a synthetic speech-like harmonic signal with natural pauses.

    Synthesizes fundamental frequency F0 plus 3 formant harmonics (e.g. 2*F0, 3*F0, 5*F0)
    modulated by an envelope, mimicking voiced educational speech.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = max(0, int(round(duration_sec * sample_rate)))
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    # Voiced speech simulation (formant synthesis)
    # F0 = 130 Hz (Lecturer male pitch), F1 = 500 Hz, F2 = 1500 Hz, F3 = 2500 Hz
    s0 = 0.35 * np.sin(2.0 * np.pi * f0 * t)
    s1 = 0.25 * np.sin(2.0 * np.pi * (2.0 * f0) * t)
    s2 = 0.15 * np.sin(2.0 * np.pi * (4.0 * f0) * t)
    s3 = 0.08 * np.sin(2.0 * np.pi * (6.0 * f0) * t)
    voiced = s0 + s1 + s2 + s3

    # Add 4 Hz syllabic amplitude modulation envelope
    syllable_envelope = 0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * t)
    signal = voiced * syllable_envelope

    # Default speech intervals if not provided: speech in [0.5, 4.0] and [5.5, 9.5]
    if speech_intervals is None:
        speech_intervals = [(0.5, 4.0), (5.5, 9.5)]

    # Mask everything outside speech intervals as silence
    mask = np.zeros(n_samples, dtype=bool)
    for start_s, end_s in speech_intervals:
        idx_s = max(0, int(round(start_s * sample_rate)))
        idx_e = min(n_samples, int(round(end_s * sample_rate)))
        if idx_s < idx_e:
            mask[idx_s:idx_e] = True

    signal[~mask] = 0.0

    # Slight background acoustic room noise (15 dB SNR)
    room_noise = np.random.normal(0.0, 0.003, n_samples).astype(np.float32)
    signal += room_noise

    signal = np.clip(signal, -1.0, 1.0)
    int16_samples = (signal * MAX_INT16).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())

    return path


def generate_extreme_audio_wav(
    path: str | Path,
    mode: str = "empty",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Generate degenerate audio boundary files for stress testing.

    Modes:
        'empty': 0 frames, valid header.
        'tiny': 0.02s (320 samples), shorter than minimum VAD window.
        'clipping': Extreme saturated square wave at +32767 and -32768.
        'dc_offset': Constant maximum non-zero DC bias (+0.8).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "empty":
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(sample_rate)
            wf.writeframes(b"")
        return path

    if mode == "tiny":
        # 20 milliseconds (320 samples)
        n_samples = int(0.02 * sample_rate)
        samples = (np.sin(np.linspace(0, 10, n_samples)) * MAX_INT16).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        return path

    if mode == "clipping":
        n_samples = sample_rate * 2  # 2 seconds
        # Alternating max positive and max negative
        samples = np.array([MAX_INT16 if (i // 100) % 2 == 0 else MIN_INT16 for i in range(n_samples)], dtype=np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        return path

    if mode == "dc_offset":
        n_samples = sample_rate * 2
        samples = np.full(n_samples, int(0.8 * MAX_INT16), dtype=np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
        return path

    raise ValueError(f"Unknown extreme audio mode: {mode}")


def read_wav_metadata(path: str | Path) -> Dict[str, Any]:
    """Inspect and return technical audio metadata from a WAV header."""
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration_sec = n_frames / float(framerate) if framerate > 0 else 0.0

    return {
        "channels": n_channels,
        "sample_width": sampwidth,
        "framerate": framerate,
        "n_frames": n_frames,
        "duration_sec": duration_sec,
    }


def read_wav_samples(path: str | Path) -> np.ndarray:
    """Read 16-bit mono WAV into float32 numpy array normalized to [-1.0, 1.0]."""
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / float(MAX_INT16)
    samples = np.clip(samples, -1.0, 1.0)
    return samples


# ==============================================================================
# Transcript Fixtures & Loaders
# ==============================================================================

def load_mod01lec01_raw(workspace_root: Optional[str | Path] = None) -> str:
    """Load raw lecture transcript from texts/mod01lec01.txt."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent.parent
    else:
        workspace_root = Path(workspace_root)

    transcript_path = workspace_root / "texts" / "mod01lec01.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {transcript_path}")

    return transcript_path.read_text(encoding="utf-8")


def get_mod01lec01_sample_sentences(count: int = 25) -> List[str]:
    """Extract clean, punctuated sample sentences from texts/mod01lec01.txt."""
    raw = load_mod01lec01_raw()
    # Strip YouTube music / speaker markers like [Music]
    cleaned = re.sub(r"\[.*?\]", " ", raw)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    # Group short transcript lines into natural sentence-length chunks
    sentences: List[str] = []
    buffer: List[str] = []
    for line in lines:
        buffer.append(line)
        if len(" ".join(buffer).split()) >= 12:
            s = " ".join(buffer).capitalize() + "."
            sentences.append(s)
            buffer = []
            if len(sentences) >= count:
                break

    if buffer and len(sentences) < count:
        sentences.append(" ".join(buffer).capitalize() + ".")

    return sentences


def get_synthetic_10min_lecture_fixture() -> Dict[str, Any]:
    """Generate a high-fidelity synthetic 10-minute educational lecture fixture.

    Contains 3 distinct segments:
    1. Operating System Process Abstraction & State Transitions (Definitions & Formulas).
    2. Administrative Digression (4 sentences on homework submission & midterm schedule).
    3. Virtual Memory & Multi-Level Page Tables (Address translation VPN -> PFN).
    """
    sentences_data = [
        # Segment 1: Process Abstraction & Operating Systems
        {
            "sentence_id": 0,
            "text": "Welcome to Operating Systems lecture one on process management and memory abstractions.",
            "start_ms": 1000,
            "end_ms": 5200,
            "pause_before_ms": 0.0,
            "pitch_reset": 0.0,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 1,
            "text": "In this section we formally define the process abstraction as a program in execution.",
            "start_ms": 5800,
            "end_ms": 11000,
            "pause_before_ms": 600.0,
            "pitch_reset": 1.2,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 2,
            "text": "A process address space comprises text, initialized data, heap, and call stack segments.",
            "start_ms": 11500,
            "end_ms": 17200,
            "pause_before_ms": 500.0,
            "pitch_reset": -0.4,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 3,
            "text": "The operating system maintains a Process Control Block or PCB containing CPU registers and state.",
            "start_ms": 17800,
            "end_ms": 23900,
            "pause_before_ms": 600.0,
            "pitch_reset": 0.8,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 4,
            "text": "When switching execution context, total overhead is given by T_switch equals T_save plus T_restore.",
            "start_ms": 24500,
            "end_ms": 31200,
            "pause_before_ms": 600.0,
            "pitch_reset": 2.1,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },

        # Segment 2: Administrative Digression (>= 3 sentences sustained, d_topic > 0.65)
        {
            "sentence_id": 5,
            "text": "Before we proceed further, please remember that assignment number two is due next Wednesday.",
            "start_ms": 33000,
            "end_ms": 39000,
            "pause_before_ms": 1800.0,
            "pitch_reset": 3.5,
            "speaker_tag": "Lecturer",
            "is_digression": True,
            "digression_type": "administrative",
        },
        {
            "sentence_id": 6,
            "text": "The grading portal will strictly close at 11:59 PM with no late extensions permitted.",
            "start_ms": 39500,
            "end_ms": 45100,
            "pause_before_ms": 500.0,
            "pitch_reset": -0.5,
            "speaker_tag": "Lecturer",
            "is_digression": True,
            "digression_type": "administrative",
        },
        {
            "sentence_id": 7,
            "text": "Also the midterm exam seating arrangements have been posted on the course web portal.",
            "start_ms": 45600,
            "end_ms": 51200,
            "pause_before_ms": 500.0,
            "pitch_reset": 0.2,
            "speaker_tag": "Lecturer",
            "is_digression": True,
            "digression_type": "administrative",
        },
        {
            "sentence_id": 8,
            "text": "Please bring your student identification cards and scientific calculators to room 401.",
            "start_ms": 51800,
            "end_ms": 57400,
            "pause_before_ms": 600.0,
            "pitch_reset": -0.8,
            "speaker_tag": "Lecturer",
            "is_digression": True,
            "digression_type": "administrative",
        },

        # Segment 3: Return to Core Pedagogy - Virtual Memory & Page Tables
        {
            "sentence_id": 9,
            "text": "Returning to our core topic, let us now examine virtual memory architecture and paging.",
            "start_ms": 60000,
            "end_ms": 66500,
            "pause_before_ms": 2600.0,
            "pitch_reset": 4.2,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 10,
            "text": "Virtual memory provides each process with the illusion of a contiguous, private memory space.",
            "start_ms": 67100,
            "end_ms": 73200,
            "pause_before_ms": 600.0,
            "pitch_reset": 0.5,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 11,
            "text": "The Memory Management Unit hardware translates virtual page numbers to physical frame numbers.",
            "start_ms": 73800,
            "end_ms": 80400,
            "pause_before_ms": 600.0,
            "pitch_reset": -0.2,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 12,
            "text": "When a page table entry indicates non-present data, the CPU raises an interrupt known as a page fault.",
            "start_ms": 81000,
            "end_ms": 87900,
            "pause_before_ms": 600.0,
            "pitch_reset": 1.1,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
        {
            "sentence_id": 13,
            "text": "In two-level paging, address translation splits the virtual address into directory index, page index, and offset.",
            "start_ms": 88500,
            "end_ms": 96200,
            "pause_before_ms": 600.0,
            "pitch_reset": -0.7,
            "speaker_tag": "Lecturer",
            "is_digression": False,
            "digression_type": "none",
        },
    ]

    segments_data = [
        {
            "segment_id": 0,
            "start_sentence_idx": 0,
            "end_sentence_idx": 4,
            "start_time_str": "00:00:01",
            "end_time_str": "00:00:31",
            "topic_title": "Process Abstraction and Execution Context Overhead",
            "prerequisite_tags": ["CPU Architecture", "Assembly Registers"],
            "cue_questions": [
                "What constituent segments compose a process address space?",
                "How is total context switch overhead modeled mathematically?",
            ],
            "sentences": sentences_data[0:5],
            "raw_text": " ".join(s["text"] for s in sentences_data[0:5]),
        },
        {
            "segment_id": 1,
            "start_sentence_idx": 5,
            "end_sentence_idx": 8,
            "start_time_str": "00:00:33",
            "end_time_str": "00:00:57",
            "topic_title": "Administrative Announcements: Assignment 2 & Midterm Logistics",
            "prerequisite_tags": [],
            "cue_questions": [
                "What is the deadline for assignment two submission?",
            ],
            "sentences": sentences_data[5:9],
            "raw_text": " ".join(s["text"] for s in sentences_data[5:9]),
        },
        {
            "segment_id": 2,
            "start_sentence_idx": 9,
            "end_sentence_idx": 13,
            "start_time_str": "00:01:00",
            "end_time_str": "00:01:36",
            "topic_title": "Virtual Memory Architecture, MMU Translation and Page Faults",
            "prerequisite_tags": ["Process Abstraction", "Physical RAM Hierarchy"],
            "cue_questions": [
                "How does the MMU map virtual page numbers to physical frame numbers?",
                "What sequence of events occurs when the CPU encounters a page fault?",
            ],
            "sentences": sentences_data[9:14],
            "raw_text": " ".join(s["text"] for s in sentences_data[9:14]),
        },
    ]

    return {
        "metadata": {
            "course": "Operating Systems",
            "lecture_title": "Processes, Context Switching and Virtual Memory",
            "duration_ms": 96200,
            "num_sentences": len(sentences_data),
            "num_segments": len(segments_data),
        },
        "sentences": sentences_data,
        "segments": segments_data,
    }


# ==============================================================================
# Interface Contract Validators
# ==============================================================================

def validate_contract1_sentence(s: Dict[str, Any]) -> None:
    """Validate that a sentence dictionary adheres strictly to Interface Contract 1."""
    required_keys = {
        "sentence_id",
        "text",
        "start_ms",
        "end_ms",
        "pause_before_ms",
        "pitch_reset",
        "speaker_tag",
        "is_digression",
        "digression_type",
    }
    missing = required_keys - set(s.keys())
    if missing:
        raise AssertionError(f"Contract 1 Violation: Missing keys {missing} in sentence {s}")

    assert isinstance(s["sentence_id"], int), f"sentence_id must be int, got {type(s['sentence_id'])}"
    assert isinstance(s["text"], str), f"text must be str, got {type(s['text'])}"
    assert isinstance(s["start_ms"], int), f"start_ms must be int, got {type(s['start_ms'])}"
    assert isinstance(s["end_ms"], int), f"end_ms must be int, got {type(s['end_ms'])}"
    assert isinstance(s["pause_before_ms"], (int, float)), f"pause_before_ms must be float, got {type(s['pause_before_ms'])}"
    assert isinstance(s["pitch_reset"], (int, float)), f"pitch_reset must be float, got {type(s['pitch_reset'])}"
    assert isinstance(s["speaker_tag"], str), f"speaker_tag must be str, got {type(s['speaker_tag'])}"
    assert isinstance(s["is_digression"], bool), f"is_digression must be bool, got {type(s['is_digression'])}"
    assert isinstance(s["digression_type"], str), f"digression_type must be str, got {type(s['digression_type'])}"

    assert s["start_ms"] >= 0, f"start_ms must be non-negative: {s['start_ms']}"
    assert s["end_ms"] >= s["start_ms"], f"end_ms ({s['end_ms']}) < start_ms ({s['start_ms']})"
    assert s["pause_before_ms"] >= 0.0, f"pause_before_ms must be non-negative: {s['pause_before_ms']}"

    if s["is_digression"]:
        assert s["digression_type"] != "none", "If is_digression is True, digression_type cannot be 'none'"


def validate_contract2_payload(payload: Dict[str, Any]) -> None:
    """Validate that a segmentation payload conforms strictly to Interface Contract 2."""
    assert "num_sentences" in payload, "Missing 'num_sentences' in Contract 2 payload"
    assert "topic_segments" in payload, "Missing 'topic_segments' in Contract 2 payload"

    num_sentences = payload["num_sentences"]
    segments = payload["topic_segments"]
    assert isinstance(num_sentences, int), f"num_sentences must be int, got {type(num_sentences)}"
    assert isinstance(segments, list), f"topic_segments must be list, got {type(segments)}"

    if num_sentences == 0:
        assert len(segments) == 0, "Empty transcript must have 0 topic segments"
        return

    time_regex = re.compile(r"^\d{2}:\d{2}:\d{2}$")
    last_end_idx = -1

    for seg_idx, seg in enumerate(segments):
        required_keys = {
            "segment_id",
            "start_sentence_idx",
            "end_sentence_idx",
            "start_time_str",
            "end_time_str",
            "topic_title",
            "prerequisite_tags",
            "cue_questions",
            "sentences",
            "raw_text",
        }
        missing = required_keys - set(seg.keys())
        if missing:
            raise AssertionError(f"Contract 2 Segment #{seg_idx} missing keys {missing}")

        assert seg["segment_id"] == seg_idx, f"segment_id {seg['segment_id']} != expected {seg_idx}"
        assert seg["start_sentence_idx"] == last_end_idx + 1, (
            f"Segment {seg_idx} start_sentence_idx ({seg['start_sentence_idx']}) "
            f"not contiguous with previous end ({last_end_idx})"
        )
        assert seg["end_sentence_idx"] >= seg["start_sentence_idx"], "end_sentence_idx < start_sentence_idx"
        last_end_idx = seg["end_sentence_idx"]

        assert time_regex.match(seg["start_time_str"]), f"Invalid start_time_str format: {seg['start_time_str']}"
        assert time_regex.match(seg["end_time_str"]), f"Invalid end_time_str format: {seg['end_time_str']}"
        assert isinstance(seg["topic_title"], str) and len(seg["topic_title"].strip()) > 0, "topic_title must be non-empty"
        assert isinstance(seg["prerequisite_tags"], list), "prerequisite_tags must be list"
        assert isinstance(seg["cue_questions"], list), "cue_questions must be list"
        assert isinstance(seg["sentences"], list), "sentences must be list"
        assert isinstance(seg["raw_text"], str), "raw_text must be str"

        # Check constituent sentences adhere to Contract 1
        for s in seg["sentences"]:
            validate_contract1_sentence(s)

    assert last_end_idx == num_sentences - 1, (
        f"Last segment end_sentence_idx ({last_end_idx}) != total num_sentences - 1 ({num_sentences - 1})"
    )


def get_contract1_sample_sentences() -> List[Dict[str, Any]]:
    """Return a valid sample list of Contract 1 sentence dictionaries."""
    lecture = get_synthetic_10min_lecture_fixture()
    return lecture["sentences"]


def get_contract2_sample_payload() -> Dict[str, Any]:
    """Return a valid sample Contract 2 payload dictionary."""
    lecture = get_synthetic_10min_lecture_fixture()
    return {
        "num_sentences": lecture["metadata"]["num_sentences"],
        "topic_segments": lecture["segments"],
    }
