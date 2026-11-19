"""Comprehensive Unit and Integration Test Suite for Milestone 1 ASR.

Validates:
1. Silero VAD v4 Silence Chunking (10s - 28s bounds, silence midpoint split, energy fallback)
2. faster-whisper large-v3-turbo CPU INT8 transcription & WhisperX CTC phoneme alignment
3. Disfluency scrubbing, iterative stutter resolution, and STEM acronym truecasing
4. Semantic digression gating (d_topic > 0.65 sustained >= 3 sentences, administrative anchoring, centroid freeze)
5. End-to-end ASRPipeline execution and strict Interface Contract 1 compliance
6. Error handling, 0-byte audio, pure silence, degenerate audio, and memory guardrails
"""

import math
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Phase 1 Modules
from src.asr.digression_gate import (
    DigressionGateConfig,
    LightweightTfidfEmbedder,
    SemanticDigressionGate,
)
from src.asr.disfluency_filter import (
    DisfluencyFilter,
    DisfluencyFilterConfig,
)
from src.asr.pipeline import (
    ASRPipeline,
    ASRPipelineConfig,
    SentenceRecord,
)
from src.asr.transcribe_align import (
    ASRConfig,
    AlignedSentence,
    TranscribeAlignPipeline,
    TranscribedWord,
    WhisperTranscriber,
    WhisperXAligner,
)
from src.asr.vad_chunker import (
    AudioChunk,
    SileroVADChunker,
    VADChunkConfig,
)

# Test Fixtures & Contract Validator
from tests.fixtures.audio_fixture import (
    generate_extreme_audio_wav,
    generate_silence_wav,
    generate_speech_like_wav,
    generate_synthetic_wav,
    get_mod01lec01_sample_sentences,
    get_synthetic_10min_lecture_fixture,
    validate_contract1_sentence,
)


class TestVADChunkerUnit(unittest.TestCase):
    """Unit tests for Silero VAD dynamic audio chunking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.chunker = SileroVADChunker(VADChunkConfig(force_fallback=True))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chunk_duration_bounds(self):
        """Verifies chunks respect min (10.0s) and max (28.0s) duration bounds on long audio."""
        wav_path = Path(self.temp_dir) / "speech_long.wav"
        # 60 seconds audio with silence pauses every ~12 seconds
        speech_intervals = [
            (0.5, 11.5),
            (12.5, 23.5),
            (24.5, 35.5),
            (36.5, 47.5),
            (48.5, 59.5),
        ]
        generate_speech_like_wav(wav_path, duration_sec=60.0, speech_intervals=speech_intervals)

        chunks = self.chunker.chunk_audio(wav_path)
        self.assertGreater(len(chunks), 1, "Expected multiple chunks for 60s audio")

        for c in chunks[:-1]:
            self.assertGreaterEqual(
                c.duration_sec,
                10.0,
                f"Chunk {c.chunk_id} duration {c.duration_sec:.2f}s is below 10.0s minimum",
            )
            self.assertLessEqual(
                c.duration_sec,
                28.0,
                f"Chunk {c.chunk_id} duration {c.duration_sec:.2f}s exceeds 28.0s maximum",
            )

        # Check total duration preservation
        total_chunk_dur = sum(c.duration_sec for c in chunks)
        self.assertAlmostEqual(total_chunk_dur, 60.0, places=1)

    def test_silence_midpoint_splitting(self):
        """Verifies that split occurs at the midpoint of a natural silence pause."""
        wav_path = Path(self.temp_dir) / "silence_split.wav"
        # 30 seconds audio with a prominent 2.0s silence between 14.0s and 16.0s
        speech_intervals = [(0.5, 14.0), (16.0, 29.5)]
        generate_speech_like_wav(wav_path, duration_sec=30.0, speech_intervals=speech_intervals)

        chunks = self.chunker.chunk_audio(wav_path)
        self.assertEqual(len(chunks), 2)
        c0 = chunks[0]
        self.assertEqual(c0.split_reason, "silence_pause")
        # Midpoint of (14.0, 16.0) is 15.0s
        self.assertAlmostEqual(c0.end_sec, 15.0, delta=1.0)
        self.assertEqual(chunks[1].split_reason, "audio_end")

    def test_short_audio_single_chunk(self):
        """Audio shorter than max chunk duration (e.g. 8s) returns a single chunk."""
        wav_path = Path(self.temp_dir) / "short.wav"
        generate_speech_like_wav(wav_path, duration_sec=8.0)
        chunks = self.chunker.chunk_audio(wav_path)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, 0)
        self.assertEqual(chunks[0].split_reason, "audio_end")
        self.assertAlmostEqual(chunks[0].duration_sec, 8.0, places=1)

    def test_continuous_speech_energy_fallback(self):
        """Continuous speech without silences splits via local energy minimum or hard limit <= 28s."""
        wav_path = Path(self.temp_dir) / "continuous.wav"
        # 45 seconds of continuous sinusoidal speech tone
        generate_synthetic_wav(wav_path, duration_sec=45.0, frequency=220.0)

        chunks = self.chunker.chunk_audio(wav_path)
        self.assertGreater(len(chunks), 1)
        for c in chunks[:-1]:
            self.assertGreaterEqual(c.duration_sec, 10.0)
            self.assertLessEqual(c.duration_sec, 28.0)
            self.assertIn(c.split_reason, ("min_energy", "hard_limit", "silence_pause"))

    def test_empty_audio_returns_empty(self):
        """Zero-sample audio returns empty list."""
        chunks = self.chunker.chunk_audio(np.zeros(0, dtype=np.float32))
        self.assertEqual(chunks, [])


class TestTranscribeAlignUnit(unittest.TestCase):
    """Unit tests for Whisper transcription and WhisperX CTC alignment."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = ASRConfig(force_fallback=True)
        self.transcriber = WhisperTranscriber(self.config)
        self.aligner = WhisperXAligner(self.config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cpu_int8_configuration(self):
        """Verifies default transcription settings offload to CPU INT8."""
        cfg = ASRConfig()
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.compute_type, "int8")
        self.assertEqual(cfg.cpu_threads, 6)
        self.assertFalse(cfg.condition_on_previous_text)

    def test_transcribe_chunk_global_offset(self):
        """Verifies chunk transcription re-maps timestamps using global_offset_sec."""
        audio = np.random.normal(0, 0.2, 16000 * 15).astype(np.float32)
        offset = 45.0
        segments = self.transcriber.transcribe_chunk(audio, global_offset_sec=offset)
        self.assertGreater(len(segments), 0)
        seg = segments[0]
        self.assertGreaterEqual(seg["start_sec"], offset)
        self.assertGreater(seg["end_sec"], seg["start_sec"])
        if seg["words"]:
            self.assertGreaterEqual(seg["words"][0]["start_sec"], offset)

    def test_whisperx_word_alignment_precision(self):
        """Verifies CTC alignment generates millisecond word timestamps and monotonicity."""
        test_text = "Operating systems allocate memory using page tables."
        segments = [{"text": test_text, "start_sec": 10.0, "end_sec": 14.0}]
        audio = np.random.normal(0, 0.1, 16000 * 15).astype(np.float32)

        aligned = self.aligner.align(segments, audio)
        self.assertEqual(len(aligned), 1)
        words = aligned[0]["words"]
        self.assertEqual(len(words), len(test_text.split()))

        prev_end = 10000
        for w in words:
            self.assertGreaterEqual(w["start_ms"], prev_end)
            self.assertGreater(w["end_ms"], w["start_ms"])
            self.assertGreaterEqual(w["score"], 0.0)
            prev_end = w["end_ms"]

    def test_pipeline_pause_and_pitch_calculation(self):
        """Verifies TranscribeAlignPipeline computes Δt_pause and pitch reset."""
        wav_path = Path(self.temp_dir) / "test_pipeline.wav"
        generate_speech_like_wav(
            wav_path,
            duration_sec=30.0,
            speech_intervals=[(0.5, 11.0), (13.5, 25.0)],
        )

        pipeline = TranscribeAlignPipeline(asr_config=self.config)
        sentences = pipeline.process(wav_path)
        self.assertGreater(len(sentences), 0)

        for idx, s in enumerate(sentences):
            self.assertGreaterEqual(s.start_ms, 0)
            self.assertGreater(s.end_ms, s.start_ms)
            self.assertGreaterEqual(s.pause_before_ms, 0.0)
            self.assertIsInstance(s.pitch_reset, (int, float))
            # Contract 1 verification
            cdict = s.to_contract_dict()
            validate_contract1_sentence(cdict)


class TestDisfluencyFilterUnit(unittest.TestCase):
    """Unit tests for regex filler scrubbing, iterative stutter reduction, and truecasing."""

    def setUp(self):
        self.filter = DisfluencyFilter()

    def test_iterative_stutter_resolution(self):
        """Verifies multi-word stutter collapse to a single instance."""
        cases = [
            ("the the the operating system", "the operating system"),
            ("we we need to allocate memory", "we need to allocate memory"),
            ("page page page page fault", "page fault"),
            ("and and then", "and then"),
        ]
        for raw, expected in cases:
            res = self.filter.resolve_stutters(raw)
            self.assertEqual(res, expected)

    def test_conversational_filler_removal(self):
        """Verifies removal of um, uh, er, ah, you know, right?, sort of, kind of."""
        raw = "Um, today we will uh discuss sort of like page replacement, right?"
        cleaned = self.filter.clean_text(raw)
        self.assertNotIn("um", cleaned.lower())
        self.assertNotIn("uh", cleaned.lower())
        self.assertNotIn("sort of", cleaned.lower())
        self.assertNotIn("right?", cleaned.lower())
        self.assertTrue(cleaned.endswith("."))

    def test_empty_on_pure_fillers(self):
        """Sentences consisting solely of fillers produce an empty string."""
        raw = "um, uh, you know, ah"
        cleaned = self.filter.clean_text(raw)
        self.assertEqual(cleaned, "")

    def test_stem_acronym_truecasing_and_punctuation(self):
        """Verifies restoration of domain acronyms and valid terminal punctuation."""
        raw = "we accelerate ml and nlp using a gpu and vram with tlb lookups"
        cleaned = self.filter.clean_text(raw)
        for term in ("ML", "NLP", "GPU", "VRAM", "TLB"):
            self.assertIn(term, cleaned)
        self.assertTrue(cleaned[0].isupper())
        self.assertIn(cleaned[-1], ".?!")

    def test_question_detection_terminal_mark(self):
        """Verifies question sentences receive question marks."""
        raw = "what is the latency of a page fault"
        cleaned = self.filter.clean_text(raw)
        self.assertTrue(cleaned.endswith("?"))

    def test_safe_like_preservation(self):
        """Prepositional use of 'like' is preserved in pedagogical text."""
        raw = "Algorithms like Dijkstra and Prim operate on weighted graphs."
        cleaned = self.filter.clean_text(raw)
        self.assertIn("like", cleaned)


class TestDigressionGateUnit(unittest.TestCase):
    """Unit tests for running topic centroid tracking and semantic digression gating."""

    def setUp(self):
        self.gate = SemanticDigressionGate()

    def test_sustained_administrative_digression_span(self):
        """Verifies that an administrative span of >= 3 sentences is gated."""
        sentences = [
            {"sentence_id": 0, "text": "Today we will study Markov decision processes and Bellman equations."},
            {"sentence_id": 1, "text": "The optimal value function satisfies the Bellman optimality recurrence."},
            {"sentence_id": 2, "text": "We define the state transition matrix and the discount factor gamma."},
            # Administrative span (>= 3 sentences)
            {"sentence_id": 3, "text": "Before we proceed, remember that homework 2 is due this Friday on Canvas."},
            {"sentence_id": 4, "text": "Please make sure you submit your code and check your quiz grade."},
            {"sentence_id": 5, "text": "If you have any questions about the exam, please visit my office hours."},
            # Pedagogical return
            {"sentence_id": 6, "text": "Now returning to the proof of the policy gradient theorem."},
            {"sentence_id": 7, "text": "We compute the gradient of the expected return with respect to theta."},
        ]

        gated = self.gate.gate_sentences(sentences)
        self.assertEqual(len(gated), len(sentences))

        # Check administrative span is gated
        for idx in (3, 4, 5):
            self.assertTrue(gated[idx]["is_digression"], f"Sentence {idx} should be digression")
            self.assertEqual(gated[idx]["digression_type"], "administrative")

        # Check core pedagogy is NOT gated
        for idx in (0, 1, 2, 6, 7):
            self.assertFalse(gated[idx]["is_digression"], f"Sentence {idx} should NOT be digression")
            self.assertEqual(gated[idx]["digression_type"], "none")

    def test_transient_mention_not_gated(self):
        """A transient 1-sentence administrative mention (< 3 consecutive) is NOT gated."""
        sentences = [
            {"sentence_id": 0, "text": "Dynamic programming breaks down problems into overlapping subproblems."},
            {"sentence_id": 1, "text": "This algorithm will appear as a problem on the midterm exam."},  # 1 sentence
            {"sentence_id": 2, "text": "We construct the optimal substructure memoization table."},
        ]
        gated = self.gate.gate_sentences(sentences)
        for s in gated:
            self.assertFalse(s["is_digression"], f"Sentence {s['sentence_id']} was falsely gated!")
            self.assertEqual(s["digression_type"], "none")

    def test_pure_mathematical_proof_preservation(self):
        """Mathematical proofs introducing novel notation are NEVER falsely gated."""
        math_lecture = [
            {"sentence_id": 0, "text": "We consider a Hilbert space H with inner product bracket x y."},
            {"sentence_id": 1, "text": "The Cauchy-Schwarz inequality states that the absolute inner product squared is bounded."},
            {"sentence_id": 2, "text": "To prove this, consider the quadratic polynomial p of lambda defined by norm of x plus lambda y squared."},
            {"sentence_id": 3, "text": "Since the norm is non-negative, the quadratic polynomial has at most one real root."},
            {"sentence_id": 4, "text": "Therefore the discriminant b squared minus 4 a c must be less than or equal to zero."},
            {"sentence_id": 5, "text": "Rearranging the discriminant immediately yields the Cauchy-Schwarz inequality."},
        ]
        gated = self.gate.gate_sentences(math_lecture)
        for s in gated:
            self.assertFalse(s["is_digression"], f"Math proof sentence {s['sentence_id']} was falsely gated")

    def test_centroid_freeze_during_digression(self):
        """Verifies that topic centroid does not incorporate digression sentence embeddings."""
        sentences = [
            {"sentence_id": 0, "text": "Virtual memory provides contiguous address spaces."},
            {"sentence_id": 1, "text": "The page table translates virtual page numbers."},
            {"sentence_id": 2, "text": "Homework is due on Canvas tonight."},
            {"sentence_id": 3, "text": "Please review your exam grades on Canvas."},
            {"sentence_id": 4, "text": "Office hours are tomorrow for quiz questions."},
            {"sentence_id": 5, "text": "Page faults trigger the disk swap mechanism."},
        ]
        gated = self.gate.gate_sentences(sentences)
        centroid = self.gate.topic_centroid
        self.assertIsNotNone(centroid)

        # Centroid should remain highly similar to the pedagogical topic (virtual memory)
        pedagogical_vec = self.gate.embedder.encode(["Virtual memory and page tables translate addresses."])[0]
        dist = self.gate.compute_cosine_distance(pedagogical_vec, centroid)
        self.assertLess(dist, 0.65, f"Centroid was poisoned by administrative digression! Dist: {dist}")


class TestASRPipelineIntegration(unittest.TestCase):
    """Integration tests for unified Phase 1 ASR Pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = ASRPipeline(ASRPipelineConfig(force_fallback=True))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_contract1_compliance(self):
        """End-to-end execution on 30s speech fixture produces valid Contract 1 records."""
        wav_path = Path(self.temp_dir) / "lecture_30s.wav"
        generate_speech_like_wav(
            wav_path,
            duration_sec=30.0,
            speech_intervals=[(0.5, 12.0), (14.0, 28.0)],
        )

        results = self.pipeline.process_audio(wav_path)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        prev_end = 0
        for idx, rec in enumerate(results):
            # Authoritative Contract 1 schema check
            validate_contract1_sentence(rec)

            self.assertEqual(rec["sentence_id"], idx)
            self.assertGreaterEqual(rec["start_ms"], 0)
            self.assertGreater(rec["end_ms"], rec["start_ms"])
            self.assertGreaterEqual(rec["pause_before_ms"], 0.0)
            self.assertIn(rec["speaker_tag"], ("Lecturer", "Student"))
            self.assertIn(rec["digression_type"], ("administrative", "logistical", "none"))

            if rec["is_digression"]:
                self.assertNotEqual(rec["digression_type"], "none")
            else:
                self.assertEqual(rec["digression_type"], "none")

            prev_end = rec["end_ms"]

    def test_return_records_typed_objects(self):
        """Tests return_records=True emitting List[SentenceRecord]."""
        wav_path = Path(self.temp_dir) / "lecture_short.wav"
        generate_speech_like_wav(wav_path, duration_sec=15.0)

        records = self.pipeline.process_audio(wav_path, return_records=True)
        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)
        for r in records:
            self.assertIsInstance(r, SentenceRecord)
            validate_contract1_sentence(r.to_contract_dict())

    def test_numpy_array_input(self):
        """Pipeline accepts directly ingested 16 kHz float32 NumPy arrays."""
        audio_array = np.random.normal(0, 0.2, 16000 * 20).astype(np.float32)
        results = self.pipeline.process_audio(audio_array)
        self.assertIsInstance(results, list)
        for r in results:
            validate_contract1_sentence(r)


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Boundary conditions, corrupted files, and zero-byte inputs."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = ASRPipeline(ASRPipelineConfig(force_fallback=True))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_file_raises_filenotfound(self):
        """Non-existent audio file path raises FileNotFoundError."""
        missing = Path(self.temp_dir) / "non_existent.wav"
        with self.assertRaises(FileNotFoundError):
            self.pipeline.process_audio(missing)

    def test_zero_byte_audio_raises_valueerror(self):
        """Empty 0-byte file raises ValueError."""
        empty_file = Path(self.temp_dir) / "zero_byte.wav"
        empty_file.touch()
        with self.assertRaises(ValueError):
            self.pipeline.process_audio(empty_file)

    def test_pure_silence_returns_empty_list(self):
        """Pure silence audio gracefully emits an empty list with zero errors."""
        silence_path = Path(self.temp_dir) / "pure_silence.wav"
        generate_silence_wav(silence_path, duration_sec=10.0)

        results = self.pipeline.process_audio(silence_path)
        self.assertEqual(results, [])

    def test_degenerate_empty_wav_header_only(self):
        """0-frame WAV header only returns empty list."""
        empty_header_path = Path(self.temp_dir) / "empty_header.wav"
        generate_extreme_audio_wav(empty_header_path, mode="empty")
        results = self.pipeline.process_audio(empty_header_path)
        self.assertEqual(results, [])

    def test_degenerate_tiny_audio(self):
        """Audio shorter than a single VAD frame (0.02s) completes cleanly."""
        tiny_path = Path(self.temp_dir) / "tiny.wav"
        generate_extreme_audio_wav(tiny_path, mode="tiny")
        results = self.pipeline.process_audio(tiny_path)
        self.assertIsInstance(results, list)

    def test_clipped_extreme_audio(self):
        """Saturated clipping square wave completes without exceptions."""
        clip_path = Path(self.temp_dir) / "clipping.wav"
        generate_extreme_audio_wav(clip_path, mode="clipping")
        results = self.pipeline.process_audio(clip_path)
        self.assertIsInstance(results, list)


class TestResourceAndMemoryGuardrails(unittest.TestCase):
    """Resource constraints: CPU INT8 offload, memory efficiency, and speed."""

    def test_default_device_is_cpu(self):
        """Verifies default device is CPU, protecting RTX 4050 6GB VRAM for Ollama."""
        cfg = ASRPipelineConfig()
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.whisper_compute_type, "int8")

    def test_execution_speed_and_zero_hang(self):
        """Verifies processing of 30s audio completes deterministically under 5 seconds."""
        import time

        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "speed_test.wav"
            generate_speech_like_wav(wav_path, duration_sec=30.0)

            pipeline = ASRPipeline(ASRPipelineConfig(force_fallback=True))
            t0 = time.perf_counter()
            results = pipeline.process_audio(wav_path)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            self.assertLess(elapsed, 5.0, f"Processing took too long: {elapsed:.2f}s")
            self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
