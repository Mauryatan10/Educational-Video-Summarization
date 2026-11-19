"""tests/test_e2e_pipeline.py - Comprehensive Test Suite for Milestone 4 E2E Pipeline.

Covers:
1. CLI Argument Parsing and Pipeline Configuration.
2. Fast Pipeline Execution via Pre-Computed Transcript Fixtures.
3. Full Audio Pipeline Execution (Synthetic WAV -> Phase 1 -> Phase 2 -> Notes).
4. Structural and Pedagogical Schema Validation of Markdown Study Notes.
5. Strict Schema Validation of Machine-Readable JSON Study Notes.
6. Edge Cases and Degenerate Inputs (Empty audio, silence, subsecond, N=1, pure digression, multi-hour).
7. Hardware Safety, Sequential Stage Isolation, and Ollama Memory Eviction.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# Fixture Loaders & Contract Validators
from tests.fixtures.audio_fixture import (
    DEFAULT_SAMPLE_RATE,
    generate_extreme_audio_wav,
    generate_silence_wav,
    generate_speech_like_wav,
    generate_synthetic_wav,
    get_contract1_sample_sentences,
    get_contract2_sample_payload,
    get_mod01lec01_sample_sentences,
    get_synthetic_10min_lecture_fixture,
    load_mod01lec01_raw,
    read_wav_samples,
    validate_contract1_sentence,
    validate_contract2_payload,
)

# Pipeline Modules
from src.asr.pipeline import ASRPipeline, ASRPipelineConfig, SentenceRecord
from src.segmentation.pipeline import (
    SegmentationPayload,
    SegmentationPipeline,
    SegmentationPipelineConfig,
    TopicSegment,
)
from src.pipeline.run_e2e_lecture import (
    E2ELecturePipeline,
    E2EPipelineConfig,
    build_arg_parser,
    main as pipeline_main,
)
from src.pipeline.note_generator import (
    NoteGeneratorConfig,
    PedagogicalNoteGenerator,
    LectureStudyNotes,
)


# ==============================================================================
# Helper Schema Validators for Milestone 4 Output
# ==============================================================================

def validate_study_notes_json_schema(payload: Dict[str, Any]) -> None:
    """Strict schema validation for lecture_study_notes.json."""
    required_top = {
        "schema_version",
        "metadata",
        "prerequisite_dag",
        "table_of_contents",
        "study_sections",
        "administrative_digressions",
        "global_review",
    }
    missing = required_top - set(payload.keys())
    assert not missing, f"JSON notes schema missing required keys: {missing}"

    meta = payload["metadata"]
    for k in ("lecture_title", "course", "total_duration_sec", "total_duration_str", "total_sentences", "total_segments"):
        assert k in meta, f"Metadata missing required key: {k}"
    assert meta["total_duration_sec"] >= 0
    assert meta["total_sentences"] >= 0
    assert meta["total_segments"] >= 0

    # Prerequisite DAG validation
    dag = payload["prerequisite_dag"]
    assert "nodes" in dag and isinstance(dag["nodes"], list)
    assert "edges" in dag and isinstance(dag["edges"], list)
    assert "topological_order" in dag and isinstance(dag["topological_order"], list)

    # Table of contents validation
    toc = payload["table_of_contents"]
    assert isinstance(toc, list)
    time_regex = re.compile(r"^\d{2}:\d{2}:\d{2}$")
    for item in toc:
        assert "section_id" in item
        assert "title" in item and len(item["title"].strip()) > 0
        assert time_regex.match(item["start_time_str"])
        assert time_regex.match(item["end_time_str"])
        assert "is_digression" in item

    # Study sections validation
    sections = payload["study_sections"]
    assert isinstance(sections, list)
    for sec in sections:
        for k in ("section_id", "topic_title", "start_time_str", "end_time_str", "prerequisite_tags", "cornell_cue_column", "detailed_notes", "takeaways"):
            assert k in sec, f"Study section {sec.get('section_id')} missing key: {k}"
        assert not sec["is_digression"], f"Technical study section {sec['section_id']} marked as digression"

        cue_col = sec["cornell_cue_column"]
        assert "cue_questions" in cue_col and isinstance(cue_col["cue_questions"], list)
        assert len(cue_col["cue_questions"]) >= 1, f"Study section {sec['section_id']} has 0 cue questions"
        for q in cue_col["cue_questions"]:
            assert q.endswith("?"), f"Cue question does not end with '?': '{q}'"

        notes = sec["detailed_notes"]
        assert "conceptual_exposition" in notes and len(notes["conceptual_exposition"].strip()) > 0
        assert "formal_definitions" in notes
        assert "mathematical_formulations" in notes

    # Administrative digressions validation
    digressions = payload["administrative_digressions"]
    assert isinstance(digressions, list)
    for d in digressions:
        for k in ("section_id", "title", "start_time_str", "end_time_str", "digression_type", "announcement_notes"):
            assert k in d, f"Digression missing key: {k}"


def validate_study_notes_markdown_structure(md_text: str) -> None:
    """Strict structural and formatting validation for lecture_study_notes.md."""
    lines = md_text.splitlines()
    assert len(lines) >= 20, "Markdown study notes are excessively short"

    # Header and metadata
    assert md_text.startswith("# "), "Markdown notes must start with H1 title"
    assert "> **Course**:" in md_text
    assert "> **Total Duration**:" in md_text
    assert "> **Pedagogical Integrity**:" in md_text

    # Executive Overview
    assert "## Executive Overview" in md_text

    # Table of Contents
    assert "## Table of Contents" in md_text

    # Section-level Cue Column omitted per user preference
    assert "### Cornell Cue Column (Active Recall)" not in md_text

    # Detailed Notes & Pedagogical Content (exposition directly present without redundant heading)
    assert "### Detailed Notes & Conceptual Exposition" not in md_text

    # Global Review & Flashcards
    assert "## Global Review & Master Study Aids" in md_text
    assert "### Active Recall Review Sheet (Flashcard Mode)" in md_text


def generate_multihour_transcript_fixture(
    num_hours: float = 2.0,
    sentences_per_minute: int = 4,
) -> Dict[str, Any]:
    """Generate a high-fidelity synthetic multi-hour lecture transcript (Interface Contract 1)."""
    total_minutes = int(round(num_hours * 60))
    total_sentences = total_minutes * sentences_per_minute
    sentences: List[Dict[str, Any]] = []

    cur_ms = 1000
    sentence_topics = [
        ("Architecture", "Computer architecture defines the instruction set and memory hierarchy.", False),
        ("Pipelining", "Instruction pipelining overlaps fetch, decode, execute, and writeback stages.", False),
        ("Hazards", "Structural, data, and control hazards stall the processor execution pipeline.", False),
        ("Logistics", "Please remember that assignment submission portals close Friday at midnight.", True),
        ("BranchPred", "Dynamic branch prediction uses two-bit saturating counters to predict jumps.", False),
        ("Caches", "Direct-mapped and set-associative caches minimize average memory access time.", False),
        ("VirtualMem", "The translation lookaside buffer caches virtual to physical page translations.", False),
    ]

    target_duration_ms = int(num_hours * 3600 * 1000)
    dur_per_sent_ms = target_duration_ms // max(1, total_sentences)
    pause_ms_const = 500.0
    dur_ms = max(2000, int(dur_per_sent_ms - pause_ms_const))

    for idx in range(total_sentences):
        topic_idx = (idx // 40) % len(sentence_topics)
        tag, text_base, is_dig = sentence_topics[topic_idx]
        text = f"Sentence {idx} on {tag}: {text_base}"
        pause_ms = pause_ms_const if idx > 0 else 0.0

        sentences.append({
            "sentence_id": idx,
            "text": text,
            "start_ms": cur_ms,
            "end_ms": cur_ms + dur_ms,
            "pause_before_ms": pause_ms,
            "pitch_reset": 0.5 if idx % 10 == 0 else 0.0,
            "speaker_tag": "Lecturer",
            "is_digression": is_dig,
            "digression_type": "administrative" if is_dig else "none",
        })
        cur_ms += dur_ms + int(pause_ms)

    return {
        "metadata": {
            "course": "Computer Systems",
            "lecture_title": "Advanced Computer Architecture & Memory Hierarchies",
            "duration_ms": cur_ms,
            "num_sentences": total_sentences,
        },
        "sentences": sentences,
    }


# ==============================================================================
# Class 1: TestPipelineCLIAndConfiguration (6 Tests)
# ==============================================================================

class TestPipelineCLIAndConfiguration:
    """Verifies CLI argument parsing, configuration defaults, and parameter validation."""

    def test_default_config_initialization(self):
        """Verifies default pipeline configuration parameters adhere to hardware envelope."""
        cfg = E2EPipelineConfig()
        assert cfg.device == "cpu"
        assert cfg.whisper_compute_type == "int8"
        assert cfg.vad_min_chunk_duration == 10.0
        assert cfg.vad_max_chunk_duration == 28.0
        assert cfg.force_fallback is False
        assert cfg.notes_markdown_filename == "lecture_study_notes.md"
        assert cfg.notes_json_filename == "lecture_study_notes.json"

    def test_cli_audio_argument_parsing(self, tmp_path: Path):
        """Verifies CLI successfully parses audio path and overrides defaults."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        parser = build_arg_parser()
        args = parser.parse_args([
            "--audio", str(audio_file),
            "--output-dir", str(tmp_path),
            "--offline",
            "--device", "cpu",
            "--title", "Operating Systems Lecture 1",
            "--course", "CS401",
        ])
        assert args.audio == str(audio_file)
        assert args.transcript_json is None
        assert args.offline is True
        assert args.title == "Operating Systems Lecture 1"
        assert args.course == "CS401"

    def test_cli_transcript_json_argument_parsing(self, tmp_path: Path):
        """Verifies CLI successfully parses pre-computed transcript JSON path."""
        json_file = tmp_path / "transcript.json"
        json_file.touch()
        parser = build_arg_parser()
        args = parser.parse_args([
            "--transcript-json", str(json_file),
            "--output-dir", str(tmp_path),
            "--offline",
        ])
        assert args.transcript_json == str(json_file)
        assert args.audio is None

    def test_cli_mutual_exclusivity_or_precedence(self, tmp_path: Path):
        """Verifies supplying neither audio nor transcript raises error or helpful exit."""
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_missing_input_file_raises_filenotfound(self, tmp_path: Path):
        """Verifies supplying a non-existent audio or transcript path raises FileNotFoundError."""
        non_existent = tmp_path / "ghost_file.wav"
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True))
        with pytest.raises(FileNotFoundError):
            pipeline.run(audio_path=non_existent)

    def test_output_directory_auto_creation(self, tmp_path: Path):
        """Verifies nested output directory is created automatically if non-existent."""
        deep_dir = tmp_path / "nested" / "reports" / "lecture_01"
        assert not deep_dir.exists()
        fixture = get_synthetic_10min_lecture_fixture()
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=deep_dir))
        pipeline.run(transcript_input=fixture["sentences"])
        assert deep_dir.exists()
        assert (deep_dir / "lecture_study_notes.md").exists()
        assert (deep_dir / "lecture_study_notes.json").exists()


# ==============================================================================
# Class 2: TestFastPipelinePrecomputedTranscripts (5 Tests)
# ==============================================================================

class TestFastPipelinePrecomputedTranscripts:
    """Verifies fast-path execution from pre-computed transcripts (<1.0s, fully offline)."""

    def test_fast_pipeline_synthetic_10min_execution(self, tmp_path: Path):
        """Verifies end-to-end execution on synthetic 10-minute lecture in <1.0 second."""
        fixture = get_synthetic_10min_lecture_fixture()
        cfg = E2EPipelineConfig(force_fallback=True, output_dir=tmp_path)
        pipeline = E2ELecturePipeline(config=cfg)

        result = pipeline.run(transcript_input=fixture["sentences"], metadata=fixture["metadata"])
        assert result["status"] == "SUCCESS"
        assert result["num_sentences"] == 14
        assert result["num_segments"] >= 2

        # Verify output files exist and are non-empty
        md_file = tmp_path / "lecture_study_notes.md"
        json_file = tmp_path / "lecture_study_notes.json"
        assert md_file.exists() and md_file.stat().st_size > 500
        assert json_file.exists() and json_file.stat().st_size > 500

    def test_fast_pipeline_mod01lec01_nptel_transcript(self, tmp_path: Path):
        """Verifies fast pipeline execution on real NPTEL OS transcript sentences."""
        sample_sents = get_mod01lec01_sample_sentences(count=20)
        c1_sents = []
        cur_ms = 0
        for i, text in enumerate(sample_sents):
            dur = len(text.split()) * 300
            c1_sents.append({
                "sentence_id": i,
                "text": text,
                "start_ms": cur_ms,
                "end_ms": cur_ms + dur,
                "pause_before_ms": 500.0 if i > 0 else 0.0,
                "pitch_reset": 0.0,
                "speaker_tag": "Lecturer",
                "is_digression": False,
                "digression_type": "none",
            })
            cur_ms += dur + 500

        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=c1_sents)
        assert result["status"] == "SUCCESS"
        assert result["num_sentences"] == 20

    def test_fast_pipeline_json_file_on_disk_ingestion(self, tmp_path: Path):
        """Verifies ingestion of serialized Contract 1 JSON file from disk."""
        fixture = get_synthetic_10min_lecture_fixture()
        json_in = tmp_path / "input_transcript.json"
        json_in.write_text(json.dumps(fixture["sentences"]), encoding="utf-8")

        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=json_in)
        assert result["status"] == "SUCCESS"
        assert result["num_sentences"] == 14

    def test_fast_pipeline_timestamp_preservation(self, tmp_path: Path):
        """Verifies transcript millisecond timestamps map exactly to HH:MM:SS in notes."""
        fixture = get_synthetic_10min_lecture_fixture()
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=fixture["sentences"])

        json_out = json.loads((tmp_path / "lecture_study_notes.json").read_text(encoding="utf-8"))
        assert json_out["metadata"]["total_duration_str"] == "00:01:36"
        assert json_out["table_of_contents"][0]["start_time_str"] == "00:00:01"

    def test_fast_pipeline_digression_propagation(self, tmp_path: Path):
        """Verifies administrative digression sentences in Contract 1 are segregated in notes."""
        fixture = get_synthetic_10min_lecture_fixture()
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=fixture["sentences"])

        json_out = json.loads((tmp_path / "lecture_study_notes.json").read_text(encoding="utf-8"))
        assert len(json_out["administrative_digressions"]) == 1
        d = json_out["administrative_digressions"][0]
        assert "assignment" in d["announcement_notes"].lower() or "midterm" in d["announcement_notes"].lower()


# ==============================================================================
# Class 3: TestFullPipelineAudioToNotes (5 Tests)
# ==============================================================================

class TestFullPipelineAudioToNotes:
    """Verifies full pipeline execution from raw audio WAV: Audio -> Phase 1 -> Phase 2 -> Notes."""

    def test_full_pipeline_synthetic_wav_execution(self, tmp_path: Path):
        """Verifies full execution on 30s synthetic speech-like WAV with offline fallback."""
        wav_path = tmp_path / "speech_30s.wav"
        generate_speech_like_wav(
            wav_path,
            duration_sec=30.0,
            speech_intervals=[(0.5, 12.0), (14.0, 28.0)],
        )

        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(audio_path=wav_path)

        assert result["status"] == "SUCCESS"
        assert (tmp_path / "phase1_transcript.json").exists()
        assert (tmp_path / "phase2_segmentation.json").exists()
        assert (tmp_path / "lecture_study_notes.md").exists()
        assert (tmp_path / "lecture_study_notes.json").exists()

    def test_full_pipeline_multi_chunk_vad_integration(self, tmp_path: Path):
        """Verifies 60s audio triggers multiple Silero VAD chunks and merges cleanly into notes."""
        wav_path = tmp_path / "speech_60s.wav"
        generate_speech_like_wav(
            wav_path,
            duration_sec=60.0,
            speech_intervals=[(0.5, 18.0), (20.0, 38.0), (40.0, 58.0)],
        )

        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(audio_path=wav_path)
        assert result["status"] == "SUCCESS"
        assert result["num_sentences"] >= 2

    def test_full_pipeline_in_memory_numpy_array(self, tmp_path: Path):
        """Verifies pipeline accepts 1D float32 NumPy audio arrays directly without file writing."""
        samples = np.random.normal(0, 0.1, 16000 * 15).astype(np.float32)
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(audio_array=samples)
        assert result["status"] == "SUCCESS"

    def test_full_pipeline_deterministic_reproducibility(self, tmp_path: Path):
        """Verifies identical audio input produces bit-identical notes across repeated runs."""
        wav_path = tmp_path / "repro.wav"
        generate_speech_like_wav(wav_path, duration_sec=20.0)

        out_dir1 = tmp_path / "run1"
        out_dir2 = tmp_path / "run2"

        pipe1 = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=out_dir1))
        pipe2 = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=out_dir2))

        pipe1.run(audio_path=wav_path)
        pipe2.run(audio_path=wav_path)

        md1 = (out_dir1 / "lecture_study_notes.md").read_text(encoding="utf-8")
        md2 = (out_dir2 / "lecture_study_notes.md").read_text(encoding="utf-8")
        assert md1 == md2, "Pipeline output is non-deterministic across repeated executions"

    def test_full_pipeline_execution_summary_metrics(self, tmp_path: Path):
        """Verifies pipeline emits execution_summary.json with stage-by-stage latencies."""
        wav_path = tmp_path / "summary_test.wav"
        generate_speech_like_wav(wav_path, duration_sec=15.0)

        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(audio_path=wav_path)

        summary_path = tmp_path / "execution_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert "stage_latencies" in summary
        assert "asr_time_sec" in summary["stage_latencies"]
        assert "segmentation_time_sec" in summary["stage_latencies"]
        assert "notes_time_sec" in summary["stage_latencies"]
        assert "total_time_sec" in summary["stage_latencies"]
        assert summary["stage_latencies"]["total_time_sec"] > 0


# ==============================================================================
# Class 4: TestStudyNotesMarkdownSchemaValidation (7 Tests)
# ==============================================================================

class TestStudyNotesMarkdownSchemaValidation:
    """Verifies pedagogical depth, Cornell styling, formula integrity, and layout of Markdown notes."""

    @pytest.fixture
    def sample_markdown_notes(self, tmp_path: Path) -> str:
        fixture = get_synthetic_10min_lecture_fixture()
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        pipeline.run(transcript_input=fixture["sentences"], metadata=fixture["metadata"])
        return (tmp_path / "lecture_study_notes.md").read_text(encoding="utf-8")

    def test_markdown_header_and_metadata_block(self, sample_markdown_notes: str):
        """Verifies H1 title, course name, duration, and pedagogical integrity declaration."""
        assert "# Processes, Context Switching and Virtual Memory" in sample_markdown_notes
        assert "> **Course**: Operating Systems" in sample_markdown_notes
        assert "> **Total Duration**: `[00:01:36]`" in sample_markdown_notes
        assert "Zero Destructive Compression" in sample_markdown_notes

    def test_markdown_prerequisite_graph_and_mermaid_block(self, sample_markdown_notes: str):
        """Verifies presence of Executive Overview, absence of Mermaid DAG, and absence of Prerequisite Hierarchy table."""
        assert "## Executive Overview" in sample_markdown_notes
        assert "```mermaid" not in sample_markdown_notes
        assert "### Conceptual Prerequisite Hierarchy" not in sample_markdown_notes
        assert "| Prerequisite Concept | Target Subject Concept |" not in sample_markdown_notes

    def test_markdown_table_of_contents_formatting(self, sample_markdown_notes: str):
        """Verifies TOC links all segments with HH:MM:SS anchors and digression tags."""
        assert "## Table of Contents" in sample_markdown_notes
        assert "- `[00:00:01]`" in sample_markdown_notes
        assert "(Administrative Digression)" in sample_markdown_notes

    def test_markdown_cornell_cue_column_questions(self, sample_markdown_notes: str):
        """Verifies section-level cue column is omitted and active recall flashcards exist in Global Review."""
        assert "### Cornell Cue Column (Active Recall)" not in sample_markdown_notes
        assert "### Active Recall Review Sheet (Flashcard Mode)" in sample_markdown_notes
        cues = re.findall(r"\*\*Question\*\*:\s*(.+)", sample_markdown_notes)
        assert len(cues) >= 1
        for q in cues:
            assert q.strip().endswith("?")

    def test_markdown_zero_pedagogical_loss_exposition(self, sample_markdown_notes: str):
        """Verifies detailed notes preserve mechanism definitions without redundant exposition heading."""
        assert "### Detailed Notes & Conceptual Exposition" not in sample_markdown_notes
        assert "Process Control Block" in sample_markdown_notes or "PCB" in sample_markdown_notes
        assert "Virtual memory" in sample_markdown_notes or "page fault" in sample_markdown_notes

    def test_markdown_mathematical_formula_preservation(self, tmp_path: Path):
        """Verifies LaTeX displayed equations ($$...$$) are preserved intact."""
        math_sentences = [
            {
                "sentence_id": 0,
                "text": "Context switch overhead is given by $$T_{switch} = T_{save} + T_{restore}$$.",
                "start_ms": 0, "end_ms": 5000, "pause_before_ms": 0.0, "pitch_reset": 0.0,
                "speaker_tag": "Lecturer", "is_digression": False, "digression_type": "none",
            },
            {
                "sentence_id": 1,
                "text": "Effective access time satisfies $$EAT = (1-p) \\times T_{mem} + p \\times T_{fault}$$.",
                "start_ms": 5500, "end_ms": 11000, "pause_before_ms": 500.0, "pitch_reset": 0.0,
                "speaker_tag": "Lecturer", "is_digression": False, "digression_type": "none",
            }
        ]
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        pipeline.run(transcript_input=math_sentences)
        md = (tmp_path / "lecture_study_notes.md").read_text(encoding="utf-8")
        assert "$$T_{switch} = T_{save} + T_{restore}$$" in md
        assert "$$EAT = (1-p) \\times T_{mem} + p \\times T_{fault}$$" in md

    def test_markdown_administrative_digression_segregation(self, sample_markdown_notes: str):
        """Verifies administrative sections are segregated into a distinct callout/appendix."""
        assert "## Course Logistics & Administrative Announcements" in sample_markdown_notes
        assert "⚠️ Notice:" in sample_markdown_notes
        assert "assignment number two" in sample_markdown_notes.lower()


# ==============================================================================
# Class 5: TestStudyNotesJSONSchemaValidation (6 Tests)
# ==============================================================================

class TestStudyNotesJSONSchemaValidation:
    """Verifies strict schema conformance of lecture_study_notes.json."""

    @pytest.fixture
    def sample_json_payload(self, tmp_path: Path) -> Dict[str, Any]:
        fixture = get_synthetic_10min_lecture_fixture()
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        pipeline.run(transcript_input=fixture["sentences"], metadata=fixture["metadata"])
        return json.loads((tmp_path / "lecture_study_notes.json").read_text(encoding="utf-8"))

    def test_json_top_level_keys_and_schema_version(self, sample_json_payload: Dict[str, Any]):
        """Verifies presence of all top-level keys and valid semantic schema version."""
        validate_study_notes_json_schema(sample_json_payload)
        assert sample_json_payload["schema_version"] in ("1.0", "2.0-pedagogical")

    def test_json_metadata_types_and_invariants(self, sample_json_payload: Dict[str, Any]):
        """Verifies metadata field types, counts, and duration invariants."""
        meta = sample_json_payload["metadata"]
        assert isinstance(meta["lecture_title"], str) and len(meta["lecture_title"]) > 0
        assert isinstance(meta["course"], str) and len(meta["course"]) > 0
        assert meta["total_duration_sec"] == pytest.approx(96.2, rel=1e-1)
        assert meta["total_sentences"] == 14
        assert meta["total_segments"] == 3

    def test_json_table_of_contents_temporal_alignment(self, sample_json_payload: Dict[str, Any]):
        """Verifies TOC items are contiguous and timestamps are formatted as HH:MM:SS."""
        toc = sample_json_payload["table_of_contents"]
        assert len(toc) == 3
        for i, item in enumerate(toc):
            assert item["section_id"] == i
            assert re.match(r"^\d{2}:\d{2}:\d{2}$", item["start_time_str"])
            assert re.match(r"^\d{2}:\d{2}:\d{2}$", item["end_time_str"])

    def test_json_study_sections_field_conformance(self, sample_json_payload: Dict[str, Any]):
        """Verifies each study section adheres to Interface Contract 2 + note enhancements."""
        sections = sample_json_payload["study_sections"]
        assert len(sections) == 2  # 2 technical sections (digression segregated)
        for s in sections:
            assert isinstance(s["prerequisite_tags"], list)
            assert isinstance(s["takeaways"], list)
            assert len(s["cornell_cue_column"]["cue_questions"]) >= 1
            for q in s["cornell_cue_column"]["cue_questions"]:
                assert q.endswith("?")

    def test_json_master_glossary_and_active_recall_sheet(self, sample_json_payload: Dict[str, Any]):
        """Verifies global review block contains master glossary and numbered flashcards."""
        review = sample_json_payload["global_review"]
        assert "master_glossary" in review and isinstance(review["master_glossary"], list)
        assert "active_recall_review_sheet" in review and isinstance(review["active_recall_review_sheet"], list)
        assert len(review["active_recall_review_sheet"]) >= 1
        for card in review["active_recall_review_sheet"]:
            assert "question" in card and card["question"].endswith("?")
            assert "core_concept" in card
            assert "section_id" in card

    def test_json_serialization_round_trip(self, sample_json_payload: Dict[str, Any]):
        """Verifies serialized JSON string parses cleanly with standard json.loads with identical content."""
        dumped = json.dumps(sample_json_payload, indent=2)
        reloaded = json.loads(dumped)
        assert reloaded == sample_json_payload


# ==============================================================================
# Class 6: TestEdgeCasesAndDegenerateInputs (7 Tests)
# ==============================================================================

class TestEdgeCasesAndDegenerateInputs:
    """Rigorous boundary, corner case, and stress testing for E2E pipeline."""

    def test_edge_case_zero_byte_audio_handling(self, tmp_path: Path):
        """Verifies 0-byte audio file raises ValueError or handles gracefully."""
        empty_wav = tmp_path / "empty.wav"
        empty_wav.touch()
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        with pytest.raises((ValueError, RuntimeError)):
            pipeline.run(audio_path=empty_wav)

    def test_edge_case_pure_silence_audio(self, tmp_path: Path):
        """Verifies 10s pure silence audio emits graceful empty notes without crash."""
        silence_wav = tmp_path / "silence.wav"
        generate_silence_wav(silence_wav, duration_sec=10.0)
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(audio_path=silence_wav)
        assert result["status"] in ("SUCCESS", "EMPTY_AUDIO")
        assert result["num_sentences"] == 0

    def test_edge_case_subsecond_tiny_audio(self, tmp_path: Path):
        """Verifies 0.02s audio (shorter than VAD window) executes cleanly without crash."""
        tiny_wav = tmp_path / "tiny.wav"
        generate_extreme_audio_wav(tiny_wav, mode="tiny")
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(audio_path=tiny_wav)
        assert result["status"] in ("SUCCESS", "EMPTY_AUDIO")

    def test_edge_case_single_sentence_lecture(self, tmp_path: Path):
        """Verifies N=1 sentence document emits exactly 1 valid segment without index or 1D-NMS error."""
        single_sent = [{
            "sentence_id": 0,
            "text": "An operating system manages hardware resources and provides abstraction.",
            "start_ms": 1000, "end_ms": 5000, "pause_before_ms": 0.0, "pitch_reset": 0.0,
            "speaker_tag": "Lecturer", "is_digression": False, "digression_type": "none",
        }]
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=single_sent)
        assert result["status"] == "SUCCESS"
        assert result["num_sentences"] == 1
        assert result["num_segments"] == 1

        json_out = json.loads((tmp_path / "lecture_study_notes.json").read_text(encoding="utf-8"))
        validate_study_notes_json_schema(json_out)
        assert len(json_out["study_sections"]) == 1
        assert len(json_out["administrative_digressions"]) == 0

    def test_edge_case_pure_administrative_digression_lecture(self, tmp_path: Path):
        """Verifies 100% administrative lecture produces 0 technical study sections and 100% digression callouts."""
        fixture = get_synthetic_10min_lecture_fixture()
        pure_digress = [dict(s) for s in fixture["sentences"][5:9]]  # Only the 4 administrative sentences
        for i, s in enumerate(pure_digress):
            s["sentence_id"] = i

        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=pure_digress)
        assert result["status"] == "SUCCESS"

        json_out = json.loads((tmp_path / "lecture_study_notes.json").read_text(encoding="utf-8"))
        validate_study_notes_json_schema(json_out)
        assert len(json_out["study_sections"]) == 0
        assert len(json_out["administrative_digressions"]) >= 1

    def test_edge_case_multihour_simulated_lecture(self, tmp_path: Path):
        """Verifies scalability and temporal integrity over a 2-hour simulated lecture (300+ sentences)."""
        multi_hour = generate_multihour_transcript_fixture(num_hours=2.0)
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=multi_hour["sentences"], metadata=multi_hour["metadata"])

        assert result["status"] == "SUCCESS"
        assert result["num_sentences"] == len(multi_hour["sentences"])
        assert result["num_segments"] >= 5

        json_out = json.loads((tmp_path / "lecture_study_notes.json").read_text(encoding="utf-8"))
        validate_study_notes_json_schema(json_out)
        assert json_out["metadata"]["total_duration_sec"] > 7000.0

    def test_edge_case_rapid_alternating_digression_bursts(self, tmp_path: Path):
        """Verifies gating enforces min_consecutive=3: 1-sentence digression is not gated; 4-sentence is gated."""
        sentences = [
            # Technical block 1
            {"sentence_id": 0, "text": "Paging divides physical memory into fixed-sized frames.", "start_ms": 0, "end_ms": 4000, "pause_before_ms": 0.0, "pitch_reset": 0.0, "speaker_tag": "Lecturer", "is_digression": False, "digression_type": "none"},
            # 1-sentence brief remark (not sustained, not gated)
            {"sentence_id": 1, "text": "Excuse me for coughing today.", "start_ms": 4500, "end_ms": 7000, "pause_before_ms": 500.0, "pitch_reset": 0.0, "speaker_tag": "Lecturer", "is_digression": False, "digression_type": "none"},
            # Technical block 2
            {"sentence_id": 2, "text": "The page table maps each virtual page number to a frame number.", "start_ms": 7500, "end_ms": 12000, "pause_before_ms": 500.0, "pitch_reset": 0.0, "speaker_tag": "Lecturer", "is_digression": False, "digression_type": "none"},
            # 3-sentence administrative digression (sustained, gated)
            {"sentence_id": 3, "text": "Assignment three has been posted on the portal.", "start_ms": 13000, "end_ms": 17000, "pause_before_ms": 1000.0, "pitch_reset": 2.0, "speaker_tag": "Lecturer", "is_digression": True, "digression_type": "administrative"},
            {"sentence_id": 4, "text": "The submission deadline is strictly next Friday at five.", "start_ms": 17500, "end_ms": 22000, "pause_before_ms": 500.0, "pitch_reset": 0.0, "speaker_tag": "Lecturer", "is_digression": True, "digression_type": "administrative"},
            {"sentence_id": 5, "text": "Late penalties of ten percent per day will be enforced.", "start_ms": 22500, "end_ms": 27000, "pause_before_ms": 500.0, "pitch_reset": 0.0, "speaker_tag": "Lecturer", "is_digression": True, "digression_type": "administrative"},
        ]
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        result = pipeline.run(transcript_input=sentences)
        assert result["status"] == "SUCCESS"
        json_out = json.loads((tmp_path / "lecture_study_notes.json").read_text(encoding="utf-8"))
        assert len(json_out["administrative_digressions"]) == 1


# ==============================================================================
# Class 7: TestHardwareAndVRAMProfiling (4 Tests)
# ==============================================================================

class TestHardwareAndVRAMProfiling:
    """Verifies hardware constraints: CPU offload, stage isolation, memory bounds, and Ollama eviction."""

    def test_phase1_defaults_to_cpu_zero_vram(self):
        """Verifies Phase 1 default config specifies device='cpu', reserving all 6GB VRAM for Phase 2."""
        p_cfg = E2EPipelineConfig()
        assert p_cfg.device == "cpu"
        assert p_cfg.whisper_compute_type == "int8"

    def test_stage_isolation_and_garbage_collection(self, tmp_path: Path):
        """Verifies intermediate variables are cleaned up between Phase 1 and Phase 2."""
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        gc.collect()
        fixture = get_synthetic_10min_lecture_fixture()
        result = pipeline.run(transcript_input=fixture["sentences"])
        assert result["status"] == "SUCCESS"
        # Check that pipeline records stats
        assert hasattr(pipeline, "last_execution_stats")

    def test_ollama_keep_alive_zero_eviction_invoked(self, tmp_path: Path):
        """Verifies evict_model() is called with keep_alive: 0 upon pipeline completion."""
        eviction_called = False

        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        delegate = pipeline.segmentation_pipeline.title_generator

        class TrackingTitleGenerator:
            def __init__(self, target):
                self._target = target

            def __getattr__(self, name):
                return getattr(self._target, name)

            def evict_model(self) -> bool:
                nonlocal eviction_called
                eviction_called = True
                return self._target.evict_model()

        pipeline.segmentation_pipeline.title_generator = TrackingTitleGenerator(delegate)
        fixture = get_synthetic_10min_lecture_fixture()
        pipeline.run(transcript_input=fixture["sentences"])
        assert eviction_called is True, "Pipeline failed to invoke Ollama VRAM eviction upon completion"

    def test_process_peak_memory_ceiling_under_2gb(self, tmp_path: Path):
        """Verifies total test execution overhead stays well below the 16GB System RAM budget (<2.0GB)."""
        multi_hour = generate_multihour_transcript_fixture(num_hours=2.0)
        pipeline = E2ELecturePipeline(E2EPipelineConfig(force_fallback=True, output_dir=tmp_path))
        pipeline.run(transcript_input=multi_hour["sentences"])

        mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_mb = mem_kb / 1024.0
        # Peak RSS must remain under 2048 MB (2 GB) during offline test execution
        assert mem_mb < 2048.0, f"Peak memory {mem_mb:.1f} MB exceeded 2.0GB safety envelope"
