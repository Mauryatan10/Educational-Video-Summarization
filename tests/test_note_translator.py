"""tests/test_note_translator.py - Unit and Integration Tests for Hindi Study Notes Translator."""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.note_generator import (
    ActiveRecallQuestion,
    AdministrativeDigression,
    AlgorithmicCodeBlock,
    CornellCueColumn,
    DetailedNotesSection,
    FormalDefinition,
    GlobalReviewBlock,
    LectureStudyNotes,
    MasterGlossaryEntry,
    MathematicalFormulation,
    PrerequisiteDAG,
    StudyModule,
    StudySection,
)
from src.pipeline.note_translator import (
    NoteTranslator,
    NoteTranslatorConfig,
    PlaceholderMasker,
)


# ==============================================================================
# 1. Tests for PlaceholderMasker
# ==============================================================================

def test_placeholder_masker_roundtrip_basic():
    masker = PlaceholderMasker()
    text = (
        "In spectral theory, we compute the eigenvalue $\\lambda$ of matrix $A \\in \\mathbb{R}^{n \\times n}$.\n"
        "The characteristic equation is:\n"
        "$$\n"
        "\\det(A - \\lambda I) = 0\n"
        "$$\n"
        "Use `np.linalg.eig(A)` to solve."
    )

    masked, mapping = masker.mask(text)

    # Assert equations and code were replaced with placeholders
    assert "$$" not in masked
    assert "\\lambda" not in masked
    assert "`np.linalg.eig(A)`" not in masked
    assert len(mapping) >= 3

    # Assert exact roundtrip restoration
    unmasked = masker.unmask(masked, mapping)
    assert unmasked == text


def test_placeholder_masker_multiline_code_and_math():
    masker = PlaceholderMasker()
    text = (
        "Here is the gradient descent implementation:\n"
        "```python\n"
        "def gradient_step(w, grad, lr):\n"
        "    return w - lr * grad\n"
        "```\n"
        "The update satisfies $w_{t+1} = w_t - \\eta \\nabla L(w_t)$."
    )

    masked, mapping = masker.mask(text)
    assert "```python" not in masked
    assert "$w_{t+1}" not in masked

    unmasked = masker.unmask(masked, mapping)
    assert unmasked == text


def test_placeholder_masker_whitespace_and_devanagari_mutations():
    """Verify whitespace, single-bracket, and Devanagari digit mutations are gracefully restored."""
    masker = PlaceholderMasker()
    text = "The loss is $\\mathcal{L}(y, \\hat{y})$ with rate $\\alpha$."
    masked, mapping = masker.mask(text)

    # 1. Spaced brackets: '[[0]]' -> '[ [ 0 ] ]'
    mutated_ws = re.sub(r"\[\[(\d+)\]\]", r"[ [ \1 ] ]", masked)
    assert mutated_ws != masked
    assert masker.unmask(mutated_ws, mapping) == text

    # 2. Devanagari digits + single bracket degradation: '[[0]]' -> '[[०]]', '[[1]]' -> '[१]'
    mutated_dev = masked.replace("[[0]]", "[[०]]").replace("[[1]]", "[१]")
    assert masker.unmask(mutated_dev, mapping) == text


def test_note_translator_piecewise_recovery_on_dropped_placeholder():
    """Verify 100% inline math preservation even when the LLM drops or corrupts placeholders."""
    translator = NoteTranslator(NoteTranslatorConfig(adapter_path=None))

    def corrupted_llm_chunk(chunk: str) -> str:
        # Simulate Sarvam-1 corrupting placeholders if present in the chunk
        if "[[" in chunk:
            return "लक्ष्य मानों के बीच वर्ग त्रुटि जहाँ Math_(math_) है।"
        return f"हिन्दी({chunk})"

    translator._translate_raw_chunk = corrupted_llm_chunk
    translator._is_loaded = True

    src = "The loss between target $y_i$ and prediction $f(x_i; \\theta)$ depends on $\\theta$."
    translated = translator.translate_text(src)

    assert "$y_i$" in translated
    assert "$f(x_i; \\theta)$" in translated
    assert "$\\theta$" in translated
    assert "Math_(math_)" not in translated


def test_placeholder_masker_empty_and_plain():
    masker = PlaceholderMasker()
    assert masker.mask("")[0] == ""
    assert masker.unmask("", {}) == ""

    plain = "This is a plain text without math or code."
    masked, mapping = masker.mask(plain)
    assert masked == plain
    assert len(mapping) == 0
    assert masker.unmask(masked, mapping) == plain


# ==============================================================================
# 2. Tests for NoteTranslator Chunking & Mechanics
# ==============================================================================

def test_note_translator_sentence_chunking():
    translator = NoteTranslator(NoteTranslatorConfig(max_words_per_chunk=30))
    long_text = (
        "This is the first sentence about neural networks and backpropagation. "
        "Here is the second sentence describing how the chain rule propagates gradients. "
        "Next, the third sentence elaborates on learning rates and adaptive moment estimation like Adam. "
        "Finally, the fourth sentence concludes with regularizations such as weight decay and dropout."
    )

    chunks = translator._split_into_chunks(long_text, max_words=30)
    assert len(chunks) >= 2
    # Check that sentences are not broken mid-word
    for chunk in chunks:
        assert len(chunk.split()) <= 45


def test_note_translator_mock_inference():
    """Verifies that full NoteTranslator translates LectureStudyNotes and renders Hindi Markdown."""
    translator = NoteTranslator(NoteTranslatorConfig(adapter_path=None))

    # Mock the internal _translate_raw_chunk to simulate Hindi output
    def mock_translate_raw_chunk(masked_chunk: str) -> str:
        # Simple simulated translation adding Hindi anchor
        return f"अनुवादित: {masked_chunk}"

    translator._translate_raw_chunk = mock_translate_raw_chunk
    translator._is_loaded = True

    # Construct sample LectureStudyNotes
    sec = StudySection(
        section_id=0,
        topic_title="Eigenvalues and Spectral Decomposition",
        start_time_str="00:00:00",
        end_time_str="00:05:00",
        is_digression=False,
        digression_type="none",
        prerequisite_tags=["Linear Algebra"],
        cornell_cue_column=CornellCueColumn(
            cue_questions=["What is a symmetric matrix?"],
            core_keywords=["Eigenvalue", "Orthogonal Matrix"],
        ),
        detailed_notes=DetailedNotesSection(
            conceptual_exposition="In spectral theory, matrix $A$ satisfies $Av = \\lambda v$.",
            formal_definitions=[
                FormalDefinition(term="Symmetric Matrix", definition="A square matrix equal to its transpose.")
            ],
            mathematical_formulations=[
                MathematicalFormulation(latex_expression="$$A = Q \\Lambda Q^T$$", explanation="Spectral theorem.")
            ],
            algorithmic_code_blocks=[
                AlgorithmicCodeBlock(language="python", code="Q, L = np.linalg.eigh(A)", caption="Eigen decomposition.")
            ],
            concrete_examples=["Consider a 2x2 covariance matrix."],
        ),
        takeaways=["Symmetric matrices have real eigenvalues."],
    )

    notes = LectureStudyNotes(
        schema_version="2.1",
        metadata={
            "lecture_title": "Spectral Graph Theory",
            "course": "Mathematics",
            "total_duration_str": "00:05:00",
            "total_sentences": 15,
        },
        prerequisite_dag=PrerequisiteDAG(nodes=[], edges=[], topological_order=[]),
        table_of_contents=[{"section_id": 0, "title": "Eigenvalues", "start_time_str": "00:00:00", "end_time_str": "00:05:00", "is_digression": False}],
        study_sections=[sec],
        administrative_digressions=[
            AdministrativeDigression(
                section_id=99,
                title="Midterm Exam Logistics",
                start_time_str="00:05:00",
                end_time_str="00:06:00",
                digression_type="administrative",
                announcement_notes="Exam is on Friday.",
            )
        ],
        global_review=GlobalReviewBlock(
            master_glossary=[
                MasterGlossaryEntry(term="Eigenvalue", definition="Scalar factor for eigenvector.", referenced_in_sections=[0])
            ],
            active_recall_review_sheet=[
                ActiveRecallQuestion(question="How is PCA related to SVD?", section_id=0, core_concept="PCA")
            ],
            executive_synthesis="Comprehensive lecture on spectral theory.",
        ),
    )

    hindi_md, hindi_json = translator.translate_study_notes(notes, auto_unload=False)

    # 1. Verify Markdown structure
    assert "# अनुवादित: Spectral Graph Theory (अध्ययन नोट्स)" in hindi_md
    assert "## कार्यकारी अवलोकन (Executive Overview)" in hindi_md
    assert "## विषय - सूची (Table of Contents)" in hindi_md
    assert "### औपचारिक परिभाषाएँ (Formal Definitions)" in hindi_md
    assert "### गणितीय सूत्र (Mathematical Formulations)" in hindi_md
    assert "### एल्गोरिथम संरचना (Algorithmic Formulation)" in hindi_md

    # 2. Verify LaTeX Math and Code were preserved VERBATIM
    assert "$$A = Q \\Lambda Q^T$$" in hindi_md
    assert "$A$" in hindi_md
    assert "$Av = \\lambda v$" in hindi_md
    assert "```python\nQ, L = np.linalg.eigh(A)\n```" in hindi_md

    # 3. Verify JSON schema
    assert hindi_json["schema_version"] == "2.1"
    assert "अनुवादित" in hindi_json["metadata"]["lecture_title"]
    assert hindi_json["study_sections"][0]["detailed_notes"]["mathematical_formulations"][0]["latex_expression"] == "$$A = Q \\Lambda Q^T$$"
    assert hindi_json["study_sections"][0]["detailed_notes"]["algorithmic_code_blocks"][0]["code"] == "Q, L = np.linalg.eigh(A)"


def test_e2e_pipeline_with_translate_hindi(tmp_path):
    """Verifies that E2ELecturePipeline with translate_hindi=True produces both English and Hindi notes."""
    from src.pipeline.run_e2e_lecture import E2ELecturePipeline, E2EPipelineConfig

    config = E2EPipelineConfig(
        device="cpu",
        offline=True,
        force_fallback=True,
        translate_hindi=True,
        output_dir=tmp_path / "artifacts",
    )
    pipeline = E2ELecturePipeline(config=config)

    # Mock the internal translate function to avoid downloading base model in unit test
    def mock_translate_raw(chunk: str) -> str:
        return f"हिन्दी: {chunk}"

    pipeline.note_translator._translate_raw_chunk = mock_translate_raw
    pipeline.note_translator._is_loaded = True

    sample_transcript = [
        {"sentence_id": 0, "text": "Welcome to deep learning lecture. Today we cover backpropagation.", "start_ms": 0, "end_ms": 3000},
        {"sentence_id": 1, "text": "The gradient is computed with the chain rule.", "start_ms": 3000, "end_ms": 6000},
        {"sentence_id": 2, "text": "We optimize weights using Adam with learning rate 0.001.", "start_ms": 6000, "end_ms": 9000},
    ]

    result = pipeline.run_from_transcript(sample_transcript, metadata={"course": "CS", "lecture_title": "Deep Learning"})

    assert result.status == "SUCCESS"
    assert result.study_notes_markdown is not None
    assert result.study_notes_json is not None
    assert result.study_notes_hindi_markdown is not None
    assert result.study_notes_hindi_json is not None

    # Check Hindi content
    assert "हिन्दी" in result.study_notes_hindi_markdown
    assert result.study_notes_hindi_json["schema_version"] == result.study_notes_json["schema_version"]

    # Check execution summary latencies
    latencies = result.execution_summary["stage_latencies"]
    assert "translation_time_sec" in latencies
    assert latencies["translation_time_sec"] >= 0.0

    # Check exported artifacts
    assert "study_notes_hindi_markdown" in result.artifacts
    assert "study_notes_hindi_json" in result.artifacts
    assert result.artifacts["study_notes_hindi_markdown"].exists()
    assert result.artifacts["study_notes_hindi_json"].exists()
