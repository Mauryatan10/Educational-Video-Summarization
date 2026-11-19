"""Unit tests for Ollama note generator truncation resilience and JSON repair."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.note_generator import (
    NoteGeneratorConfig,
    OllamaNoteElaborator,
    PedagogicalNoteGenerator,
    StudySection,
)
from src.pipeline.run_e2e_lecture import E2EPipelineConfig


class TestNoteGeneratorTruncationResilience:
    """Test suite ensuring Ollama length limits and JSON truncations never crash the pipeline."""

    def test_default_config_token_limits(self):
        """Verify default num_predict is increased to 4096 tokens."""
        note_cfg = NoteGeneratorConfig()
        assert note_cfg.ollama_num_predict == 4096
        assert note_cfg.ollama_reduce_num_predict == 4096

        e2e_cfg = E2EPipelineConfig()
        assert e2e_cfg.ollama_num_predict == 4096
        assert e2e_cfg.ollama_reduce_num_predict == 4096

    def test_repair_truncated_exposition_json(self):
        """Verify JSON truncated mid-exposition is properly closed and parsed."""
        raw = (
            '{"conceptual_exposition": "The GRPO algorithm calculates advantages using '
            'relative rewards within sampled candidate groups without a separate critic.'
        )
        parsed = OllamaNoteElaborator(NoteGeneratorConfig())._parse_and_repair_json(raw)
        assert parsed is not None
        assert "conceptual_exposition" in parsed
        assert "GRPO algorithm" in parsed["conceptual_exposition"]

    def test_repair_truncated_takeaways_json(self):
        """Verify JSON truncated mid-takeaways list is repaired and salvaged."""
        raw = (
            '{\n'
            '  "conceptual_exposition": "GRPO formulation and loss function.",\n'
            '  "formal_definitions": [{"term": "GRPO", "definition": "Group Relative Policy Optimization"}],\n'
            '  "takeaways": ["Group normalization reduces variance", "No critic network required'
        )
        parsed = OllamaNoteElaborator(NoteGeneratorConfig())._parse_and_repair_json(raw)
        assert parsed is not None
        assert parsed["conceptual_exposition"] == "GRPO formulation and loss function."
        assert len(parsed["formal_definitions"]) == 1
        assert parsed["formal_definitions"][0]["term"] == "GRPO"

    def test_repair_dangling_key_json(self):
        """Verify JSON truncated right at an incomplete key is repaired."""
        raw = (
            '{\n'
            '  "conceptual_exposition": "Detailed explanation of mathematical formulation.",\n'
            '  "cue_questions": ["What is GRPO?"],\n'
            '  "takeaways":'
        )
        parsed = OllamaNoteElaborator(NoteGeneratorConfig())._parse_and_repair_json(raw)
        assert parsed is not None
        assert parsed["conceptual_exposition"] == "Detailed explanation of mathematical formulation."
        assert parsed["cue_questions"] == ["What is GRPO?"]

    def test_salvage_regex_when_syntax_heavily_corrupted(self):
        """Verify regex salvage extracts exposition when JSON brackets are invalid."""
        raw = (
            'Random preface from model...\n'
            '"conceptual_exposition": "DeepSeek-Math GRPO objective optimizes policy gradients over group batches.",\n'
            '"takeaways": ["Point A", "Point B'
        )
        parsed = OllamaNoteElaborator(NoteGeneratorConfig())._parse_and_repair_json(raw)
        assert parsed is not None
        assert "conceptual_exposition" in parsed
        assert "DeepSeek-Math GRPO" in parsed["conceptual_exposition"]

    @patch("urllib.request.urlopen")
    def test_elaborate_segment_on_length_limit(self, mock_urlopen):
        """Verify elaborate_segment does NOT raise RuntimeError when done_reason='length'."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        truncated_ollama_payload = {
            "model": "qwen3.5-4b",
            "done": True,
            "done_reason": "length",
            "eval_count": 2048,
            "response": (
                '{"conceptual_exposition": "The GRPO objective replaces standard advantage estimates '
                'with normalized rewards computed across groups of G completions for each prompt.", '
                '"formal_definitions": [{"term": "GRPO", "definition": "Group Relative Policy Optimization"}], '
                '"takeaways": ["Eliminates critic network'
            ),
        }
        mock_resp.read.return_value = json.dumps(truncated_ollama_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        elaborator = OllamaNoteElaborator(NoteGeneratorConfig())
        result = elaborator.elaborate_segment(
            segment_text="GRPO eliminates the critic network...",
            topic_title="GRPO Algorithm and Group Normalization",
            prerequisites=["PPO", "Reinforcement Learning"],
        )

        assert result is not None
        assert "conceptual_exposition" in result
        assert "GRPO objective" in result["conceptual_exposition"]

    @patch("urllib.request.urlopen")
    def test_elaborate_segment_graceful_on_http_error(self, mock_urlopen):
        """Verify elaborate_segment returns None instead of crashing when Ollama is unreachable."""
        mock_urlopen.side_effect = ConnectionRefusedError("Ollama connection failed")

        elaborator = OllamaNoteElaborator(NoteGeneratorConfig(force_fallback=False))
        # Must return None, never raise RuntimeError
        result = elaborator.elaborate_segment(
            segment_text="Sample text",
            topic_title="Section Title",
            prerequisites=[],
        )
        assert result is None

    def test_pedagogical_generator_fallback_when_llm_none(self):
        """Verify PedagogicalNoteGenerator produces valid StudySection when elaborator returns None."""
        generator = PedagogicalNoteGenerator(NoteGeneratorConfig(use_ollama=False, force_fallback=False))
        # Mock elaborator returning None
        generator.elaborator.elaborate_segment = MagicMock(return_value=None)

        seg_dict = {
            "segment_id": 0,
            "topic_title": "GRPO Algorithm and Group Normalization",
            "start_time_str": "00:01:00",
            "end_time_str": "00:05:00",
            "prerequisite_tags": ["PPO", "Reinforcement Learning"],
            "raw_text": "GRPO calculates policy gradient advantages over groups. Let $$J(\\theta)$$ be the objective.",
            "sentences": [
                {"sentence_id": 0, "text": "GRPO calculates policy gradient advantages over groups."},
                {"sentence_id": 1, "text": "Let $$J(\\theta)$$ be the objective."},
            ],
            "cue_questions": ["What is GRPO?"],
        }

        sec = generator._process_technical_segment(seg_dict)
        assert isinstance(sec, StudySection)
        assert sec.topic_title == "GRPO Algorithm and Group Normalization"
        assert sec.detailed_notes.conceptual_exposition == seg_dict["raw_text"]
        assert len(sec.detailed_notes.mathematical_formulations) >= 1
        assert sec.detailed_notes.mathematical_formulations[0].latex_expression == "$$J(\\theta)$$"
        assert len(sec.cornell_cue_column.cue_questions) >= 1

    @patch("urllib.request.urlopen")
    def test_synthesize_global_review_on_length_limit(self, mock_urlopen):
        """Verify synthesize_global_review does NOT crash on length limit."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        truncated_payload = {
            "model": "qwen3.5-4b",
            "done": True,
            "done_reason": "length",
            "eval_count": 2048,
            "response": '{"executive_synthesis": "This lecture provides an extensive investigation into modern RL algorithms',
        }
        mock_resp.read.return_value = json.dumps(truncated_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        elaborator = OllamaNoteElaborator(NoteGeneratorConfig())
        result = elaborator.synthesize_global_review(
            study_sections=[MagicMock(section_id=0, topic_title="S1", start_time_str="00:00:00", end_time_str="00:01:00", prerequisite_tags=[], takeaways=["T1"], detailed_notes=MagicMock(formal_definitions=[]))],
            toc=[{"section_id": 0, "start_time_str": "00:00:00", "end_time_str": "00:01:00", "title": "S1"}],
        )

        assert result is not None
        assert "executive_synthesis" in result
        assert "RL algorithms" in result["executive_synthesis"]
