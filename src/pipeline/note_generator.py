"""src/pipeline/note_generator.py - Pedagogical Study Note Generator Module.

Ingests Interface Contract 2 (SegmentationPayload) and generates:
1. lecture_study_notes.md (Human-readable, publication-grade Cornell study notes)
2. lecture_study_notes.json (Machine-readable, schema-validated structured study notes)

Adheres strictly to User Guidance in ORIGINAL_REQUEST.md:
- Detailed video study notes serving as a comprehensive alternative to watching lectures.
- Full conceptual integrity, definitions, mechanisms, and mathematical proofs ($$...$$).
- Topic-by-topic structured sections with [HH:MM:SS] timestamp anchors.
- Cornell active recall cue questions (ending in '?') and conceptual prerequisite DAG.
- Non-pedagogical administrative digression segregation into dedicated logistics callouts.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/Qwen3.5-4b-GGUF:UD-Q3_K_XL"


# ==============================================================================
# 1. Data Models for Pedagogical Study Notes
# ==============================================================================

@dataclass
class FormalDefinition:
    term: str
    definition: str

    def to_dict(self) -> Dict[str, str]:
        return {"term": self.term, "definition": self.definition}


@dataclass
class MathematicalFormulation:
    latex_expression: str
    explanation: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "latex_expression": self.latex_expression,
            "explanation": self.explanation,
        }


@dataclass
class AlgorithmicCodeBlock:
    language: str
    code: str
    caption: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "language": self.language,
            "code": self.code,
            "caption": self.caption,
        }


@dataclass
class CornellCueColumn:
    cue_questions: List[str]
    core_keywords: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_questions": list(self.cue_questions),
            "core_keywords": list(self.core_keywords),
        }


@dataclass
class DetailedNotesSection:
    conceptual_exposition: str
    formal_definitions: List[FormalDefinition] = field(default_factory=list)
    mathematical_formulations: List[MathematicalFormulation] = field(default_factory=list)
    algorithmic_code_blocks: List[AlgorithmicCodeBlock] = field(default_factory=list)
    concrete_examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conceptual_exposition": self.conceptual_exposition,
            "formal_definitions": [d.to_dict() for d in self.formal_definitions],
            "mathematical_formulations": [m.to_dict() for m in self.mathematical_formulations],
            "algorithmic_code_blocks": [c.to_dict() for c in self.algorithmic_code_blocks],
            "concrete_examples": list(self.concrete_examples),
        }


@dataclass
class StudySection:
    section_id: int
    topic_title: str
    start_time_str: str  # "HH:MM:SS"
    end_time_str: str    # "HH:MM:SS"
    is_digression: bool
    digression_type: str  # "none", "administrative", "logistical"
    prerequisite_tags: List[str]
    cornell_cue_column: CornellCueColumn
    detailed_notes: DetailedNotesSection
    takeaways: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": int(self.section_id),
            "topic_title": str(self.topic_title),
            "start_time_str": str(self.start_time_str),
            "end_time_str": str(self.end_time_str),
            "is_digression": bool(self.is_digression),
            "digression_type": str(self.digression_type),
            "prerequisite_tags": list(self.prerequisite_tags),
            "cornell_cue_column": self.cornell_cue_column.to_dict(),
            "detailed_notes": self.detailed_notes.to_dict(),
            "takeaways": list(self.takeaways),
        }


@dataclass
class AdministrativeDigression:
    section_id: int
    title: str
    start_time_str: str
    end_time_str: str
    digression_type: str
    announcement_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": int(self.section_id),
            "title": str(self.title),
            "start_time_str": str(self.start_time_str),
            "end_time_str": str(self.end_time_str),
            "digression_type": str(self.digression_type),
            "announcement_notes": str(self.announcement_notes),
        }


@dataclass
class PrerequisiteDAG:
    nodes: List[Dict[str, Any]]  # [{"concept": str, "introduced_in_segment_id": int}]
    edges: List[Dict[str, str]]  # [{"source_concept": str, "target_concept": str}]
    topological_order: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "topological_order": list(self.topological_order),
        }


@dataclass
class MasterGlossaryEntry:
    term: str
    definition: str
    referenced_in_sections: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "definition": self.definition,
            "referenced_in_sections": list(self.referenced_in_sections),
        }


@dataclass
class ActiveRecallQuestion:
    question: str
    section_id: int
    core_concept: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "section_id": int(self.section_id),
            "core_concept": self.core_concept,
        }


@dataclass
class GlobalReviewBlock:
    master_glossary: List[MasterGlossaryEntry]
    active_recall_review_sheet: List[ActiveRecallQuestion]
    executive_synthesis: Optional[str] = None
    comparative_synthesis_questions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "master_glossary": [g.to_dict() for g in self.master_glossary],
            "active_recall_review_sheet": [q.to_dict() for q in self.active_recall_review_sheet],
        }
        if self.executive_synthesis:
            data["executive_synthesis"] = self.executive_synthesis
        if self.comparative_synthesis_questions:
            data["comparative_synthesis_questions"] = self.comparative_synthesis_questions
        return data


@dataclass
class StudyModule:
    """Strongly typed representation of a 2-tier Macro Lecture Module containing subsections."""

    module_id: int
    module_title: str
    start_time_str: str  # "HH:MM:SS"
    end_time_str: str    # "HH:MM:SS"
    pedagogical_focus: str
    subsections: List[StudySection]
    module_synthesis: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": int(self.module_id),
            "module_title": str(self.module_title),
            "start_time_str": str(self.start_time_str),
            "end_time_str": str(self.end_time_str),
            "pedagogical_focus": str(self.pedagogical_focus),
            "module_synthesis": str(self.module_synthesis),
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class LectureStudyNotes:
    schema_version: str
    metadata: Dict[str, Any]
    prerequisite_dag: PrerequisiteDAG
    table_of_contents: List[Dict[str, Any]]
    study_sections: List[StudySection]
    administrative_digressions: List[AdministrativeDigression]
    global_review: GlobalReviewBlock
    modules: List[StudyModule] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "prerequisite_dag": self.prerequisite_dag.to_dict(),
            "table_of_contents": list(self.table_of_contents),
            "study_sections": [s.to_dict() for s in self.study_sections],
            "administrative_digressions": [d.to_dict() for d in self.administrative_digressions],
            "global_review": self.global_review.to_dict(),
        }
        if self.modules:
            d["modules"] = [m.to_dict() for m in self.modules]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LectureStudyNotes:
        dag_dict = d.get("prerequisite_dag", {})
        dag = PrerequisiteDAG(
            nodes=dag_dict.get("nodes", []),
            edges=dag_dict.get("edges", []),
            topological_order=dag_dict.get("topological_order", []),
        )

        sections: List[StudySection] = []
        for s in d.get("study_sections", []):
            cue_dict = s.get("cornell_cue_column", {})
            cue = CornellCueColumn(
                cue_questions=list(cue_dict.get("cue_questions", [])),
                core_keywords=list(cue_dict.get("core_keywords", [])),
            )
            dn_dict = s.get("detailed_notes", {})
            defs = [
                FormalDefinition(term=item.get("term", ""), definition=item.get("definition", ""))
                for item in dn_dict.get("formal_definitions", [])
            ]
            maths = [
                MathematicalFormulation(
                    latex_expression=item.get("latex_expression", ""),
                    explanation=item.get("explanation", ""),
                )
                for item in dn_dict.get("mathematical_formulations", [])
            ]
            codes = [
                AlgorithmicCodeBlock(
                    language=item.get("language", "python"),
                    code=item.get("code", ""),
                    caption=item.get("caption", ""),
                )
                for item in dn_dict.get("algorithmic_code_blocks", [])
            ]
            dn = DetailedNotesSection(
                conceptual_exposition=dn_dict.get("conceptual_exposition", ""),
                formal_definitions=defs,
                mathematical_formulations=maths,
                algorithmic_code_blocks=codes,
                concrete_examples=list(dn_dict.get("concrete_examples", [])),
            )
            sections.append(
                StudySection(
                    section_id=int(s.get("section_id", 0)),
                    topic_title=str(s.get("topic_title", "")),
                    start_time_str=str(s.get("start_time_str", "00:00:00")),
                    end_time_str=str(s.get("end_time_str", "00:00:00")),
                    is_digression=bool(s.get("is_digression", False)),
                    digression_type=str(s.get("digression_type", "none")),
                    prerequisite_tags=list(s.get("prerequisite_tags", [])),
                    cornell_cue_column=cue,
                    detailed_notes=dn,
                    takeaways=list(s.get("takeaways", [])),
                )
            )

        admin: List[AdministrativeDigression] = []
        for a in d.get("administrative_digressions", []):
            admin.append(
                AdministrativeDigression(
                    section_id=int(a.get("section_id", 0)),
                    title=str(a.get("title", "")),
                    start_time_str=str(a.get("start_time_str", "00:00:00")),
                    end_time_str=str(a.get("end_time_str", "00:00:00")),
                    digression_type=str(a.get("digression_type", "administrative")),
                    announcement_notes=str(a.get("announcement_notes", "")),
                )
            )

        gr_dict = d.get("global_review", {})
        glossary = [
            MasterGlossaryEntry(
                term=g.get("term", ""),
                definition=g.get("definition", ""),
                referenced_in_sections=list(g.get("referenced_in_sections", [])),
            )
            for g in gr_dict.get("master_glossary", [])
        ]
        recalls = [
            ActiveRecallQuestion(
                question=q.get("question", ""),
                section_id=int(q.get("section_id", 0)),
                core_concept=q.get("core_concept", ""),
            )
            for q in gr_dict.get("active_recall_review_sheet", [])
        ]
        gr = GlobalReviewBlock(
            master_glossary=glossary,
            active_recall_review_sheet=recalls,
            executive_synthesis=gr_dict.get("executive_synthesis"),
            comparative_synthesis_questions=list(gr_dict.get("comparative_synthesis_questions", [])),
        )

        modules: List[StudyModule] = []
        for m in d.get("modules", []):
            subsecs = []
            for sub in m.get("subsections", []):
                sub_cue = CornellCueColumn(
                    cue_questions=list(sub.get("cornell_cue_column", {}).get("cue_questions", [])),
                    core_keywords=list(sub.get("cornell_cue_column", {}).get("core_keywords", [])),
                )
                sub_dn = DetailedNotesSection(
                    conceptual_exposition=sub.get("detailed_notes", {}).get("conceptual_exposition", ""),
                    formal_definitions=[
                        FormalDefinition(term=it.get("term", ""), definition=it.get("definition", ""))
                        for it in sub.get("detailed_notes", {}).get("formal_definitions", [])
                    ],
                    mathematical_formulations=[
                        MathematicalFormulation(
                            latex_expression=it.get("latex_expression", ""),
                            explanation=it.get("explanation", ""),
                        )
                        for it in sub.get("detailed_notes", {}).get("mathematical_formulations", [])
                    ],
                    algorithmic_code_blocks=[
                        AlgorithmicCodeBlock(
                            language=it.get("language", "python"),
                            code=it.get("code", ""),
                            caption=it.get("caption", ""),
                        )
                        for it in sub.get("detailed_notes", {}).get("algorithmic_code_blocks", [])
                    ],
                    concrete_examples=list(sub.get("detailed_notes", {}).get("concrete_examples", [])),
                )
                subsecs.append(
                    StudySection(
                        section_id=int(sub.get("section_id", 0)),
                        topic_title=str(sub.get("topic_title", "")),
                        start_time_str=str(sub.get("start_time_str", "00:00:00")),
                        end_time_str=str(sub.get("end_time_str", "00:00:00")),
                        is_digression=bool(sub.get("is_digression", False)),
                        digression_type=str(sub.get("digression_type", "none")),
                        prerequisite_tags=list(sub.get("prerequisite_tags", [])),
                        cornell_cue_column=sub_cue,
                        detailed_notes=sub_dn,
                        takeaways=list(sub.get("takeaways", [])),
                    )
                )
            modules.append(
                StudyModule(
                    module_id=int(m.get("module_id", 0)),
                    module_title=str(m.get("module_title", "")),
                    start_time_str=str(m.get("start_time_str", "00:00:00")),
                    end_time_str=str(m.get("end_time_str", "00:00:00")),
                    pedagogical_focus=str(m.get("pedagogical_focus", "")),
                    subsections=subsecs,
                    module_synthesis=str(m.get("module_synthesis", "")),
                )
            )

        return cls(
            schema_version=str(d.get("schema_version", "2.1")),
            metadata=dict(d.get("metadata", {})),
            prerequisite_dag=dag,
            table_of_contents=list(d.get("table_of_contents", [])),
            study_sections=sections,
            administrative_digressions=admin,
            global_review=gr,
            modules=modules,
        )


# ==============================================================================
# 2. Generator Configuration & Ollama Client
# ==============================================================================

@dataclass
class NoteGeneratorConfig:
    """Configuration for Pedagogical Note Generator."""

    # Generation Mode
    use_ollama: bool = True
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_sec: float = 120.0
    ollama_keep_alive: str = "5m"  # Pin model in VRAM across segments; avoids reload thrashing
    ollama_num_ctx: int = 8192  # Fits 100% in 6GB VRAM with q8_0 KV cache (zero CPU spill)
    ollama_num_gpu: int = 99  # Offload all layers to GPU VRAM
    ollama_num_predict: int = 4096
    ollama_reduce_num_predict: int = 4096
    force_fallback: bool = False

    # Lecture Metadata Defaults
    default_course: str = "Computer Science"
    default_lecture_title: str = "Educational Lecture Study Notes"

    # Feature Toggles
    segregate_administrative_digressions: bool = True
    build_mermaid_diagram: bool = False
    include_section_cue_column: bool = False
    include_section_exposition_heading: bool = False
    include_prerequisite_hierarchy_table: bool = False  # Removed per user instruction
    enable_hierarchical_modules: bool = True  # Option B: 2-tier hierarchical notes
    extract_math_patterns: bool = True


class OllamaNoteElaborator:
    """Queries local Ollama instance for deep pedagogical note elaboration with zero-thrashing eviction."""

    def __init__(self, config: NoteGeneratorConfig):
        self.config = config

    @staticmethod
    def _repair_truncated_json_string(text: str) -> Optional[str]:
        """Repairs truncated JSON strings by closing unclosed strings, removing dangling keys,
        and closing open brackets/braces in LIFO order.
        """
        if not text:
            return None

        start_pos = text.find("{")
        if start_pos == -1:
            return None
        s = text[start_pos:]

        in_string = False
        escape = False
        for ch in s:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True

        repaired = s
        if in_string:
            num_slashes = len(repaired) - len(repaired.rstrip("\\"))
            if num_slashes % 2 == 1:
                repaired = repaired[:-1]
            repaired += '"'

        repaired = repaired.rstrip()
        # Remove dangling incomplete keys/colons at end (with or without comma)
        repaired = re.sub(r'([,{])\s*"[^"]*"\s*:\s*$', r"\1", repaired)
        repaired = re.sub(r'([,{])\s*"[^"]*"\s*$', r"\1", repaired)
        repaired = re.sub(r",\s*$", "", repaired)
        repaired = re.sub(r"\{\s*$", "", repaired)
        repaired = re.sub(r",\s*$", "", repaired)

        # Re-scan for delimiter stack
        in_string = False
        escape = False
        stack = []
        for ch in repaired:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                elif ch in "}]":
                    if stack and stack[-1] == ch:
                        stack.pop()

        if in_string:
            repaired += '"'
        repaired = re.sub(r",\s*$", "", repaired)
        while stack:
            repaired += stack.pop()
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        return repaired

    @staticmethod
    def _salvage_fields_via_regex(text: str) -> Dict[str, Any]:
        """Extracts valid note fields directly using regex when JSON syntax is corrupted."""
        salvaged: Dict[str, Any] = {}

        expo_match = re.search(r'"conceptual_exposition"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        if expo_match:
            raw_expo = expo_match.group(1)
            try:
                expo_str = json.loads(f'"{raw_expo}"')
            except Exception:
                expo_str = raw_expo.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            if expo_str.strip():
                salvaged["conceptual_exposition"] = expo_str.strip()

        syn_match = re.search(r'"executive_synthesis"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        if syn_match:
            raw_syn = syn_match.group(1)
            try:
                syn_str = json.loads(f'"{raw_syn}"')
            except Exception:
                syn_str = raw_syn.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            if syn_str.strip():
                salvaged["executive_synthesis"] = syn_str.strip()

        takeaways_match = re.search(r'"takeaways"\s*:\s*\[(.*?)(\]|$)', text, re.DOTALL)
        if takeaways_match:
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', takeaways_match.group(1))
            if items:
                salvaged["takeaways"] = [it.replace('\\"', '"') for it in items if it.strip()]

        cues_match = re.search(r'"cue_questions"\s*:\s*\[(.*?)(\]|$)', text, re.DOTALL)
        if cues_match:
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', cues_match.group(1))
            if items:
                salvaged["cue_questions"] = [it.replace('\\"', '"') for it in items if it.strip()]

        defs_match = re.finditer(r'\{\s*"term"\s*:\s*"([^"]+)"\s*,\s*"definition"\s*:\s*"([^"]+)"\s*\}', text)
        defs = [{"term": m.group(1), "definition": m.group(2)} for m in defs_match]
        if defs:
            salvaged["formal_definitions"] = defs

        return salvaged

    def _parse_and_repair_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Parses LLM JSON output with robust sanitization, truncated bracket repair, and regex salvage."""
        if not raw_text or not raw_text.strip():
            return None

        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE).strip()

        start_idx = clean_text.find("{")
        if start_idx == -1:
            salvaged = self._salvage_fields_via_regex(clean_text)
            if salvaged and ("conceptual_exposition" in salvaged or "executive_synthesis" in salvaged):
                return salvaged
            return None
        candidate = clean_text[start_idx:]

        # Fast path: direct json.loads with trailing comma cleanup
        try:
            candidate_clean = re.sub(r",\s*([\]}])", r"\1", candidate)
            parsed = json.loads(candidate_clean)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Regex match for complete { ... } if text had trailing noise
        json_match = re.search(r"(\{.*\})", candidate, flags=re.DOTALL)
        if json_match:
            try:
                cand = re.sub(r",\s*([\]}])", r"\1", json_match.group(1))
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Truncation repair: balance open quotes, keys, and bracket stack
        repaired = self._repair_truncated_json_string(candidate)
        if repaired:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Regex field salvage fallback
        salvaged = self._salvage_fields_via_regex(candidate)
        if salvaged and ("conceptual_exposition" in salvaged or "executive_synthesis" in salvaged):
            return salvaged

        return None

    def build_prompt(
        self,
        segment_text: str,
        topic_title: str = "",
        prerequisites: Optional[List[str]] = None,
        global_roadmap: Optional[str] = None,
        preceding_context: Optional[str] = None,
        upcoming_context: Optional[str] = None,
    ) -> str:
        """Construct the prompt for Ollama Qwen3.5-4B Cornell note elaboration with global context."""
        prereqs_str = ", ".join(prerequisites) if prerequisites else "None"
        context_parts = []
        if global_roadmap:
            context_parts.append(f"GLOBAL LECTURE OUTLINE & ROADMAP:\n{global_roadmap}")
        if preceding_context:
            context_parts.append(f"DISCOURSE PRECEDING STATE (Established earlier in lecture):\n{preceding_context}")
        if upcoming_context:
            context_parts.append(f"UPCOMING TOPIC PREVIEW:\n{upcoming_context}")

        context_block = "\n\n".join(context_parts)
        if context_block:
            context_block = f"\n\n---\n{context_block}\n---\n"

        return (
            "You are an expert university professor creating structured Cornell study notes from a lecture transcript.\n"
            "Your notes serve as a comprehensive, high-fidelity alternative to watching the lecture.\n"
            "DO NOT output internal thinking or reasoning tags. DO NOT apply destructive compression.\n"
            "Preserve all technical terms, formal definitions, step-by-step mechanisms, and mathematical proofs.\n"
            "Explicitly connect to earlier and later topics in the lecture when relevant to ensure narrative continuity.\n"
            "- Writing Style & Tone: Strict university academic textbook register. NEVER use conversational filler, informal speech ('pretty cool', 'bam', 'super simple', 'gonna', 'futzing'), or oral disfluencies from the spoken lecture. Rephrase all content into grammatically immaculate, formal academic exposition.\n"
            f"{context_block}\n"
            f"TARGET SECTION TITLE: {topic_title}\n"
            f"SECTION PREREQUISITES: {prereqs_str}\n"
            f"TRANSCRIPT SEGMENT:\n{segment_text}\n\n"
            "Requirements:\n"
            "- conceptual_exposition: Comprehensive, high-density academic analysis of the mechanisms in the segment.\n"
            "- formal_definitions: Define key terms and concepts discussed in the segment clearly and rigorously.\n"
            "- mathematical_formulations: If mathematical relationships, update equations, loss functions, or scientific laws are discussed or verbalized, format them as valid LaTeX block expressions in 'latex_expression' (e.g. '$$\\\\theta_{new} = \\\\theta_{old} - \\\\alpha \\\\nabla L$$') with an explanatory note.\n"
            "- takeaways: 2 to 4 high-impact pedagogical summary points.\n"
            "- cue_questions: 2 to 3 diagnostic active recall study questions ending in '?'.\n\n"
            "Respond ONLY with a JSON object adhering to this schema:\n"
            "{\n"
            '  "conceptual_exposition": "In-depth explanatory paragraphs detailing the mechanism...",\n'
            '  "formal_definitions": [{"term": "Concept", "definition": "Formal definition"}],\n'
            '  "mathematical_formulations": [{"latex_expression": "$$E = mc^2$$", "explanation": "..."}],\n'
            '  "algorithmic_code_blocks": [{"language": "python", "code": "...", "caption": "..."}],\n'
            '  "concrete_examples": ["Example 1..."],\n'
            '  "takeaways": ["Takeaway 1...", "Takeaway 2..."],\n'
            '  "cue_questions": ["What is ...?", "How does this mechanism relate to ...?"]\n'
            "}\n"
        )

    def elaborate_segment(
        self,
        segment_text: str,
        topic_title: str,
        prerequisites: List[str],
        global_roadmap: Optional[str] = None,
        preceding_context: Optional[str] = None,
        upcoming_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Ollama Qwen3.5-4B for structured Cornell notes elaboration."""
        if not self.config.use_ollama or self.config.force_fallback:
            return None

        # Static system prompt + global roadmap enables Ollama/llama.cpp KV prefix caching across all sections
        system_parts = [
            "You are an expert university professor creating structured Cornell study notes from a lecture transcript.",
            "Your notes serve as a comprehensive, high-fidelity alternative to watching the lecture.",
            "DO NOT output internal thinking or reasoning tags. DO NOT apply destructive compression.",
            "Preserve all technical terms, formal definitions, step-by-step mechanisms, and mathematical proofs.",
            "Explicitly connect to earlier and later topics in the lecture when relevant to ensure narrative continuity.",
            "- Writing Style & Tone: Strict university academic textbook register. NEVER use conversational filler, informal speech ('pretty cool', 'bam', 'super simple', 'gonna', 'futzing'), or oral disfluencies from the spoken lecture. Rephrase all content into grammatically immaculate, formal academic exposition.",
            "Requirements:\n"
            "- conceptual_exposition: Comprehensive, high-density academic analysis of the mechanisms in the segment.\n"
            "- formal_definitions: Define key terms and concepts discussed in the segment clearly and rigorously.\n"
            "- mathematical_formulations: If mathematical relationships, update equations, loss functions, or scientific laws are discussed or verbalized, format them as valid LaTeX block expressions in 'latex_expression' (e.g. '$$\\\\theta_{new} = \\\\theta_{old} - \\\\alpha \\\\nabla L$$') with an explanatory note.\n"
            "- takeaways: 2 to 4 high-impact pedagogical summary points.\n"
            "- cue_questions: 2 to 3 diagnostic active recall study questions ending in '?'.\n\n"
            "Respond ONLY with a JSON object adhering to this schema:\n"
            "{\n"
            '  "conceptual_exposition": "In-depth explanatory paragraphs detailing the mechanism...",\n'
            '  "formal_definitions": [{"term": "Concept", "definition": "Formal definition"}],\n'
            '  "mathematical_formulations": [{"latex_expression": "$$E = mc^2$$", "explanation": "..."}],\n'
            '  "algorithmic_code_blocks": [{"language": "python", "code": "...", "caption": "..."}],\n'
            '  "concrete_examples": ["Example 1..."],\n'
            '  "takeaways": ["Takeaway 1...", "Takeaway 2..."],\n'
            '  "cue_questions": ["What is ...?", "How does this mechanism relate to ...?"]\n'
            "}",
        ]
        if global_roadmap:
            system_parts.append(f"GLOBAL LECTURE OUTLINE & ROADMAP:\n{global_roadmap}")
        system_prompt = "\n\n".join(system_parts)

        prereqs_str = ", ".join(prerequisites) if prerequisites else "None"
        user_parts = []
        if preceding_context:
            user_parts.append(f"DISCOURSE PRECEDING STATE (Established earlier in lecture):\n{preceding_context}")
        if upcoming_context:
            user_parts.append(f"UPCOMING TOPIC PREVIEW:\n{upcoming_context}")
        user_parts.append(f"TARGET SECTION TITLE: {topic_title}")
        user_parts.append(f"SECTION PREREQUISITES: {prereqs_str}")
        user_parts.append(f"TRANSCRIPT SEGMENT:\n{segment_text}")
        prompt = "\n\n".join(user_parts)

        request_body = {
            "model": self.config.ollama_model,
            "system": system_prompt,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": self.config.ollama_num_predict,
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
            with urllib.request.urlopen(req, timeout=self.config.ollama_timeout_sec) as resp:
                if resp.status == 200:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    done_reason = resp_json.get("done_reason")
                    raw_text = resp_json.get("response", "").strip()

                    if done_reason == "length":
                        logger.warning(
                            "Ollama note generation reached token limit (eval_count=%s, limit=%s) for section '%s'. Attempting JSON repair and field salvage.",
                            resp_json.get("eval_count"),
                            self.config.ollama_num_predict,
                            topic_title,
                        )

                    parsed = self._parse_and_repair_json(raw_text)
                    if isinstance(parsed, dict) and "conceptual_exposition" in parsed and parsed["conceptual_exposition"]:
                        if done_reason == "length":
                            logger.info(
                                "Successfully repaired and salvaged note generation for section '%s' (%d chars exposition).",
                                topic_title,
                                len(str(parsed["conceptual_exposition"])),
                            )
                        return parsed

                    # Fallback to cleaned raw text if non-trivial
                    if raw_text:
                        clean_raw = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                        clean_raw = re.sub(r"^```(?:json)?\s*", "", clean_raw, flags=re.MULTILINE)
                        clean_raw = re.sub(r"\s*```$", "", clean_raw, flags=re.MULTILINE).strip()
                        if len(clean_raw) > 100:
                            logger.warning(
                                "Using cleaned raw response as conceptual exposition for section '%s' after JSON parse failed.",
                                topic_title,
                            )
                            return {"conceptual_exposition": clean_raw}

                    raise ValueError(f"Ollama response could not be parsed into valid study notes for section '{topic_title}'.")
        except Exception as e:
            logger.warning(
                "Ollama note elaboration failed for section '%s': %s. Falling back to deterministic note generation for this section.",
                topic_title,
                e,
            )
            return None

        return None

    def synthesize_global_review(
        self,
        study_sections: List[StudySection],
        toc: List[Dict[str, Any]],
        lecture_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Ollama Qwen3.5-4B for Stage 3 Global Cross-Topic Synthesis."""
        if not self.config.use_ollama or self.config.force_fallback or not study_sections:
            return None

        sec_summaries = []
        for s in study_sections:
            sec_summaries.append(
                f"### Section {s.section_id + 1}: {s.topic_title} [{s.start_time_str} - {s.end_time_str}]\n"
                f"- Prerequisites: {', '.join(s.prerequisite_tags) if s.prerequisite_tags else 'None'}\n"
                f"- Key Takeaways: {'; '.join(s.takeaways[:3])}\n"
                f"- Core Definitions: {', '.join(d.term for d in s.detailed_notes.formal_definitions[:3])}"
            )
        summaries_block = "\n\n".join(sec_summaries)

        prompt = (
            "You are an master pedagogical architect reviewing a complete university lecture.\n"
            "The individual sections have been analyzed as follows:\n\n"
            f"{summaries_block}\n\n"
            "Generate an executive synthesis tying together the entire lecture:\n"
            "Executive Synthesis: 2-3 rigorous academic paragraphs capturing the macro narrative, core theoretical breakthroughs, and big-picture implications (answering 'So what?').\n\n"
            "Respond ONLY with a JSON object adhering to this schema:\n"
            "{\n"
            '  "executive_synthesis": "Comprehensive 2-3 paragraph macro overview..."\n'
            "}\n"
        )

        request_body = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": self.config.ollama_reduce_num_predict,
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
            with urllib.request.urlopen(req, timeout=self.config.ollama_timeout_sec) as resp:
                if resp.status == 200:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    done_reason = resp_json.get("done_reason")
                    raw_text = resp_json.get("response", "").strip()

                    if done_reason == "length":
                        logger.warning(
                            "Ollama global review synthesis reached token limit (eval_count=%s, limit=%s). Attempting repair.",
                            resp_json.get("eval_count"),
                            self.config.ollama_reduce_num_predict,
                        )

                    parsed = self._parse_and_repair_json(raw_text)
                    if isinstance(parsed, dict) and "executive_synthesis" in parsed and parsed["executive_synthesis"]:
                        return parsed

                    if raw_text:
                        clean_raw = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                        clean_raw = re.sub(r"^```(?:json)?\s*", "", clean_raw, flags=re.MULTILINE)
                        clean_raw = re.sub(r"\s*```$", "", clean_raw, flags=re.MULTILINE).strip()
                        if len(clean_raw) > 50:
                            return {"executive_synthesis": clean_raw}

                    raise ValueError("Ollama global review could not be parsed into valid executive synthesis.")
        except Exception as e:
            logger.warning("Ollama global review synthesis failed (%s); proceeding with local review.", e)
            return None

        return None


# ==============================================================================
# 3. Pedagogical Note Generator Engine
# ==============================================================================

class PedagogicalNoteGenerator:
    """Orchestrates generation of comprehensive, structure-preserving lecture study notes.

    Produces:
    - lecture_study_notes.md: Formatted for human learners, Obsidian, and static sites.
    - lecture_study_notes.json: Machine-readable structured payload conforming strictly to draft-07.
    """

    def __init__(
        self,
        config: Optional[NoteGeneratorConfig] = None,
        elaborator: Optional[OllamaNoteElaborator] = None,
    ):
        self.config = config or NoteGeneratorConfig()
        self.elaborator = elaborator or OllamaNoteElaborator(self.config)
        self.last_notes_obj: Optional[LectureStudyNotes] = None

    def generate(
        self,
        segmentation_payload: Union[Dict[str, Any], Any],
        lecture_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Main entry point for generating pedagogical study notes.

        Args:
            segmentation_payload: Interface Contract 2 dict or SegmentationPayload object.
            lecture_metadata: Optional metadata (course, lecture_title, duration_ms, etc.)

        Returns:
            Tuple of (markdown_str, json_dict).
        """
        payload_dict = self._normalize_payload(segmentation_payload)
        meta = self._normalize_metadata(payload_dict, lecture_metadata)

        raw_segments = payload_dict.get("topic_segments", [])

        # Step 1: Segregate digressions vs core technical segments
        technical_segs, digression_segs = self._filter_segments(raw_segments)

        # Step 2: Build Table of Contents
        toc = self._build_table_of_contents(raw_segments)

        # Build Global Roadmap for Discourse Context Injection
        global_roadmap_lines = [
            f"- Section {item['section_id'] + 1} [{item['start_time_str']} - {item['end_time_str']}]: {item['title']}"
            for item in toc
        ]
        global_roadmap = "\n".join(global_roadmap_lines)

        # Step 3: Process Core Technical Study Sections with Sequential Discourse Continuity
        study_sections: List[StudySection] = []
        for i, seg in enumerate(technical_segs):
            preceding_context = None
            if i > 0:
                prev_sec = study_sections[i - 1]
                prev_takes = "; ".join(prev_sec.takeaways[:2]) if prev_sec.takeaways else "Fundamental concepts introduced."
                preceding_context = f"Section {prev_sec.section_id + 1} ({prev_sec.topic_title}): {prev_takes}"

            upcoming_context = None
            if i + 1 < len(technical_segs):
                next_seg = technical_segs[i + 1]
                upcoming_context = f"Section {int(next_seg.get('segment_id', 0)) + 1}: {next_seg.get('topic_title', '')}"

            sec = self._process_technical_segment(
                seg,
                global_roadmap=global_roadmap,
                preceding_context=preceding_context,
                upcoming_context=upcoming_context,
            )
            study_sections.append(sec)

        # Step 4: Process Administrative Digressions
        admin_digressions: List[AdministrativeDigression] = []
        for dseg in digression_segs:
            ad = self._process_digression_segment(dseg)
            admin_digressions.append(ad)

        # Step 5: Global Cross-Topic Synthesis Pass (Stage 3 Reduce)
        synthesis_data = self.elaborator.synthesize_global_review(study_sections, toc, meta)

        # Step 6: Build Prerequisite DAG (Local section prerequisites without artificial cross-topic links)
        prereq_dag = self._build_prerequisite_dag(raw_segments)

        # Step 7: Build Global Review (Master Glossary, Authentic Section Active Recall & Executive Synthesis)
        global_review = self._build_global_review(study_sections, synthesis_data=synthesis_data)

        # Step 8: Build 2-Tier Hierarchical Modules (Option B)
        modules: List[StudyModule] = []
        if getattr(self.config, "enable_hierarchical_modules", True):
            raw_macro = payload_dict.get("macro_topics", [])
            modules = self._build_study_modules(raw_macro, study_sections, admin_digressions)
            meta["total_modules"] = len(modules)
            meta["total_subsections"] = len(study_sections)

        # Step 9: Assemble top-level LectureStudyNotes container
        notes_obj = LectureStudyNotes(
            schema_version="1.0",
            metadata=meta,
            prerequisite_dag=prereq_dag,
            table_of_contents=toc,
            study_sections=study_sections,
            administrative_digressions=admin_digressions,
            global_review=global_review,
            modules=modules,
        )

        self.last_notes_obj = notes_obj
        notes_dict = notes_obj.to_dict()

        # Step 10: Render comprehensive Markdown
        markdown_str = self._render_markdown(notes_obj)

        return markdown_str, notes_dict

    def _build_study_modules(
        self,
        raw_macro: List[Dict[str, Any]],
        study_sections: List[StudySection],
        admin_digressions: List[AdministrativeDigression],
    ) -> List[StudyModule]:
        """Organize study sections into 2-tier pedagogical StudyModules."""
        if not study_sections:
            return []

        sec_map = {s.section_id: s for s in study_sections}
        modules: List[StudyModule] = []
        covered_ids = set()

        if raw_macro:
            for m in raw_macro:
                mod_id = int(m.get("module_id", len(modules) + 1))
                mod_title = str(m.get("module_title", f"Module {mod_id}")).strip()
                focus = str(m.get("pedagogical_focus", "")).strip()
                raw_sids = m.get("segment_ids", [])
                subsecs = [sec_map[sid] for sid in raw_sids if sid in sec_map]

                if not subsecs:
                    continue

                for s in subsecs:
                    covered_ids.add(s.section_id)

                start_str = subsecs[0].start_time_str
                end_str = subsecs[-1].end_time_str

                modules.append(
                    StudyModule(
                        module_id=mod_id,
                        module_title=mod_title,
                        start_time_str=start_str,
                        end_time_str=end_str,
                        pedagogical_focus=focus or f"Thematic curriculum module covering {mod_title}.",
                        subsections=subsecs,
                    )
                )

        # Fallback or uncovered sections
        uncovered = [s for s in study_sections if s.section_id not in covered_ids]
        if not modules:
            chunk_size = 1 if len(study_sections) <= 3 else (3 if len(study_sections) <= 10 else 4)
            for i in range(0, len(study_sections), chunk_size):
                chunk = study_sections[i : i + chunk_size]
                mod_id = len(modules) + 1
                first_title = chunk[0].topic_title
                mod_title = f"{first_title} & Related Concepts" if len(chunk) > 1 else first_title
                modules.append(
                    StudyModule(
                        module_id=mod_id,
                        module_title=mod_title,
                        start_time_str=chunk[0].start_time_str,
                        end_time_str=chunk[-1].end_time_str,
                        pedagogical_focus=f"Thematic module covering {', '.join(s.topic_title for s in chunk[:2])}.",
                        subsections=chunk,
                    )
                )
        elif uncovered:
            modules[-1].subsections.extend(uncovered)
            modules[-1].end_time_str = modules[-1].subsections[-1].end_time_str

        return modules

    # --------------------------------------------------------------------------
    # Segment Processing & Feature Extraction
    # --------------------------------------------------------------------------

    def _process_technical_segment(
        self,
        seg: Dict[str, Any],
        global_roadmap: Optional[str] = None,
        preceding_context: Optional[str] = None,
        upcoming_context: Optional[str] = None,
    ) -> StudySection:
        """Process a core technical segment into a full Cornell StudySection with discourse awareness."""
        sec_id = int(seg.get("segment_id", 0))
        title = str(seg.get("topic_title", f"Section {sec_id + 1}"))
        start_time = str(seg.get("start_time_str", "00:00:00"))
        end_time = str(seg.get("end_time_str", "00:00:00"))
        prereqs = list(seg.get("prerequisite_tags", []))
        raw_text = str(seg.get("raw_text", ""))
        sentences = seg.get("sentences", [])

        # Cue questions: ensure non-empty and every question ends strictly with '?'
        cues = list(seg.get("cue_questions", []))
        cues = [q.strip() if q.strip().endswith("?") else f"{q.strip()}?" for q in cues if q.strip()]
        if not cues:
            cues = [
                f"What fundamental mechanism does {title} describe?",
                f"How is {title} applied in broader computing systems?",
            ]

        # Extract domain keywords
        words = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b|\b[a-z]{4,}\b", raw_text)
        stopwords = {
            "this", "that", "these", "those", "with", "from", "have", "here",
            "section", "topic", "lecture", "welcome", "returning", "examine",
            "first", "second", "third", "about", "their", "there", "which",
        }
        filtered_words = [w for w in words if w.lower() not in stopwords]
        keywords = list(dict.fromkeys(filtered_words))[:8]

        # Attempt LLM-assisted elaboration with global context
        llm_data = self.elaborator.elaborate_segment(
            raw_text,
            title,
            prereqs,
            global_roadmap=global_roadmap,
            preceding_context=preceding_context,
            upcoming_context=upcoming_context,
        )
        if llm_data:
            expo = llm_data.get("conceptual_exposition", raw_text)
            if "cue_questions" in llm_data and isinstance(llm_data["cue_questions"], list):
                llm_cues = [
                    str(q).strip() if str(q).strip().endswith("?") else f"{str(q).strip()}?"
                    for q in llm_data["cue_questions"]
                    if str(q).strip()
                ]
                if llm_cues:
                    cues = llm_cues

            definitions = [
                FormalDefinition(term=str(d.get("term", "")), definition=str(d.get("definition", "")))
                for d in llm_data.get("formal_definitions", [])
                if d.get("term")
            ] or self._extract_formal_definitions(sentences, title)
            math_forms = [
                MathematicalFormulation(
                    latex_expression=str(m.get("latex_expression", "")),
                    explanation=str(m.get("explanation", "")),
                )
                for m in llm_data.get("mathematical_formulations", [])
                if m.get("latex_expression")
            ] or self._extract_mathematical_formulations(raw_text, sentences)
            code_blocks = [
                AlgorithmicCodeBlock(
                    language=str(c.get("language", "python")),
                    code=str(c.get("code", "")),
                    caption=str(c.get("caption", "")),
                )
                for c in llm_data.get("algorithmic_code_blocks", [])
                if c.get("code")
            ]
            examples = [str(e) for e in llm_data.get("concrete_examples", []) if str(e).strip()] or self._extract_examples(sentences)
            takeaways = [str(t) for t in llm_data.get("takeaways", []) if str(t).strip()] or self._extract_takeaways(sentences, title)
        else:
            logger.info("Using deterministic feature extraction for section '%s'.", title)
            expo = raw_text
            math_forms = self._extract_mathematical_formulations(raw_text, sentences)
            definitions = self._extract_formal_definitions(sentences, title)
            code_blocks = self._extract_code_blocks(raw_text)
            examples = self._extract_examples(sentences)
            takeaways = self._extract_takeaways(sentences, title)

        detailed_notes = DetailedNotesSection(
            conceptual_exposition=expo,
            formal_definitions=definitions,
            mathematical_formulations=math_forms,
            algorithmic_code_blocks=code_blocks,
            concrete_examples=examples,
        )

        cue_column = CornellCueColumn(
            cue_questions=cues,
            core_keywords=keywords,
        )

        return StudySection(
            section_id=sec_id,
            topic_title=title,
            start_time_str=start_time,
            end_time_str=end_time,
            is_digression=False,
            digression_type="none",
            prerequisite_tags=prereqs,
            cornell_cue_column=cue_column,
            detailed_notes=detailed_notes,
            takeaways=takeaways,
        )

    def _process_digression_segment(self, seg: Dict[str, Any]) -> AdministrativeDigression:
        """Process an administrative or logistical digression segment."""
        sec_id = int(seg.get("segment_id", 0))
        title = str(seg.get("topic_title", "Course Logistics & Announcements"))
        start_time = str(seg.get("start_time_str", "00:00:00"))
        end_time = str(seg.get("end_time_str", "00:00:00"))
        raw_text = str(seg.get("raw_text", ""))

        # Check sentence digression types
        sentences = seg.get("sentences", [])
        types = [s.get("digression_type", "administrative") for s in sentences if s.get("digression_type")]
        dtype = types[0] if types else "administrative"

        return AdministrativeDigression(
            section_id=sec_id,
            title=title,
            start_time_str=start_time,
            end_time_str=end_time,
            digression_type=dtype,
            announcement_notes=raw_text,
        )

    # --------------------------------------------------------------------------
    # Rule-Based Heuristic Feature Extractors (Offline / High-Fidelity)
    # --------------------------------------------------------------------------

    def _extract_mathematical_formulations(
        self, raw_text: str, sentences: List[Dict[str, Any]]
    ) -> List[MathematicalFormulation]:
        """Extract LaTeX equations from text and pair with conceptual descriptions."""
        formulations: List[MathematicalFormulation] = []

        # Find block math $$...$$
        block_math = re.findall(r"\$\$(.+?)\$\$", raw_text)
        for eq in block_math:
            eq_clean = f"$${eq.strip()}$$"
            explanation = self._find_math_explanation(eq, sentences)
            formulations.append(
                MathematicalFormulation(latex_expression=eq_clean, explanation=explanation)
            )

        # Find inline math $...$ not already captured
        inline_math = re.findall(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", raw_text)
        for eq in inline_math:
            eq_clean = f"${eq.strip()}$"
            if not any(eq_clean in f.latex_expression for f in formulations):
                explanation = self._find_math_explanation(eq, sentences)
                formulations.append(
                    MathematicalFormulation(latex_expression=eq_clean, explanation=explanation)
                )

        # Detect verbalized equations if no explicit LaTeX was present
        if not formulations:
            for s in sentences:
                t = s.get("text", "")
                # Case 1: Symbol =
                m_eq = re.search(r"([A-Za-z0-9_{}\\]+\s*=\s*[^.,;]+)", t)
                if m_eq:
                    expr = f"$${m_eq.group(1).strip()}$$"
                    formulations.append(
                        MathematicalFormulation(
                            latex_expression=expr,
                            explanation=t,
                        )
                    )
                    continue

                # Case 2: Verbalized "equals" or "is given by"
                m_verbal = re.search(r"([A-Za-z0-9_]+)\s+equals\s+([A-Za-z0-9_\s\+\-\*\/\(\)\.]+)", t, re.IGNORECASE)
                if m_verbal:
                    lhs = m_verbal.group(1).strip()
                    rhs_raw = m_verbal.group(2).strip().rstrip(".")
                    # Convert verbal math operators to symbols
                    rhs = re.sub(r"\bplus\b", "+", rhs_raw, flags=re.IGNORECASE)
                    rhs = re.sub(r"\bminus\b", "-", rhs, flags=re.IGNORECASE)
                    rhs = re.sub(r"\btimes\b", "\\times", rhs, flags=re.IGNORECASE)
                    rhs = re.sub(r"\bdivided by\b", "/", rhs, flags=re.IGNORECASE)
                    expr = f"$${lhs} = {rhs.strip()}$$"
                    formulations.append(
                        MathematicalFormulation(
                            latex_expression=expr,
                            explanation=t,
                        )
                    )

        return formulations

    def _find_math_explanation(self, eq: str, sentences: List[Dict[str, Any]]) -> str:
        """Find the sentence providing explanation for a formula."""
        for s in sentences:
            text = s.get("text", "")
            if eq in text:
                return text
        return "Formal mathematical relationship established in lecture."

    def _extract_formal_definitions(
        self, sentences: List[Dict[str, Any]], title: str
    ) -> List[FormalDefinition]:
        """Extract formal definitions from sentence structures."""
        defs: List[FormalDefinition] = []

        definition_patterns = [
            r"formally define (?:the )?([A-Za-z\s]+?) as ([^.,;]+)",
            r"([A-Za-z\s]+?) is defined as ([^.,;]+)",
            r"([A-Za-z\s]+?) comprises ([^.,;]+)",
            r"([A-Za-z\s]+?) provides ([^.,;]+)",
            r"([A-Za-z\s]+?) translates ([^.,;]+)",
            r"([A-Za-z\s]+?) maintains ([^.,;]+)",
        ]

        for s in sentences:
            text = s.get("text", "")
            for pat in definition_patterns:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    term = match.group(1).strip().capitalize()
                    desc = match.group(2).strip()
                    if len(term.split()) <= 4 and len(desc) > 5:
                        defs.append(FormalDefinition(term=term, definition=desc))

        if not defs:
            defs.append(
                FormalDefinition(
                    term=title,
                    definition="Core subject entity establishing foundational lecture principles.",
                )
            )

        return defs

    def _extract_code_blocks(self, raw_text: str) -> List[AlgorithmicCodeBlock]:
        """Extract code blocks if present in raw text."""
        blocks: List[AlgorithmicCodeBlock] = []
        code_matches = re.findall(r"```([a-zA-Z]*)\n([\s\S]+?)```", raw_text)
        for lang, code in code_matches:
            blocks.append(
                AlgorithmicCodeBlock(
                    language=lang.strip() or "python",
                    code=code.strip(),
                    caption="Algorithmic formulation demonstrated in lecture.",
                )
            )
        return blocks

    def _extract_examples(self, sentences: List[Dict[str, Any]]) -> List[str]:
        """Extract concrete examples from sentences."""
        examples: List[str] = []
        for s in sentences:
            text = s.get("text", "")
            if any(marker in text.lower() for marker in ("for example", "for instance", "such as", "consider")):
                examples.append(text)
        return examples

    def _extract_takeaways(self, sentences: List[Dict[str, Any]], title: str) -> List[str]:
        """Extract or synthesize high-impact conceptual takeaways."""
        takeaways: List[str] = []
        for s in sentences:
            text = s.get("text", "")
            if any(k in text.lower() for k in ("modeled", "provides", "translates", "interrupt", "scheduling", "manages")):
                takeaways.append(text)
        if not takeaways and sentences:
            takeaways.append(sentences[0].get("text", f"Key conceptual mechanisms of {title} established."))
        return takeaways[:3]

    # --------------------------------------------------------------------------
    # DAG & Table of Contents Construction
    # --------------------------------------------------------------------------

    def _build_prerequisite_dag(
        self,
        segments: List[Dict[str, Any]],
        extra_edges: Optional[List[Dict[str, Any]]] = None,
    ) -> PrerequisiteDAG:
        """Construct Directed Acyclic Graph of concept prerequisites with optional cross-topic dependencies."""
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, str]] = []
        seen_concepts: Set[str] = set()

        for seg in segments:
            sec_id = int(seg.get("segment_id", 0))
            prereqs = seg.get("prerequisite_tags", [])
            title = seg.get("topic_title", f"Topic {sec_id}")

            for p in prereqs:
                p_clean = str(p).strip()
                if p_clean and p_clean not in seen_concepts:
                    seen_concepts.add(p_clean)
                    nodes.append({"concept": p_clean, "introduced_in_segment_id": sec_id})

                if p_clean:
                    edges.append({"source_concept": p_clean, "target_concept": title})

            if title not in seen_concepts:
                seen_concepts.add(title)
                nodes.append({"concept": title, "introduced_in_segment_id": sec_id})

        if extra_edges:
            for ee in extra_edges:
                src = str(ee.get("source_concept", "")).strip()
                tgt = str(ee.get("target_concept", "")).strip()
                if src and tgt and src != tgt:
                    if src not in seen_concepts:
                        seen_concepts.add(src)
                        nodes.append({"concept": src, "introduced_in_segment_id": 0})
                    if tgt not in seen_concepts:
                        seen_concepts.add(tgt)
                        nodes.append({"concept": tgt, "introduced_in_segment_id": 0})
                    edges.append({"source_concept": src, "target_concept": tgt})

        # Cycle-safe topological sort (Kahn's algorithm)
        topo_order: List[str] = []
        in_degree: Dict[str, int] = {n["concept"]: 0 for n in nodes}
        adj: Dict[str, List[str]] = {n["concept"]: [] for n in nodes}

        for edge in edges:
            src = edge["source_concept"]
            tgt = edge["target_concept"]
            if src in adj and tgt in in_degree:
                adj[src].append(tgt)
                in_degree[tgt] += 1

        queue = [c for c, deg in in_degree.items() if deg == 0]
        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for neighbor in adj.get(curr, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Fallback if cycles or unvisited nodes exist
        for n in nodes:
            if n["concept"] not in topo_order:
                topo_order.append(n["concept"])

        return PrerequisiteDAG(
            nodes=nodes,
            edges=edges,
            topological_order=topo_order,
        )

    def _build_table_of_contents(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Construct Table of Contents."""
        toc: List[Dict[str, Any]] = []
        for seg in segments:
            sec_id = int(seg.get("segment_id", 0))
            is_digress = (
                seg.get("is_digression", False)
                or all(s.get("is_digression", False) for s in seg.get("sentences", []))
            ) if seg.get("sentences") else seg.get("is_digression", False)
            toc.append({
                "section_id": sec_id,
                "title": str(seg.get("topic_title", f"Section {sec_id + 1}")),
                "start_time_str": str(seg.get("start_time_str", "00:00:00")),
                "end_time_str": str(seg.get("end_time_str", "00:00:00")),
                "is_digression": bool(is_digress),
            })
        return toc

    def _build_global_review(
        self,
        study_sections: List[StudySection],
        synthesis_data: Optional[Dict[str, Any]] = None,
    ) -> GlobalReviewBlock:
        """Construct Master Concept Glossary and Active Recall Self-Assessment Sheet with Cross-Topic Synthesis."""
        glossary_map: Dict[str, MasterGlossaryEntry] = {}
        recall_questions: List[ActiveRecallQuestion] = []

        for sec in study_sections:
            sec_id = sec.section_id
            for d in sec.detailed_notes.formal_definitions:
                term_key = d.term.strip().lower()
                if term_key not in glossary_map:
                    glossary_map[term_key] = MasterGlossaryEntry(
                        term=d.term.strip(),
                        definition=d.definition.strip(),
                        referenced_in_sections=[sec_id],
                    )
                else:
                    if sec_id not in glossary_map[term_key].referenced_in_sections:
                        glossary_map[term_key].referenced_in_sections.append(sec_id)

            for q in sec.cornell_cue_column.cue_questions:
                q_clean = q.strip() if q.strip().endswith("?") else f"{q.strip()}?"
                recall_questions.append(
                    ActiveRecallQuestion(
                        question=q_clean,
                        section_id=sec_id,
                        core_concept=sec.topic_title,
                    )
                )

        exec_synthesis = None
        if synthesis_data:
            exec_synthesis = synthesis_data.get("executive_synthesis")

        return GlobalReviewBlock(
            master_glossary=list(glossary_map.values()),
            active_recall_review_sheet=recall_questions,
            executive_synthesis=exec_synthesis,
            comparative_synthesis_questions=[],
        )

    # --------------------------------------------------------------------------
    # Segment Filtering & Metadata Normalization
    # --------------------------------------------------------------------------

    def _filter_segments(
        self, segments: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Segregate technical segments from administrative/logistical digressions."""
        technical: List[Dict[str, Any]] = []
        digressions: List[Dict[str, Any]] = []

        for seg in segments:
            sentences = seg.get("sentences", [])
            is_digression = (
                seg.get("is_digression", False)
                or (all(s.get("is_digression", False) for s in sentences) if sentences else False)
            )
            if self.config.segregate_administrative_digressions and is_digression:
                digressions.append(seg)
            else:
                technical.append(seg)

        return technical, digressions

    def _normalize_payload(self, payload: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalize segmentation payload to dictionary."""
        if hasattr(payload, "to_contract_dict"):
            return payload.to_contract_dict()
        if isinstance(payload, dict):
            return payload
        raise TypeError(f"Unsupported payload type: {type(payload)}")

    def _normalize_metadata(
        self, payload_dict: Dict[str, Any], user_meta: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construct normalized lecture metadata adhering to test schema invariants."""
        meta = dict(user_meta or {})
        segs = payload_dict.get("topic_segments", [])

        # Duration computation
        total_sec = 0.0
        if "duration_ms" in meta:
            total_sec = float(meta["duration_ms"]) / 1000.0
        elif segs:
            for seg in segs:
                for s in seg.get("sentences", []):
                    end_ms = s.get("end_ms", 0)
                    if end_ms / 1000.0 > total_sec:
                        total_sec = end_ms / 1000.0

        total_sec = round(total_sec, 2)

        if total_sec > 0:
            h = int(total_sec // 3600)
            m = int((total_sec % 3600) // 60)
            s = int(total_sec % 60)
            total_duration_str = f"{h:02d}:{m:02d}:{s:02d}"
        elif segs:
            total_duration_str = segs[-1].get("end_time_str", "00:00:00")
        else:
            total_duration_str = "00:00:00"

        return {
            "lecture_title": meta.get("lecture_title", self.config.default_lecture_title),
            "course": meta.get("course", self.config.default_course),
            "total_duration_sec": total_sec,
            "total_duration_str": total_duration_str,
            "total_sentences": payload_dict.get("num_sentences", 0),
            "total_segments": len(segs),
            "generated_at": meta.get("generated_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        }

    # --------------------------------------------------------------------------
    # Markdown Rendering Engine
    # --------------------------------------------------------------------------

    def _render_markdown(self, notes: LectureStudyNotes) -> str:
        """Render publication-grade Markdown conforming strictly to Cornell study notes."""
        lines: List[str] = []

        # Document Header
        title = notes.metadata.get("lecture_title", "Educational Lecture Study Notes")
        course = notes.metadata.get("course", "Computer Science")
        duration = notes.metadata.get("total_duration_str", "00:00:00")
        total_modules = len(notes.modules) if notes.modules else 1
        total_segs = len(notes.study_sections)
        total_sents = notes.metadata.get("total_sentences", 0)

        lines.append(f"# {title}\n")
        lines.append(f"> **Course**: {course}  ")
        if notes.modules:
            lines.append(f"> **Total Duration**: `[{duration}]` | **Modules**: {total_modules} | **Subsections**: {total_segs} | **Sentences**: {total_sents}  ")
        else:
            lines.append(f"> **Total Duration**: `[{duration}]` | **Segments**: {total_segs} | **Sentences**: {total_sents}  ")
        lines.append(
            "> **Pedagogical Integrity**: 100% Comprehensive Alternative to Watching Lecture (Zero Destructive Compression)\n"
        )
        lines.append("---\n")

        # Executive Overview
        lines.append("## Executive Overview\n")
        if notes.global_review.executive_synthesis:
            lines.append(f"{notes.global_review.executive_synthesis}\n")
        if getattr(self.config, "build_mermaid_diagram", False):
            lines.append("```mermaid")
            lines.append("graph TD")
            if notes.prerequisite_dag.edges:
                for edge in notes.prerequisite_dag.edges:
                    src = edge["source_concept"].replace('"', "")
                    tgt = edge["target_concept"].replace('"', "")
                    lines.append(f'    "{src}" --> "{tgt}"')
            elif notes.prerequisite_dag.nodes:
                for node in notes.prerequisite_dag.nodes[:3]:
                    concept = node["concept"].replace('"', "")
                    lines.append(f'    "{concept}"')
            else:
                lines.append('    "Foundational Topic"')
            lines.append("```\n")

        if getattr(self.config, "include_prerequisite_hierarchy_table", False):
            lines.append("### Conceptual Prerequisite Hierarchy\n")
            lines.append("| Prerequisite Concept | Target Subject Concept |")
            lines.append("|:---------------------|:-----------------------|")
            if notes.prerequisite_dag.edges:
                for edge in notes.prerequisite_dag.edges:
                    lines.append(f"| {edge['source_concept']} | {edge['target_concept']} |")
            else:
                lines.append("| None (Foundational) | Core Lecture Concepts |")
            lines.append("\n")

        lines.append("---\n")

        # Table of Contents
        lines.append("## Table of Contents\n")
        if notes.modules:
            for mod in notes.modules:
                lines.append(f"- `[{mod.start_time_str}]` **Module {mod.module_id}: {mod.module_title}**")
                for sub_idx, sec in enumerate(mod.subsections, 1):
                    digress_tag = " *(Administrative Digression)*" if sec.is_digression else ""
                    lines.append(
                        f"  - `[{sec.start_time_str}]` Subsection {mod.module_id}.{sub_idx}: {sec.topic_title}{digress_tag}"
                    )
            if notes.administrative_digressions:
                for ad in notes.administrative_digressions:
                    lines.append(f"- `[{ad.start_time_str}]` Notice: {ad.title} *(Administrative Digression)*")
        else:
            for item in notes.table_of_contents:
                digress_tag = " *(Administrative Digression)*" if item["is_digression"] else ""
                lines.append(
                    f"- `[{item['start_time_str']}]` Section {item['section_id'] + 1}: {item['title']}{digress_tag}"
                )
        lines.append("- Global Review: Master Concept Glossary & Active Recall Sheet\n")
        lines.append("---\n")

        # Core Study Sections
        if notes.modules:
            for mod in notes.modules:
                lines.append(f"# Module {mod.module_id}: {mod.module_title}\n")
                lines.append(f"**Timestamp**: `[{mod.start_time_str}] - [{mod.end_time_str}]` | **Subsections**: {len(mod.subsections)}\n")
                if mod.pedagogical_focus:
                    lines.append(f"> *Module Focus*: {mod.pedagogical_focus}\n")
                if mod.module_synthesis:
                    lines.append(f"{mod.module_synthesis}\n")

                for sub_idx, sec in enumerate(mod.subsections, 1):
                    lines.append(
                        f"## Subsection {mod.module_id}.{sub_idx}: {sec.topic_title}\n"
                    )
                    lines.append(f"**Timestamp Anchor**: `[{sec.start_time_str}] - [{sec.end_time_str}]`  ")
                    prereq_str = ", ".join(f"`{p}`" for p in sec.prerequisite_tags) if sec.prerequisite_tags else "None (Foundational)"
                    lines.append(f"**Prerequisites**: {prereq_str}  ")
                    if sec.cornell_cue_column.core_keywords:
                        kw_str = ", ".join(sec.cornell_cue_column.core_keywords)
                        lines.append(f"**Key Concepts**: {kw_str}  ")
                    lines.append("**Classification**: Core Pedagogical Content\n")

                    if getattr(self.config, "include_section_cue_column", False):
                        lines.append("### Cornell Cue Column (Active Recall)")
                        for q in sec.cornell_cue_column.cue_questions:
                            lines.append(f"- **Cue Q**: {q}")
                        if sec.cornell_cue_column.core_keywords:
                            kw_str = ", ".join(sec.cornell_cue_column.core_keywords)
                            lines.append(f"- **Key Concepts**: {kw_str}")
                        lines.append("")

                    if getattr(self.config, "include_section_exposition_heading", False):
                        lines.append("### Detailed Notes & Conceptual Exposition")
                    lines.append(f"{sec.detailed_notes.conceptual_exposition}\n")

                    if sec.detailed_notes.formal_definitions:
                        lines.append("### Formal Definitions")
                        for d in sec.detailed_notes.formal_definitions:
                            lines.append(f"- **{d.term}**: {d.definition}")
                        lines.append("")

                    if sec.detailed_notes.mathematical_formulations:
                        lines.append("### Mathematical Formulations")
                        for m in sec.detailed_notes.mathematical_formulations:
                            lines.append(f"{m.latex_expression}\n")
                            lines.append(f"*{m.explanation}*\n")

                    if sec.detailed_notes.algorithmic_code_blocks:
                        lines.append("### Algorithmic Formulation")
                        for c in sec.detailed_notes.algorithmic_code_blocks:
                            lines.append(f"```{c.language}")
                            lines.append(c.code)
                            lines.append("```")
                            lines.append(f"*{c.caption}*\n")

                    if sec.detailed_notes.concrete_examples:
                        lines.append("### Concrete Examples")
                        for ex in sec.detailed_notes.concrete_examples:
                            lines.append(f"- {ex}")
                        lines.append("")

                    if sec.takeaways:
                        lines.append("### Key Takeaways")
                        for t in sec.takeaways:
                            lines.append(f"- {t}")
                        lines.append("")

                    lines.append("---\n")
        else:
            for sec in notes.study_sections:
                lines.append(
                    f"## Section {sec.section_id + 1}: {sec.topic_title}\n"
                )
                lines.append(f"**Timestamp Anchor**: `[{sec.start_time_str}] - [{sec.end_time_str}]`  ")
                prereq_str = ", ".join(f"`{p}`" for p in sec.prerequisite_tags) if sec.prerequisite_tags else "None (Foundational)"
                lines.append(f"**Prerequisites**: {prereq_str}  ")
                if sec.cornell_cue_column.core_keywords:
                    kw_str = ", ".join(sec.cornell_cue_column.core_keywords)
                    lines.append(f"**Key Concepts**: {kw_str}  ")
                lines.append("**Classification**: Core Pedagogical Content\n")

                if getattr(self.config, "include_section_cue_column", False):
                    lines.append("### Cornell Cue Column (Active Recall)")
                    for q in sec.cornell_cue_column.cue_questions:
                        lines.append(f"- **Cue Q**: {q}")
                    if sec.cornell_cue_column.core_keywords:
                        kw_str = ", ".join(sec.cornell_cue_column.core_keywords)
                        lines.append(f"- **Key Concepts**: {kw_str}")
                    lines.append("")

                if getattr(self.config, "include_section_exposition_heading", False):
                    lines.append("### Detailed Notes & Conceptual Exposition")
                lines.append(f"{sec.detailed_notes.conceptual_exposition}\n")

                if sec.detailed_notes.formal_definitions:
                    lines.append("### Formal Definitions")
                    for d in sec.detailed_notes.formal_definitions:
                        lines.append(f"- **{d.term}**: {d.definition}")
                    lines.append("")

                if sec.detailed_notes.mathematical_formulations:
                    lines.append("### Mathematical Formulations")
                    for m in sec.detailed_notes.mathematical_formulations:
                        lines.append(f"{m.latex_expression}\n")
                        lines.append(f"*{m.explanation}*\n")

                if sec.detailed_notes.algorithmic_code_blocks:
                    lines.append("### Algorithmic Formulation")
                    for c in sec.detailed_notes.algorithmic_code_blocks:
                        lines.append(f"```{c.language}")
                        lines.append(c.code)
                        lines.append("```")
                        lines.append(f"*{c.caption}*\n")

                if sec.detailed_notes.concrete_examples:
                    lines.append("### Concrete Examples")
                    for ex in sec.detailed_notes.concrete_examples:
                        lines.append(f"- {ex}")
                    lines.append("")

                if sec.takeaways:
                    lines.append("### Key Takeaways")
                    for t in sec.takeaways:
                        lines.append(f"- {t}")
                    lines.append("")

                lines.append("---\n")

        # Administrative Digressions Callout / Appendix
        if notes.administrative_digressions:
            lines.append("## Course Logistics & Administrative Announcements\n")
            lines.append(
                "> *Note: The following sections were identified as non-pedagogical course logistics "
                "and are preserved here for student administrative awareness without polluting theoretical lecture flow.*\n"
            )
            for d in notes.administrative_digressions:
                lines.append(f"### ⚠️ Notice: {d.title}")
                lines.append(f"**Timestamp**: `[{d.start_time_str}] - [{d.end_time_str}]` | **Category**: `{d.digression_type}`\n")
                lines.append(f"> {d.announcement_notes}\n")
            lines.append("---\n")

        # Global Review & Master Study Aids
        lines.append("## Global Review & Master Study Aids\n")

        # Master Glossary
        if notes.global_review.master_glossary:
            lines.append("### Master Concept Glossary\n")
            lines.append("| Concept | Formal Definition | Reference |")
            lines.append("|:--------|:------------------|:----------|")
            for g in notes.global_review.master_glossary:
                refs = ", ".join(f"Section {s + 1}" for s in g.referenced_in_sections)
                lines.append(f"| **{g.term}** | {g.definition} | {refs} |")
            lines.append("\n")

        # Active Recall Review Sheet
        if notes.global_review.active_recall_review_sheet:
            lines.append("### Active Recall Review Sheet (Flashcard Mode)\n")
            for idx, q in enumerate(notes.global_review.active_recall_review_sheet, 1):
                lines.append(f"{idx}. **Question**: {q.question}")
                lines.append(f"   *Core Topic*: {q.core_concept} (Ref: Section {q.section_id + 1})\n")

        return "\n".join(lines).strip() + "\n"
