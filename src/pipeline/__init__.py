"""src/pipeline/__init__.py - Unified Educational Video Lecture Processing & Note Generation Pipeline."""

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
    NoteGeneratorConfig,
    OllamaNoteElaborator,
    PedagogicalNoteGenerator,
    PrerequisiteDAG,
    StudySection,
)
from src.pipeline.run_e2e_lecture import (
    E2ELecturePipeline,
    E2EPipelineConfig,
    HardwareMemoryManager,
    PipelineResult,
    TranscriptLoader,
    build_arg_parser,
    main,
)

__all__ = [
    # Orchestrator & CLI
    "E2ELecturePipeline",
    "E2EPipelineConfig",
    "PipelineResult",
    "HardwareMemoryManager",
    "TranscriptLoader",
    "build_arg_parser",
    "main",
    # Note Generator & Data Models
    "PedagogicalNoteGenerator",
    "NoteGeneratorConfig",
    "OllamaNoteElaborator",
    "LectureStudyNotes",
    "StudySection",
    "AdministrativeDigression",
    "PrerequisiteDAG",
    "MasterGlossaryEntry",
    "ActiveRecallQuestion",
    "GlobalReviewBlock",
    "DetailedNotesSection",
    "CornellCueColumn",
    "FormalDefinition",
    "MathematicalFormulation",
    "AlgorithmicCodeBlock",
]
