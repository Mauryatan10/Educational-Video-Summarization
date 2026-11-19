"""Phase 1 Acoustic Grounding, Disfluency Scrubbing & Digression Gating Package.

Conforms to Milestone 1 specifications and Interface Contract 1.
"""

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

__all__ = [
    "ASRPipeline",
    "ASRPipelineConfig",
    "SentenceRecord",
    "SileroVADChunker",
    "AudioChunk",
    "VADChunkConfig",
    "WhisperTranscriber",
    "WhisperXAligner",
    "TranscribeAlignPipeline",
    "TranscribedWord",
    "AlignedSentence",
    "ASRConfig",
    "DisfluencyFilter",
    "DisfluencyFilterConfig",
    "SemanticDigressionGate",
    "DigressionGateConfig",
    "LightweightTfidfEmbedder",
]
