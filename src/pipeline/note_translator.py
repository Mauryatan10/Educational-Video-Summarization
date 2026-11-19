"""src/pipeline/note_translator.py - Pedagogical Study Notes Hindi Translation Module.

Ingests publication-grade Cornell study notes (LectureStudyNotes dataclass / Markdown)
and generates parallel, schema-identical Hindi counterparts:
1. lecture_study_notes_hi.md (Publication-grade Hindi Cornell study notes)
2. lecture_study_notes_hi.json (Machine-readable, schema-validated structured Hindi notes)

Key Capabilities:
- Powered by fine-tuned sarvamai/sarvam-1 Step 200 champion checkpoint (4-bit NF4 QLoRA).
- Mathematical and algorithmic code preservation via placeholder token masking:
  protects inline ($...$) and display ($$...$$) KaTeX math and code blocks verbatim.
- Subtopic-aware discourse chunking matching the 200-320 word training distribution.
- Factual deterministic decoding (greedy, repetition_penalty=1.15, length_penalty=1.1).
- Sequential memory management: clean load and unload with torch.cuda.empty_cache().
"""
from __future__ import annotations

import copy
import gc
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "sarvamai/sarvam-1"
DEFAULT_ADAPTER_PATH = "models/sarvam1_subtopic_script_constrained/best_checkpoint"
DEFAULT_TRANSLATION_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TRANSLATION_OLLAMA_MODEL = "sarvam1-hi-step200"


# ==============================================================================
# 1. Mathematical & Code Placeholder Masker
# ==============================================================================

class PlaceholderMasker:
    """Safely masks LaTeX equations, code blocks, and markdown spans prior to translation.

    Uses non-lexical numeric bracket tokens (e.g. ``[[0]]``, ``[[1]]``) instead of
    English words like ``MATH`` so Devanagari-constrained LLMs (Sarvam-1) do not
    attempt to translate or paraphrase placeholder names into ``Math_(math_)`` or
    ``गणित()``. Supports whitespace, single-bracket, and Devanagari-digit tolerance
    plus missing-placeholder detection for guaranteed verbatim recovery.
    """

    # Multi-line code blocks: ```lang ... ```
    CODE_BLOCK_RE = re.compile(r"(```[a-zA-Z0-9_\-]*\n[\s\S]*?```)")
    # Display math blocks: $$ ... $$
    DISPLAY_MATH_RE = re.compile(r"(\$\$[\s\S]*?\$\$)")
    # Inline code: `...`
    INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")
    # Inline math: $...$ (with negative lookbehind for escaped \$)
    INLINE_MATH_RE = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
    # Placeholder token splitter
    PLACEHOLDER_TOKEN_RE = re.compile(r"(\[\[\d+\]\])")

    _DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

    def mask(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replaces syntax-critical blocks with script-neutral numeric placeholders."""
        if not text:
            return "", {}

        mapping: Dict[str, str] = {}
        counter = 0

        def _replace_block(match: re.Match[str]) -> str:
            nonlocal counter
            key = f"[[{counter}]]"
            mapping[key] = match.group(0)
            counter += 1
            return key

        # Order matters: multi-line blocks must precede inline blocks
        masked = self.CODE_BLOCK_RE.sub(_replace_block, text)
        masked = self.DISPLAY_MATH_RE.sub(_replace_block, masked)
        masked = self.INLINE_CODE_RE.sub(_replace_block, masked)
        masked = self.INLINE_MATH_RE.sub(_replace_block, masked)

        return masked, mapping

    def _build_key_patterns(self, key: str) -> List[re.Pattern[str]]:
        """Build ordered regex patterns (strict -> tolerant) matching a placeholder key."""
        m_num = re.fullmatch(r"\[\[(\d+)\]\]", key)
        if m_num:
            idx = m_num.group(1)
            # Match ASCII or Devanagari digits for this index
            dev_idx = idx.translate(str.maketrans("0123456789", "०१२३४५६७८९"))
            digit_pat = f"(?:{re.escape(idx)}|{re.escape(dev_idx)})"
            return [
                # Exact or spaced double brackets: [[0]], [[ 0 ]], [ [ 0 ] ]
                re.compile(rf"\[\s*\[\s*{digit_pat}\s*\]\s*\]"),
                # Single-bracket degradation: [0] or [ 0 ] (not adjacent to another bracket)
                re.compile(rf"(?<!\[)\[\s*{digit_pat}\s*\](?!\])"),
            ]

        # Backward compatibility for legacy __PREFIX_IDX__ keys
        parts = key.strip("_").split("_")
        prefix = parts[0]
        idx = parts[1] if len(parts) > 1 else "0"
        return [re.compile(rf"__\s*{re.escape(prefix)}\s*_\s*{re.escape(idx)}\s*__", re.IGNORECASE)]

    def unmask_with_report(self, text: str, mapping: Dict[str, str]) -> Tuple[str, List[str]]:
        """Restore original verbatim blocks and return (unmasked_text, list_of_missing_keys)."""
        if not text or not mapping:
            return text or "", []

        unmasked = text
        missing: List[str] = []

        for key, val in mapping.items():
            matched = False
            for pattern in self._build_key_patterns(key):
                if pattern.search(unmasked):
                    unmasked = pattern.sub(lambda _: val, unmasked)
                    matched = True
                    break
            if not matched:
                missing.append(key)

        return unmasked, missing

    def unmask(self, text: str, mapping: Dict[str, str]) -> str:
        """Restores original verbatim blocks from placeholders with spacing/digit tolerance."""
        unmasked, _ = self.unmask_with_report(text, mapping)
        return unmasked


# ==============================================================================
# 2. Configuration for Note Translator
# ==============================================================================

@dataclass
class NoteTranslatorConfig:
    """Configuration for Pedagogical Note Translation to Hindi."""

    backend: str = "ollama"  # "ollama" (merged Q4_K_M GGUF, ~10-20x faster) with automatic fallback to "pytorch"
    ollama_url: str = DEFAULT_TRANSLATION_OLLAMA_URL
    ollama_model: str = DEFAULT_TRANSLATION_OLLAMA_MODEL
    ollama_num_ctx: int = 4096
    ollama_num_gpu: int = 99
    ollama_keep_alive: str = "5m"
    ollama_timeout_sec: float = 60.0
    base_model_id: str = DEFAULT_BASE_MODEL
    adapter_path: Optional[str] = DEFAULT_ADAPTER_PATH
    device: str = "cuda"
    load_in_4bit: bool = True
    max_new_tokens: int = 768  # Generous token headroom to prevent mid-word truncation on dense Hindi prose
    max_words_per_chunk: int = 100  # Short paragraph/sentence chunks minimize Hinglish code-switching drift
    temperature: float = 0.0  # Factual greedy decoding
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 4
    length_penalty: float = 1.1
    force_fallback: bool = False
    offline: bool = False


# ==============================================================================
# 3. Core Pedagogical Note Translator Engine
# ==============================================================================

class NoteTranslator:
    """Translates Cornell study notes into publication-grade Hindi using fine-tuned Sarvam-1.

    Uses the FP16-merged Step 200 GGUF Q4_K_M model served via local Ollama
    (``sarvam1-hi-step200``) for native llama.cpp CUDA kernel speed, automatically
    falling back to 4-bit NF4 PyTorch + PEFT LoRA if Ollama is unavailable.
    """

    # Stop markers to prevent model from continuing beyond the translated subtopic
    STOP_MARKERS = [
        "\nअंग्रेजी:", "\n\nअंग्रेजी:", "\nEnglish:", "\n\nEnglish:",
        "\nहिन्दी:", "\n\nहिन्दी:", "\nTranslate", "\n\nTranslate",
        "\nTask:", "\nExplanation:", "\nNote:", "\n###", "</s>", "<eos>"
    ]

    def __init__(self, config: Optional[NoteTranslatorConfig] = None):
        self.config = config or NoteTranslatorConfig()
        self.masker = PlaceholderMasker()
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._active_backend: str = "fallback"
        self._translation_cache: Dict[str, str] = {}

    def _is_ollama_model_available(self, retries: int = 3, timeout_sec: float = 5.0) -> bool:
        """Check whether the merged Sarvam-1 GGUF model is registered in local Ollama with retry resilience."""
        import urllib.request

        target = self.config.ollama_model
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(f"{self.config.ollama_url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    if resp.status != 200:
                        continue
                    payload = json.loads(resp.read().decode("utf-8"))
                    for m in payload.get("models", []):
                        name = str(m.get("name", ""))
                        if name == target or name == f"{target}:latest" or name.split(":")[0] == target:
                            return True
            except Exception as e:
                logger.debug("Ollama translation model check attempt %d/%d failed: %s", attempt, retries, e)
                if attempt < retries:
                    time.sleep(1.0)
        return False

    def load_model(self) -> None:
        """Selects merged GGUF Ollama backend if available, else loads 4-bit PyTorch + LoRA."""
        if self._is_loaded:
            return

        if self.config.force_fallback or self.config.offline:
            logger.info("NoteTranslator: running in offline/force_fallback mode; skipping neural weight loading.")
            self._active_backend = "fallback"
            self._is_loaded = True
            return

        # Fast Path: Merged GGUF Q4_K_M served via native llama.cpp CUDA kernels in Ollama
        if self.config.backend == "ollama" and self._is_ollama_model_available():
            self._active_backend = "ollama"
            self._is_loaded = True
            logger.info(
                "NoteTranslator using merged GGUF Ollama backend: '%s' at %s.",
                self.config.ollama_model,
                self.config.ollama_url,
            )
            return

        if self.config.backend == "ollama":
            logger.warning(
                "Ollama model '%s' not available at %s; falling back to 4-bit PyTorch + LoRA.",
                self.config.ollama_model,
                self.config.ollama_url,
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        target_device = self.config.device
        if target_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for NoteTranslator but unavailable. Falling back to CPU.")
            target_device = "cpu"

        logger.info(f"Loading NoteTranslator base model '{self.config.base_model_id}' (device={target_device})...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if target_device == "cuda" and self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                torch_dtype=torch.float32 if target_device == "cpu" else torch.float16,
                device_map="auto" if target_device == "cuda" else None,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )
            if target_device != "cuda":
                base_model = base_model.to(target_device)

        # Attach fine-tuned LoRA adapter if path provided and exists
        adapter_path = self.config.adapter_path
        if adapter_path and os.path.isdir(adapter_path):
            logger.info(f"Attaching Step 200 champion LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            if adapter_path:
                logger.warning(f"LoRA adapter path '{adapter_path}' not found. Using base model.")
            model = base_model

        model.eval()
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None
        self.model = model
        self.tokenizer = tokenizer
        self._active_backend = "pytorch"
        self._is_loaded = True
        logger.info("NoteTranslator PyTorch model successfully loaded into memory.")

    def unload_model(self) -> None:
        """Releases model weights (PyTorch or Ollama GGUF) and frees GPU VRAM."""
        if not self._is_loaded:
            return

        logger.info("Unloading NoteTranslator from memory (backend=%s)...", self._active_backend)
        if self._active_backend == "ollama":
            import urllib.request

            try:
                req = urllib.request.Request(
                    f"{self.config.ollama_url}/api/generate",
                    data=json.dumps({"model": self.config.ollama_model, "keep_alive": 0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=3.0):
                    pass
            except Exception as e:
                logger.debug("Ollama NoteTranslator eviction skipped: %s", e)

        self.model = None
        self.tokenizer = None
        self._active_backend = "fallback"
        self._is_loaded = False
        self._translation_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("NoteTranslator memory cleared.")

    def _split_into_chunks(self, text: str, max_words: int) -> List[str]:
        """Splits text first by paragraph boundaries and then by sentences within max_words.

        Keeping chunks paragraph-aligned and compact (<=100 words) prevents Sarvam-1
        from drifting into Hinglish code-switching or hitting token-length cutoffs.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
        if not paragraphs:
            return [text]

        chunks: List[str] = []
        for para in paragraphs:
            if len(para.split()) <= max_words:
                chunks.append(para)
                continue

            raw_sentences = re.split(r"(?<=[.!?])\s+", para)
            sentences = [s.strip() for s in raw_sentences if s.strip()]

            current_chunk: List[str] = []
            current_word_count = 0
            for sent in sentences:
                sent_word_count = len(sent.split())
                if current_word_count + sent_word_count > max_words and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_word_count = sent_word_count
                else:
                    current_chunk.append(sent)
                    current_word_count += sent_word_count

            if current_chunk:
                chunks.append(" ".join(current_chunk))

        return chunks or [text]

    def _estimate_max_tokens(self, text: str) -> int:
        """Compute dynamic output token budget proportional to source length to prevent truncation."""
        word_count = max(1, len(text.split()))
        # Hindi Devanagari averages ~2.5-3.5 tokens per dense technical English word on Sarvam-1
        estimated = int(word_count * 3.8) + 40
        return max(48, min(self.config.max_new_tokens, estimated))

    def _translate_raw_chunk_ollama(self, masked_chunk: str) -> str:
        """Translate a single masked chunk via merged Q4_K_M GGUF served in local Ollama."""
        import urllib.request

        prompt = f"अंग्रेजी:\n{masked_chunk.strip()}\n\nहिन्दी (देवनागरी लिपि में अनुवाद):\n"
        max_tok = self._estimate_max_tokens(masked_chunk)

        request_body = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "repeat_penalty": self.config.repetition_penalty,
                "num_predict": max_tok,
                "num_ctx": self.config.ollama_num_ctx,
                "num_gpu": self.config.ollama_num_gpu,
                "stop": [
                    "</s>",
                    "\nअंग्रेजी:",
                    "\n\nअंग्रेजी:",
                    "\nEnglish:",
                    "\n\nEnglish:",
                    "\nहिन्दी:",
                    "\n\nहिन्दी:",
                    "\n###",
                ],
            },
            "keep_alive": self.config.ollama_keep_alive,
        }

        req = urllib.request.Request(
            f"{self.config.ollama_url}/api/generate",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.ollama_timeout_sec) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                decoded = str(payload.get("response", "")).strip()
                for marker in self.STOP_MARKERS:
                    if marker in decoded:
                        decoded = decoded.split(marker)[0].strip()
                return decoded

        raise RuntimeError(f"Ollama translation request failed for model '{self.config.ollama_model}'.")

    def _translate_raw_chunk(self, masked_chunk: str) -> str:
        """Translates a single masked chunk using the Devanagari prompt anchor."""
        if not masked_chunk.strip():
            return ""

        if self.config.force_fallback or self.config.offline:
            return f"[अनुवादित: {masked_chunk.strip()}]"

        if self._active_backend == "ollama":
            return self._translate_raw_chunk_ollama(masked_chunk)

        if self.model is None:
            return f"[अनुवादित: {masked_chunk.strip()}]"

        prompt = f"अंग्रेजी:\n{masked_chunk.strip()}\n\nहिन्दी (देवनागरी लिपि में अनुवाद):\n"
        device = self.model.device
        max_tok = self._estimate_max_tokens(masked_chunk)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        word_count = len(masked_chunk.split())
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tok,
            "do_sample": (self.config.temperature > 0.0),
            "temperature": self.config.temperature if self.config.temperature > 0.0 else None,
            "top_p": 0.9 if self.config.temperature > 0.0 else None,
            "repetition_penalty": self.config.repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # Only apply CPU-synced NoRepeatNGramLogitsProcessor on longer passages
        if word_count >= 40 and self.config.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        gen_tokens = outputs[0][prompt_len:]
        decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean trailing stop markers
        for marker in self.STOP_MARKERS:
            if marker in decoded:
                decoded = decoded.split(marker)[0].strip()

        return decoded

    def _translate_piecewise_preserving_spans(
        self, masked_text: str, mapping: Dict[str, str]
    ) -> str:
        """Fallback translator that translates only non-math prose clauses and stitches
        verbatim math/code spans back into their exact positions.

        Guarantees 100% mathematical formula and code preservation whenever the LLM
        drops or mutates an inline placeholder token.
        """
        parts = self.masker.PLACEHOLDER_TOKEN_RE.split(masked_text)
        rebuilt: List[str] = []

        for part in parts:
            if not part:
                continue
            if part in mapping:
                rebuilt.append(mapping[part])
                continue

            # Preserve leading/trailing whitespace around inline spans
            lead_ws = part[: len(part) - len(part.lstrip())]
            trail_ws = part[len(part.rstrip()) :]
            core = part.strip()

            # Do not send pure punctuation (e.g., ',', '.', ')') to the LLM
            if not core or not re.search(r"[A-Za-z0-9\u0900-\u097F]", core):
                rebuilt.append(part)
                continue

            chunks = self._split_into_chunks(core, max_words=self.config.max_words_per_chunk)
            translated_core = " ".join(self._translate_raw_chunk(c) for c in chunks).strip()
            # Clean trailing sentence period if original fragment didn't end a sentence
            if not core.endswith((".", "!", "?", "।")) and translated_core.endswith(("।", ".")):
                translated_core = translated_core.rstrip("।.").strip()

            if lead_ws == "" and rebuilt and rebuilt[-1] and not rebuilt[-1][-1].isspace():
                if not core[0] in ",.;:!?)]}":
                    lead_ws = " "

            rebuilt.append(f"{lead_ws}{translated_core}{trail_ws}")

        result = "".join(rebuilt).strip()
        # Normalize accidental multiple spaces while preserving newlines
        result = re.sub(r"[ \t]{2,}", " ", result)
        return result

    def translate_text(self, text: str) -> str:
        """Translates an arbitrary English text block to Hindi with memoization & guaranteed math/code preservation."""
        if not text or not text.strip():
            return text or ""

        cache_key = text.strip()
        cached = self._translation_cache.get(cache_key)
        if cached is not None:
            return cached

        # Step 1: Mask math and code blocks with script-neutral numeric sentinels
        masked_text, mapping = self.masker.mask(text)

        # Short-circuit if text consists solely of masked math/code blocks & punctuation
        if mapping:
            residual_prose = self.masker.PLACEHOLDER_TOKEN_RE.sub("", masked_text)
            if not re.search(r"[A-Za-z]", residual_prose):
                self._translation_cache[cache_key] = text
                return text

        # Step 2: Chunk if exceeding optimal subtopic length
        chunks = self._split_into_chunks(masked_text, max_words=self.config.max_words_per_chunk)
        translated_chunks = [self._translate_raw_chunk(c) for c in chunks]
        combined_translation = "\n\n".join(translated_chunks).strip()

        # Step 3: Unmask math and code blocks verbatim & verify 100% span integrity
        unmasked, missing_keys = self.masker.unmask_with_report(combined_translation, mapping)
        if missing_keys:
            logger.debug(
                "Detected %d mutated/dropped placeholder(s) %s; invoking piecewise span-preserving translation.",
                len(missing_keys),
                missing_keys,
            )
            unmasked = self._translate_piecewise_preserving_spans(masked_text, mapping)

        self._translation_cache[cache_key] = unmasked
        return unmasked

    # ==========================================================================
    # 4. Strongly Typed Note Translation & Rendering
    # ==========================================================================

    def translate_study_notes(
        self,
        notes: Union[LectureStudyNotes, Dict[str, Any]],
        auto_unload: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Translates a full LectureStudyNotes instance or dictionary into Hindi Markdown and JSON.

        Args:
            notes: English LectureStudyNotes instance or dictionary.
            auto_unload: Whether to immediately unload weights to free VRAM when done.

        Returns:
            Tuple of (hindi_markdown_str, hindi_notes_json_dict).
        """
        if isinstance(notes, dict):
            notes = LectureStudyNotes.from_dict(notes)

        self.load_model()
        logger.info(f"Beginning Hindi translation of study notes: '{notes.metadata.get('lecture_title', '')}'...")

        # Deep-copy to avoid mutating the original English notes instance
        hi_notes = copy.deepcopy(notes)

        # 1. Translate Metadata
        orig_title = hi_notes.metadata.get("lecture_title", "")
        if orig_title:
            hi_notes.metadata["lecture_title"] = self.translate_text(orig_title)
        orig_course = hi_notes.metadata.get("course", "")
        if orig_course:
            hi_notes.metadata["course"] = self.translate_text(orig_course)

        # 2. Translate Executive Overview
        if hi_notes.global_review.executive_synthesis:
            hi_notes.global_review.executive_synthesis = self.translate_text(
                hi_notes.global_review.executive_synthesis
            )

        # 3. Translate Table of Contents
        for item in hi_notes.table_of_contents:
            if "title" in item and item["title"]:
                item["title"] = self.translate_text(item["title"])

        seen_sections: Set[int] = set()

        # 4. Translate Modules (if 2-tier hierarchical architecture enabled)
        if hi_notes.modules:
            for mod in hi_notes.modules:
                mod.module_title = self.translate_text(mod.module_title)
                if mod.pedagogical_focus:
                    mod.pedagogical_focus = self.translate_text(mod.pedagogical_focus)
                if mod.module_synthesis:
                    mod.module_synthesis = self.translate_text(mod.module_synthesis)
                for sec in mod.subsections:
                    if id(sec) not in seen_sections:
                        self._translate_study_section_inplace(sec)
                        seen_sections.add(id(sec))

        # 5. Translate Study Sections
        for sec in hi_notes.study_sections:
            if id(sec) not in seen_sections:
                self._translate_study_section_inplace(sec)
                seen_sections.add(id(sec))

        # 6. Translate Administrative Digressions
        for ad in hi_notes.administrative_digressions:
            ad.title = self.translate_text(ad.title)
            ad.announcement_notes = self.translate_text(ad.announcement_notes)

        # 7. Translate Global Review (Master Glossary & Active Recall)
        for g in hi_notes.global_review.master_glossary:
            # Preserve original term in parentheses: e.g. "सममित मैट्रिक्स (Symmetric Matrix)"
            trans_term = self.translate_text(g.term)
            g.term = f"{trans_term} ({g.term})" if trans_term.strip() != g.term.strip() else g.term
            g.definition = self.translate_text(g.definition)

        for q in hi_notes.global_review.active_recall_review_sheet:
            q.question = self.translate_text(q.question)
            q.core_concept = self.translate_text(q.core_concept)

        if hi_notes.global_review.comparative_synthesis_questions:
            for item in hi_notes.global_review.comparative_synthesis_questions:
                if "question" in item:
                    item["question"] = self.translate_text(item["question"])
                if "pedagogical_focus" in item:
                    item["pedagogical_focus"] = self.translate_text(item["pedagogical_focus"])

        # Render publication-grade Hindi Markdown
        hindi_md = self._render_hindi_markdown(hi_notes)
        hindi_json = hi_notes.to_dict()

        if auto_unload:
            self.unload_model()

        logger.info("Hindi study notes translation successfully completed.")
        return hindi_md, hindi_json

    def _translate_study_section_inplace(self, sec: StudySection) -> None:
        """Translates all text components of a single StudySection."""
        sec.topic_title = self.translate_text(sec.topic_title)

        # Cornell Cue Column
        if sec.cornell_cue_column.cue_questions:
            sec.cornell_cue_column.cue_questions = [
                self.translate_text(q) for q in sec.cornell_cue_column.cue_questions
            ]
        if sec.cornell_cue_column.core_keywords:
            sec.cornell_cue_column.core_keywords = [
                self.translate_text(k) for k in sec.cornell_cue_column.core_keywords
            ]

        # Conceptual Exposition
        sec.detailed_notes.conceptual_exposition = self.translate_text(
            sec.detailed_notes.conceptual_exposition
        )

        # Formal Definitions
        for d in sec.detailed_notes.formal_definitions:
            trans_term = self.translate_text(d.term)
            d.term = f"{trans_term} ({d.term})" if trans_term.strip() != d.term.strip() else d.term
            d.definition = self.translate_text(d.definition)

        # Mathematical Formulations: keep latex_expression verbatim, translate explanation
        for m in sec.detailed_notes.mathematical_formulations:
            m.explanation = self.translate_text(m.explanation)

        # Algorithmic Code Blocks: keep language & code verbatim, translate caption
        for c in sec.detailed_notes.algorithmic_code_blocks:
            c.caption = self.translate_text(c.caption)

        # Concrete Examples
        if sec.detailed_notes.concrete_examples:
            sec.detailed_notes.concrete_examples = [
                self.translate_text(ex) for ex in sec.detailed_notes.concrete_examples
            ]

        # Key Takeaways
        if sec.takeaways:
            sec.takeaways = [self.translate_text(t) for t in sec.takeaways]

    def _render_hindi_markdown(self, notes: LectureStudyNotes) -> str:
        """Renders publication-grade Hindi Markdown with localized headers and KaTeX formulas."""
        lines: List[str] = []

        title = notes.metadata.get("lecture_title", "शैक्षणिक व्याख्यान अध्ययन नोट्स")
        course = notes.metadata.get("course", "कंप्यूटर साइंस")
        duration = notes.metadata.get("total_duration_str", "00:00:00")
        total_modules = len(notes.modules) if notes.modules else 1
        total_segs = len(notes.study_sections)
        total_sents = notes.metadata.get("total_sentences", 0)

        lines.append(f"# {title} (अध्ययन नोट्स)\n")
        lines.append(f"> **पाठ्यक्रम (Course)**: {course}  ")
        if notes.modules:
            lines.append(f"> **कुल अवधि**: `[{duration}]` | **मॉड्यूल**: {total_modules} | **उप-अनुभाग**: {total_segs} | **वाक्य**: {total_sents}  ")
        else:
            lines.append(f"> **कुल अवधि**: `[{duration}]` | **अनुभाग**: {total_segs} | **वाक्य**: {total_sents}  ")
        lines.append("> **शैक्षणिक अखंडता**: व्याख्यान देखने का 100% व्यापक विकल्प (शून्य विनाशकारी संपीड़न)\n")
        lines.append("---\n")

        # Executive Overview
        lines.append("## कार्यकारी अवलोकन (Executive Overview)\n")
        if notes.global_review.executive_synthesis:
            lines.append(f"{notes.global_review.executive_synthesis}\n")
        lines.append("---\n")

        # Table of Contents
        lines.append("## विषय - सूची (Table of Contents)\n")
        if notes.modules:
            for mod in notes.modules:
                lines.append(f"- `[{mod.start_time_str}]` **मॉड्यूल {mod.module_id}: {mod.module_title}**")
                for sub_idx, sec in enumerate(mod.subsections, 1):
                    digress_tag = " *(प्रशासनिक उद्घोषणा)*" if sec.is_digression else ""
                    lines.append(
                        f"  - `[{sec.start_time_str}]` उप-अनुभाग {mod.module_id}.{sub_idx}: {sec.topic_title}{digress_tag}"
                    )
            if notes.administrative_digressions:
                for ad in notes.administrative_digressions:
                    lines.append(f"- `[{ad.start_time_str}]` सूचना: {ad.title} *(प्रशासनिक उद्घोषणा)*")
        else:
            for item in notes.table_of_contents:
                digress_tag = " *(प्रशासनिक उद्घोषणा)*" if item.get("is_digression") else ""
                lines.append(
                    f"- `[{item['start_time_str']}]` अनुभाग {item['section_id'] + 1}: {item['title']}{digress_tag}"
                )
        lines.append("- वैश्विक पुनरावलोकन: मास्टर अवधारणा शब्दावली एवं सक्रिय स्मरण समीक्षा पत्र\n")
        lines.append("---\n")

        # Study Sections Rendering
        if notes.modules:
            for mod in notes.modules:
                lines.append(f"# मॉड्यूल {mod.module_id}: {mod.module_title}\n")
                lines.append(f"**समय सीमा**: `[{mod.start_time_str}] - [{mod.end_time_str}]` | **उप-अनुभाग**: {len(mod.subsections)}\n")
                if mod.pedagogical_focus:
                    lines.append(f"> *मॉड्यूल का मुख्य केंद्र*: {mod.pedagogical_focus}\n")
                if mod.module_synthesis:
                    lines.append(f"{mod.module_synthesis}\n")

                for sub_idx, sec in enumerate(mod.subsections, 1):
                    lines.append(f"## उप-अनुभाग {mod.module_id}.{sub_idx}: {sec.topic_title}\n")
                    lines.append(f"**समय सीमा**: `[{sec.start_time_str}] - [{sec.end_time_str}]`  ")
                    prereq_str = ", ".join(f"`{p}`" for p in sec.prerequisite_tags) if sec.prerequisite_tags else "कोई नहीं (बुनियादी)"
                    lines.append(f"**पूर्वापेक्षाएँ (Prerequisites)**: {prereq_str}  ")
                    if sec.cornell_cue_column.core_keywords:
                        kw_str = ", ".join(sec.cornell_cue_column.core_keywords)
                        lines.append(f"**प्रमुख अवधारणाएं**: {kw_str}  ")
                    lines.append("**वर्गीकरण**: मुख्य शैक्षणिक सामग्री\n")

                    lines.append(f"{sec.detailed_notes.conceptual_exposition}\n")

                    if sec.detailed_notes.formal_definitions:
                        lines.append("### औपचारिक परिभाषाएँ (Formal Definitions)")
                        for d in sec.detailed_notes.formal_definitions:
                            lines.append(f"- **{d.term}**: {d.definition}")
                        lines.append("")

                    if sec.detailed_notes.mathematical_formulations:
                        lines.append("### गणितीय सूत्र (Mathematical Formulations)")
                        for m in sec.detailed_notes.mathematical_formulations:
                            lines.append(f"{m.latex_expression}\n")
                            lines.append(f"*{m.explanation}*\n")

                    if sec.detailed_notes.algorithmic_code_blocks:
                        lines.append("### एल्गोरिथम संरचना (Algorithmic Formulation)")
                        for c in sec.detailed_notes.algorithmic_code_blocks:
                            lines.append(f"```{c.language}")
                            lines.append(c.code)
                            lines.append("```")
                            lines.append(f"*{c.caption}*\n")

                    if sec.detailed_notes.concrete_examples:
                        lines.append("### ठोस उदाहरण (Concrete Examples)")
                        for ex in sec.detailed_notes.concrete_examples:
                            lines.append(f"- {ex}")
                        lines.append("")

                    if sec.takeaways:
                        lines.append("### मुख्य निष्कर्ष (Key Takeaways)")
                        for t in sec.takeaways:
                            lines.append(f"- {t}")
                        lines.append("")

                    lines.append("---\n")
        else:
            for sec in notes.study_sections:
                lines.append(f"## अनुभाग {sec.section_id + 1}: {sec.topic_title}\n")
                lines.append(f"**समय सीमा**: `[{sec.start_time_str}] - [{sec.end_time_str}]`  ")
                prereq_str = ", ".join(f"`{p}`" for p in sec.prerequisite_tags) if sec.prerequisite_tags else "कोई नहीं (बुनियादी)"
                lines.append(f"**पूर्वापेक्षाएँ (Prerequisites)**: {prereq_str}  ")
                if sec.cornell_cue_column.core_keywords:
                    kw_str = ", ".join(sec.cornell_cue_column.core_keywords)
                    lines.append(f"**प्रमुख अवधारणाएं**: {kw_str}  ")
                lines.append("**वर्गीकरण**: मुख्य शैक्षणिक सामग्री\n")

                lines.append(f"{sec.detailed_notes.conceptual_exposition}\n")

                if sec.detailed_notes.formal_definitions:
                    lines.append("### औपचारिक परिभाषाएँ (Formal Definitions)")
                    for d in sec.detailed_notes.formal_definitions:
                        lines.append(f"- **{d.term}**: {d.definition}")
                    lines.append("")

                if sec.detailed_notes.mathematical_formulations:
                    lines.append("### गणितीय सूत्र (Mathematical Formulations)")
                    for m in sec.detailed_notes.mathematical_formulations:
                        lines.append(f"{m.latex_expression}\n")
                        lines.append(f"*{m.explanation}*\n")

                if sec.detailed_notes.algorithmic_code_blocks:
                    lines.append("### एल्गोरिथम संरचना (Algorithmic Formulation)")
                    for c in sec.detailed_notes.algorithmic_code_blocks:
                        lines.append(f"```{c.language}")
                        lines.append(c.code)
                        lines.append("```")
                        lines.append(f"*{c.caption}*\n")

                if sec.detailed_notes.concrete_examples:
                    lines.append("### ठोस उदाहरण (Concrete Examples)")
                    for ex in sec.detailed_notes.concrete_examples:
                        lines.append(f"- {ex}")
                    lines.append("")

                if sec.takeaways:
                    lines.append("### मुख्य निष्कर्ष (Key Takeaways)")
                    for t in sec.takeaways:
                        lines.append(f"- {t}")
                    lines.append("")

                lines.append("---\n")

        # Administrative Digressions
        if notes.administrative_digressions:
            lines.append("## पाठ्यक्रम प्रबंधन एवं प्रशासनिक घोषणाएं\n")
            lines.append(
                "> *नोट: निम्नलिखित अनुभाग गैर-शैक्षणिक पाठ्यक्रम प्रबंधन के रूप में पहचाने गए थे "
                "और छात्रों की प्रशासनिक जागरूकता के लिए मुख्य सैद्धांतिक प्रवाह को प्रभावित किए बिना यहाँ सहेजे गए हैं।*\n"
            )
            for d in notes.administrative_digressions:
                lines.append(f"### ⚠️ सूचना: {d.title}")
                lines.append(f"**समय सीमा**: `[{d.start_time_str}] - [{d.end_time_str}]` | **श्रेणी**: `{d.digression_type}`\n")
                lines.append(f"> {d.announcement_notes}\n")
            lines.append("---\n")

        # Global Review & Master Study Aids
        lines.append("## वैश्विक पुनरावलोकन एवं मास्टर अध्ययन सहायता\n")

        if notes.global_review.master_glossary:
            lines.append("### मास्टर अवधारणा शब्दावली (Master Concept Glossary)\n")
            lines.append("| अवधारणा | औपचारिक परिभाषा | संदर्भ |")
            lines.append("|:--------|:----------------|:-------|")
            for g in notes.global_review.master_glossary:
                refs = ", ".join(f"अनुभाग {s + 1}" for s in g.referenced_in_sections)
                lines.append(f"| **{g.term}** | {g.definition} | {refs} |")
            lines.append("\n")

        if notes.global_review.active_recall_review_sheet:
            lines.append("### सक्रिय स्मरण समीक्षा पत्र (Active Recall Review Sheet)\n")
            for idx, q in enumerate(notes.global_review.active_recall_review_sheet, 1):
                lines.append(f"{idx}. **प्रश्न**: {q.question}")
                lines.append(f"   *मुख्य अवधारणा*: {q.core_concept} (संदर्भ: अनुभाग {q.section_id + 1})\n")

        return "\n".join(lines).strip() + "\n"
