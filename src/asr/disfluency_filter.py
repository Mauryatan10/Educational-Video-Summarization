"""Disfluency scrubbing, regex filler removal, and neural/heuristic punctuation restoration.

Conforms to Milestone 1 Feature 4 and Interface Contract 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Robust regex: handles trailing non-word '?' in 'right?', compound fillers, and terminal 'like'
DEFAULT_FILLER_REGEX = (
    r"\b(um|uh|er|ah|you know|I mean|sort of(?:\s+like)?|kind of(?:\s+like)?)\b"
    r"|\bright\?(?!\w)"
    r"|\b(like)(?=\s*,\s*|\s+(?:um|uh|you know)|\s*[\.\?!]?$)"
)
DEFAULT_STUTTER_REGEX = r"\b([A-Za-z]+)(?:\s*,\s*|\s+)\1\b"

COMMON_STEM_ACRONYMS = {
    "ai": "AI",
    "ml": "ML",
    "nlp": "NLP",
    "gpu": "GPU",
    "cpu": "CPU",
    "ram": "RAM",
    "vram": "VRAM",
    "mdp": "MDP",
    "pomdp": "POMDP",
    "rl": "RL",
    "llm": "LLM",
    "llms": "LLMs",
    "ctc": "CTC",
    "vad": "VAD",
    "asr": "ASR",
    "dbca": "DBCA",
    "qwen": "Qwen",
    "katex": "KaTeX",
    "sgd": "SGD",
    "svm": "SVM",
    "tlb": "TLB",
    "os": "OS",
    "mmu": "MMU",
    "pcb": "PCB",
    "pagedadamw8bit": "PagedAdamW8bit",
    "infonce": "InfoNCE",
}


@dataclass
class DisfluencyFilterConfig:
    """Configuration parameters for disfluency scrubbing and punctuation recovery."""

    filler_regex: str = DEFAULT_FILLER_REGEX
    stutter_regex: str = DEFAULT_STUTTER_REGEX
    max_stutter_passes: int = 5
    enable_truecasing: bool = True
    use_neural_punctuation: bool = True
    neural_model_name: str = "oliverguhr/fullstop-punctuation-multilingual-large"
    device: str = "cpu"
    acronym_map: Dict[str, str] = field(default_factory=lambda: COMMON_STEM_ACRONYMS.copy())
    preserve_safe_like: bool = True


class DisfluencyFilter:
    """Removes verbal hesitation fillers, resolves stutters, cleans punctuation artifacts,

    and restores grammatical casing and punctuation with neural/heuristic fallback.
    """

    def __init__(self, config: Optional[DisfluencyFilterConfig] = None):
        self.config = config or DisfluencyFilterConfig()
        self.filler_pattern = re.compile(self.config.filler_regex, re.IGNORECASE)
        self.stutter_pattern = re.compile(self.config.stutter_regex, re.IGNORECASE)
        self._neural_model: Any = None
        self._neural_available: bool = False

        if self.config.use_neural_punctuation:
            self._try_init_neural_model()

    def _try_init_neural_model(self) -> None:
        """Safely attempt to initialize DeepMultilingualPunctuation without crashing."""
        try:
            from deepmultilingualpunctuation import PunctuationModel

            self._neural_model = PunctuationModel(model=self.config.neural_model_name)
            self._neural_available = True
        except Exception:
            self._neural_model = None
            self._neural_available = False

    def resolve_stutters(self, text: str) -> str:
        """Iteratively collapse immediate lexical repetitions (e.g. 'the the the' -> 'the')."""
        if not text:
            return ""
        current = text
        for _ in range(self.config.max_stutter_passes):
            new_text = self.stutter_pattern.sub(r"\1", current)
            if new_text == current:
                break
            current = new_text
        return current

    def remove_fillers(self, text: str) -> str:
        """Strip conversational filler words defined in regex."""
        if not text:
            return ""
        cleaned = self.filler_pattern.sub("", text)
        if not self.config.preserve_safe_like:
            cleaned = re.sub(r"\blike\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def clean_punctuation_artifacts(self, text: str) -> str:
        """Clean leftover punctuation artifacts, double commas, orphan question marks, and spacing."""
        if not text:
            return ""
        # Remove orphan commas/colons before terminal punctuation
        text = re.sub(r"[,;:\s]+([\.\?!])", r"\1", text)
        # Collapse multiple commas/colons/dashes (e.g. ",,", ", ,", ";,")
        text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
        # Remove leading punctuation
        text = re.sub(r"^[\s,;:–\-\?]+", "", text)
        # Remove trailing non-terminal punctuation (e.g. dangling comma after filler removal)
        text = re.sub(r"[\s,;:–\-]+$", "", text)
        # Clean whitespace before punctuation
        text = re.sub(r"\s+([,\.\?!;:])", r"\1", text)
        # Ensure single space after punctuation:
        # 1. Commas, colons, semicolons followed by letters
        text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)
        # 2. Question mark, exclamation mark followed by alphanumeric
        text = re.sub(r"([\?!])([A-Za-z0-9])", r"\1 \2", text)
        # 3. Period followed by letter, protecting decimals and dotted acronyms (e.g. A.I., U.S.A., e.g., i.e.)
        text = re.sub(r"(?<!\b[A-Za-z])\.(?![0-9])(?![A-Za-z]\.)([A-Za-z])", r". \1", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def heuristic_truecase_and_punctuate(self, text: str) -> str:
        """Offline fallback: capitalizes sentence starts, truecases acronyms and 'I',

        and ensures valid terminal punctuation.
        """
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""

        # Truecase tokens
        processed_tokens = []
        for tok in tokens:
            clean_tok = re.sub(r"^[^\w]+|[^\w]+$", "", tok).lower()
            if clean_tok in self.config.acronym_map:
                replacement = self.config.acronym_map[clean_tok]
                prefix = tok[: len(tok) - len(tok.lstrip("^([{\"'"))]
                suffix = tok[len(tok.rstrip(".,!?;:\"')}")) :]
                processed_tokens.append(f"{prefix}{replacement}{suffix}")
            elif clean_tok == "i":
                suffix = tok[len(tok.rstrip(".,!?;:\"')}")) :]
                processed_tokens.append(f"I{suffix}")
            elif clean_tok in ("i'm", "i've", "i'll", "i'd"):
                suffix = tok[len(tok.rstrip(".,!?;:\"')}")) :]
                cont = clean_tok[0].upper() + clean_tok[1:]
                processed_tokens.append(f"{cont}{suffix}")
            else:
                processed_tokens.append(tok)

        text = " ".join(processed_tokens)

        # Capitalize sentence beginnings, but not after dotted acronyms (A.I., e.g., i.e.)
        def cap_sentence(m: re.Match) -> str:
            return m.group(1) + m.group(2).upper()

        text = re.sub(
            r"(^|(?<!\b[A-Za-z]\.[A-Za-z])(?<!\b[A-Za-z]\.[A-Za-z]\.[A-Za-z])[\.\?!]\s+)([a-z])",
            cap_sentence,
            text,
        )

        # Ensure terminal punctuation if none exists
        if text and text[-1] not in ".?!":
            first_word = text.split()[0].lower()
            if first_word in {
                "what",
                "why",
                "how",
                "where",
                "when",
                "who",
                "is",
                "are",
                "can",
                "does",
                "do",
                "should",
                "would",
            }:
                text += "?"
            else:
                text += "."

        return text

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline on a raw text string:

        1. Stutter resolution
        2. Filler removal
        3. Punctuation artifact cleanup
        4. Neural or heuristic punctuation & truecasing
        """
        if not text or not text.strip():
            return ""

        # 1. Stutter resolution
        t = self.resolve_stutters(text)

        # 2. Filler removal
        t = self.remove_fillers(t)

        # 3. Clean punctuation artifacts
        t = self.clean_punctuation_artifacts(t)

        if not t.strip() or not re.search(r"[A-Za-z0-9]", t):
            return ""

        # 4. Neural or heuristic punctuation & truecasing
        if self._neural_available and self._neural_model is not None:
            try:
                t = self._neural_model.restore_punctuation(t)
            except Exception:
                t = self.heuristic_truecase_and_punctuate(t)
        else:
            t = self.heuristic_truecase_and_punctuate(t)

        return t

    def clean_sentence_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean a list of sentence dicts conforming to Interface Contract 1.

        Preserves timestamps and acoustic features while updating text and sentence_id.
        Excludes sentences that become completely empty after filler removal.
        """
        cleaned_records = []
        out_id = 0
        for rec in records:
            raw_text = rec.get("text", "")
            cleaned = self.clean_text(raw_text)
            if not cleaned:
                continue  # Drop sentences that consisted solely of fillers
            new_rec = dict(rec)
            new_rec["sentence_id"] = out_id
            new_rec["text"] = cleaned
            cleaned_records.append(new_rec)
            out_id += 1
        return cleaned_records
