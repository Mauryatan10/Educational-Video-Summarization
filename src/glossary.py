"""
glossary.py — build a source->target glossary for IndicTrans2.

IndicTransToolkit has NO native terminology/constraint API, so we:
  (1) extract English terms/entities from the transcript+summary (NER or KeyBERT),
  (2) self-translate each EN term into the target language with IndicTrans2
      (translate_fn), giving (en_term, tgt_term) pairs,
  (3) inject those pairs as a `Glossary:` source prefix (see translate_stage
      build_source_with_prefix) and use them in the QE repair step.

Default backend is a multilingual-friendly English NER pipeline so the module
works without KeyBERT installed (KeyBERT is imported lazily only if selected).
"""

from __future__ import annotations

import re
from typing import Callable, List, Sequence, Tuple

from translate_stage import format_glossary_block, translate_one

GlossaryPair = Tuple[str, str]


def extract_terms_ner(
    text: str,
    model: str = "dslim/bert-base-NER",
    aggregation_strategy: str = "first",
    max_terms: int = 25,
) -> List[str]:
    """Extract entity phrases from English source text via a transformers NER pipeline."""
    from transformers import pipeline

    if not text or not text.strip():
        return []
    nlp = pipeline(
        "ner",
        model=model,
        aggregation_strategy=aggregation_strategy,
        device=-1,
    )
    ents = nlp(text)
    seen = set()
    terms: List[str] = []
    for e in ents:
        # Drop any residual WordPiece fragment and collapse internal spaces.
        word = (e.get("word") or "").replace("##", "")
        word = re.sub(r"\s+", "", word).strip()
        alpha = re.sub(r"[^A-Za-z]", "", word)
        if not word or len(alpha) < 3:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(word)
        if len(terms) >= max_terms:
            break
    return terms


def extract_terms_keybert(
    text: str,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_n: int = 15,
    ngram_range=(1, 3),
) -> List[str]:
    """Extract key phrases via KeyBERT (lazy import; not installed by default)."""
    try:
        from keybert import KeyBERT
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "KeyBERT is not installed. Run `pip install keybert` or use backend='ner'."
        ) from exc
    kw = KeyBERT(model=model)
    kws = kw.extract_keywords(
        text, keyphrase_ngram_range=ngram_range, stop_words="english", top_n=top_n
    )
    return [phrase for phrase, _ in kws if phrase]


def extract_terms(
    text: str,
    backend: str = "ner",
    **kwargs,
) -> List[str]:
    if backend == "keybert":
        return extract_terms_keybert(text, **kwargs)
    if backend == "ner":
        return extract_terms_ner(text, **kwargs)
    raise ValueError(f"Unknown glossary backend: {backend!r}. Use 'ner' or 'keybert'.")


def build_glossary(
    en_terms: Sequence[str],
    translate_fn: Callable[[str], str],
) -> List[GlossaryPair]:
    """Translate each English term into the target language via `translate_fn`.

    `translate_fn(en_term) -> tgt_term` (single string). Returns (en, tgt) pairs
    that produced a non-empty target.
    """
    pairs: List[GlossaryPair] = []
    for term in en_terms:
        term = (term or "").strip()
        if not term:
            continue
        try:
            tgt = translate_fn(term).strip()
        except Exception:
            tgt = ""
        if tgt:
            pairs.append((term, tgt))
    return pairs


def glossary_from_text(
    text: str,
    translate_fn: Callable[[str], str],
    backend: str = "ner",
    max_pairs: int = 20,
    **kwargs,
) -> List[GlossaryPair]:
    """One-shot: extract terms from `text` and self-translate them into the glossary."""
    terms = extract_terms(text, backend=backend, **kwargs)
    pairs = build_glossary(terms, translate_fn)
    return pairs[:max_pairs]


def check_glossary_coverage(
    pairs: Sequence[GlossaryPair],
    translation: str,
) -> List[GlossaryPair]:
    """Return the subset of `pairs` whose target term is NOT present in `translation`."""
    translation = (translation or "").lower()
    violated: List[GlossaryPair] = []
    for en, tgt in pairs:
        if tgt and tgt.lower() not in translation:
            violated.append((en, tgt))
    return violated


def format_glossary(pairs: Sequence[GlossaryPair]) -> str:
    return format_glossary_block(pairs)
