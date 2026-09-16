"""
translation_qe.py — confidence-based QE router for IndicTrans2.

IndicTransToolkit has no native constraint/glossary API, so we route at the segment
level instead of running a full second pass:
  * translate a batch (optionally with the context backbone),
  * get per-segment confidence (mean token log-prob) from translate_stage,
  * for any segment whose confidence < tau OR whose glossary terms are violated,
    RE-DECODE that segment with the full context+glossary prefix.

This is a router, not a second full pass — see plan §4.3. Efficiency metrics
(% re-translated, latency) come from the returned report.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from translate_stage import translate_texts, translate_texts_with_scores

from glossary import check_glossary_coverage, GlossaryPair


def translate_with_qe(
    model,
    tokenizer,
    ip,
    sentences: List[str],
    tgt_lang: str,
    context: Optional[str] = None,
    glossary_pairs: Optional[Sequence[GlossaryPair]] = None,
    tau: Optional[float] = None,
    use_context_in_base: bool = True,
    max_length: int = 256,
    num_beams: int = 5,
    device=None,
) -> Tuple[List[str], Dict]:
    """Translate `sentences` to `tgt_lang` with a QE router.

    Returns (translations, report) where report includes re-decode counts for
    the efficiency/ablation metrics in plan §6.
    """
    base_ctx = context if (use_context_in_base and context) else None
    base_gloss = glossary_pairs if use_context_in_base else None

    translations, confidences = translate_texts_with_scores(
        model, tokenizer, ip, sentences, tgt_lang,
        context=base_ctx, glossary_pairs=base_gloss,
        max_length=max_length, num_beams=num_beams, device=device,
    )

    redecode_idx: List[int] = []
    low_conf: List[int] = []
    violated: List[int] = []
    for i, (t, c) in enumerate(zip(translations, confidences)):
        is_low = (
            tau is not None
            and c is not None
            and not (isinstance(c, float) and math.isnan(c))
            and c < tau
        )
        if is_low:
            low_conf.append(i)
        has_violation = bool(glossary_pairs) and bool(check_glossary_coverage(glossary_pairs, t))
        if has_violation:
            violated.append(i)
        if is_low or has_violation:
            redecode_idx.append(i)

    if redecode_idx:
        re_srcs = [sentences[i] for i in redecode_idx]
        re_trans = translate_texts(
            model, tokenizer, ip, re_srcs, tgt_lang,
            context=context, glossary_pairs=glossary_pairs,
            max_length=max_length, num_beams=num_beams, device=device,
        )
        for j, i in enumerate(redecode_idx):
            translations[i] = re_trans[j]

    report = {
        "total": len(sentences),
        "redecoded": len(redecode_idx),
        "low_confidence": len(low_conf),
        "glossary_violated": len(violated),
        "mean_confidence": (
            float(sum(c for c in confidences if not math.isnan(c)) / max(1, len([c for c in confidences if not math.isnan(c)])))
            if confidences else float("nan")
        ),
    }
    return translations, report
