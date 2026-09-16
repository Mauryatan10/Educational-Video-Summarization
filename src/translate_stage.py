"""
translate_stage.py — IndicTrans2 En->Indic translation stage.

Extracted and extended from `eval_indictrans2_finetranslations.py` (the canonical
reference for IndicProcessor usage + the base model). Adds:
  * a shared `build_source_with_prefix` used by BOTH training-data construction
    (prepare_finetranslations.py) and inference (app.py / translate_stage.py) so
    the context/glossary effect transfers across train/inference.
  * optional `context` (B2+ backbone) and `glossary_pairs` (glossary module) prefixing.
  * a score-returning variant for the QE router (translation_qe.py).

IndicTrans2 is seq2seq, so context/glossary are prepended to the *source* string
before `ip.preprocess_batch`. There is NO native constraint API in IndicTransToolkit.
"""

from __future__ import annotations

import os
import types
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None

from IndicTransToolkit.processor import IndicProcessor


# Top-20 working set from the plan: 19 canonical present IndicTrans2 langs
# (kas_Arab / mni_Mtei / sat_Olck are ABSENT in HuggingFaceFW/finetranslations
# and are dropped) + Bhojpuri (bho_Deva, fully supported).
LANG_CODES: List[str] = [
    "hin_Deva",  # Hindi
    "ben_Beng",  # Bengali
    "mar_Deva",  # Marathi
    "tam_Taml",  # Tamil
    "npi_Deva",  # Nepali
    "urd_Arab",  # Urdu
    "mal_Mlym",  # Malayalam
    "kan_Knda",  # Kannada
    "guj_Gujr",  # Gujarati
    "tel_Telu",  # Telugu
    "pan_Guru",  # Punjabi
    "ory_Orya",  # Odia
    "snd_Arab",  # Sindhi
    "asm_Beng",  # Assamese
    "bho_Deva",  # Bhojpuri (20th language)
    "mai_Deva",  # Maithili
    "san_Deva",  # Sanskrit
    "gom_Deva",  # Konkani
    "brx_Deva",  # Bodo
    "doi_Deva",  # Dogri
]

DEFAULT_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG = "eng_Latn"


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trim_context(context: str, max_chars: int = 600) -> str:
    """Keep the context block modest so the source chunk survives max_length=256."""
    if not context:
        return ""
    context = " ".join(context.split())
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rsplit(" ", 1)[0] + "…"


def format_glossary_block(glossary_pairs: Sequence[Tuple[str, str]]) -> str:
    if not glossary_pairs:
        return ""
    items = " ; ".join(f"{en} => {tgt}" for en, tgt in glossary_pairs if en and tgt)
    if not items:
        return ""
    return f"Glossary: {items}"


def build_source_with_prefix(
    src: str,
    context: Optional[str] = None,
    glossary_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    source_label: str = "Source",
) -> str:
    """Single source of truth for the context/glossary prefix.

    Matches the B2+ training data construction exactly (see prepare_finetranslations.py).
    Order: Glossary block, then Context, then `Source: <chunk>`.
    """
    context = trim_context(context) if context else ""
    parts: List[str] = []
    glossary_block = format_glossary_block(glossary_pairs)
    if glossary_block:
        parts.append(glossary_block)
    if context:
        parts.append(f"Context: {context}")
    if parts:
        return "\n".join(parts) + f"\n{source_label}: {src}"
    return src


def _patch_past_key_values_bug(model) -> None:
    """Work around IndicTrans2 past_key_values=None bug in some HF builds."""
    original_forward = model.forward

    def _safe_forward(self, *args, **kwargs):
        pkv = kwargs.get("past_key_values", None)
        if pkv is not None:
            if isinstance(pkv, (list, tuple)):
                bad = False
                for layer in pkv:
                    if layer is None:
                        bad = True
                        break
                    if isinstance(layer, (list, tuple)) and any(x is None for x in layer):
                        bad = True
                        break
                if bad:
                    kwargs["past_key_values"] = None
            else:
                kwargs["past_key_values"] = None
        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(_safe_forward, model)


def load_translation_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
    load_4bit: bool = False,
    load_8bit: bool = False,
    adapter_dir: Optional[str] = None,
    max_memory_gb: int = 6,
):
    """Load the IndicTrans2 En->Indic model (+ tokenizer + IndicProcessor).

    If `adapter_dir` is given, a PEFT/LoRA adapter (B2 or B2+) is loaded on top of the
    base model. Returns (model, tokenizer, ip).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if load_4bit and load_8bit:
        raise ValueError("Use only one of load_4bit or load_8bit.")

    quant_config = None
    if (load_4bit or load_8bit) and BitsAndBytesConfig is None:
        raise RuntimeError("bitsandbytes/transformers quantization not available.")
    if load_4bit and BitsAndBytesConfig is not None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif load_8bit and BitsAndBytesConfig is not None:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    is_cuda = device == "cuda"
    if is_cuda:
        max_mem = {0: f"{max_memory_gb}GiB", "cpu": "30GiB"}
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_mem,
            quantization_config=quant_config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=quant_config,
        ).to(device)

    if adapter_dir and os.path.isdir(adapter_dir):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model.eval()

    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    _patch_past_key_values_bug(model)
    ip = IndicProcessor(inference=True)
    return model, tokenizer, ip


def _preprocess(
    sentences: List[str],
    tokenizer,
    ip: IndicProcessor,
    src_lang: str,
    tgt_lang: str,
    max_length: int,
):
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
        max_length=max_length,
    )
    return inputs


def translate_texts(
    model,
    tokenizer,
    ip: IndicProcessor,
    sentences: List[str],
    tgt_lang: str,
    src_lang: str = SRC_LANG,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 256,
    num_beams: int = 5,
    context: Optional[Union[str, List[str]]] = None,
    glossary_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[str]:
    """Translate a batch of English sentences to `tgt_lang`.

    `context` may be a single string (applied to all) or a list aligned to `sentences`.
    """
    if device is None:
        device = get_model_device(model)

    srcs: List[str] = []
    for i, s in enumerate(sentences):
        ctx = context[i] if isinstance(context, list) else context
        srcs.append(build_source_with_prefix(s, context=ctx, glossary_pairs=glossary_pairs))

    inputs = _preprocess(srcs, tokenizer, ip, src_lang, tgt_lang, max_length)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            use_cache=False,
            min_length=0,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return ip.postprocess_batch(decoded, lang=tgt_lang)


def sequence_mean_logprob(
    model,
    tokenizer,
    ip: IndicProcessor,
    srcs: List[str],
    generated_ids: torch.Tensor,
    src_lang: str,
    tgt_lang: str,
    device,
    max_length: int,
) -> List[float]:
    """Per-segment confidence = mean token log-probability of the generated sequence.

    IndicTrans2's `generate` ignores `output_scores` in this transformers build, so we
    score the sequence with a direct encoder+decoder forward pass (decoder teacher-forced
    on the generated ids) and read off the next-token log-probs.
    """
    try:
        dec_start = getattr(model.config, "decoder_start_token_id", None)
        if dec_start is None:
            return [float("nan")] * generated_ids.size(0)

        batch = ip.preprocess_batch(srcs, src_lang=src_lang, tgt_lang=tgt_lang)
        enc = tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt",
            return_attention_mask=True, max_length=max_length,
        )
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}

        B, T = generated_ids.shape
        dec_in = torch.cat(
            [torch.full((B, 1), dec_start, device=device), generated_ids[:, :-1]], dim=1
        )
        with torch.inference_mode():
            out = model(**enc, decoder_input_ids=dec_in, use_cache=False)
        logits = out.logits  # (B, T, V)

        next_tokens = generated_ids[:, 1:]            # (B, T-1)
        shift_logits = logits[:, :-1, :]             # (B, T-1, V)
        logp = F.log_softmax(shift_logits, dim=-1)
        gathered = logp.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        pad_id = tokenizer.pad_token_id
        mask = next_tokens != pad_id
        confs: List[float] = []
        for i in range(B):
            m = mask[i]
            confs.append(float(gathered[i][m].mean().item()) if m.any() else float("nan"))
        return confs
    except Exception:
        return [float("nan")] * generated_ids.size(0)


def translate_texts_with_scores(
    model,
    tokenizer,
    ip: IndicProcessor,
    sentences: List[str],
    tgt_lang: str,
    src_lang: str = SRC_LANG,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 256,
    num_beams: int = 5,
    context: Optional[Union[str, List[str]]] = None,
    glossary_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Tuple[List[str], List[float]]:
    """Like `translate_texts` but also returns per-segment confidence
    (mean token log-probability) for the QE router.
    """
    if device is None:
        device = get_model_device(model)

    srcs: List[str] = []
    for i, s in enumerate(sentences):
        ctx = context[i] if isinstance(context, list) else context
        srcs.append(build_source_with_prefix(s, context=ctx, glossary_pairs=glossary_pairs))

    inputs = _preprocess(srcs, tokenizer, ip, src_lang, tgt_lang, max_length)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            use_cache=False,
            min_length=0,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    translations = ip.postprocess_batch(decoded, lang=tgt_lang)

    confidences = sequence_mean_logprob(
        model, tokenizer, ip, srcs, generated_tokens, src_lang, tgt_lang, device, max_length
    )
    return translations, confidences


def translate_one(
    model, tokenizer, ip, text: str, tgt_lang: str, **kwargs
) -> str:
    return translate_texts(model, tokenizer, ip, [text], tgt_lang, **kwargs)[0]
