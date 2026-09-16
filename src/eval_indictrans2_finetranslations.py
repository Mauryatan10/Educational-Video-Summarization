import argparse
import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import torch
from datasets import get_dataset_config_names, load_dataset
import evaluate
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None
import types
from IndicTransToolkit.processor import IndicProcessor



# IndicTrans2 language codes from the official repo (excluding English).
# For languages with multiple scripts (Kashmiri, Manipuri, Sindhi) we pick the
# most widely used native script by default. You can override with --langs.
DEFAULT_22_LANGS = [
    "asm_Beng",  # Assamese
    "ben_Beng",  # Bengali
    "brx_Deva",  # Bodo
    "doi_Deva",  # Dogri
    "gom_Deva",  # Konkani (Goan Konkani)
    "guj_Gujr",  # Gujarati
    "hin_Deva",  # Hindi
    "kan_Knda",  # Kannada
    "kas_Arab",  # Kashmiri (Perso-Arabic)
    "mai_Deva",  # Maithili
    "mal_Mlym",  # Malayalam
    "mar_Deva",  # Marathi
    "mni_Mtei",  # Manipuri (Meitei)
    "npi_Deva",  # Nepali
    "ory_Orya",  # Odia
    "pan_Guru",  # Punjabi
    "san_Deva",  # Sanskrit
    "sat_Olck",  # Santali (Ol Chiki)
    "snd_Arab",  # Sindhi (Perso-Arabic)
    "tam_Taml",  # Tamil
    "tel_Telu",  # Telugu
    "urd_Arab",  # Urdu
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_chunk_pairs(sample: Dict) -> Iterable[Tuple[str, str]]:
    eng_chunks = sample.get("translated_chunks")
    tgt_chunks = sample.get("og_chunks")
    if isinstance(eng_chunks, list) and isinstance(tgt_chunks, list):
        # Only keep aligned pairs
        for en, tgt in zip(eng_chunks, tgt_chunks):
            if isinstance(en, str) and isinstance(tgt, str) and en.strip() and tgt.strip():
                yield en, tgt
        return
    # Fallback to full text if chunks are missing
    en = sample.get("translated_text")
    tgt = sample.get("og_full_text")
    if isinstance(en, str) and isinstance(tgt, str) and en.strip() and tgt.strip():
        yield en, tgt


def collect_parallel_samples(
    ds_stream,
    max_samples: int,
    lang_code: str,
) -> Tuple[List[str], List[str]]:
    sources: List[str] = []
    targets: List[str] = []
    with tqdm(
        total=max_samples,
        desc=f"[{lang_code}] Collecting pairs",
        unit="pair",
        dynamic_ncols=True,
    ) as pbar:
        for sample in ds_stream:
            # Skip samples where translation stopped early
            if sample.get("early_stop") is True:
                continue
            for en, tgt in iter_chunk_pairs(sample):
                sources.append(en)
                targets.append(tgt)
                pbar.update(1)
                if len(sources) >= max_samples:
                    return sources, targets
    return sources, targets


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def translate_batch(
    model,
    tokenizer,
    ip: IndicProcessor,
    sentences: List[str],
    src_lang: str,
    tgt_lang: str,
    device: str,
    max_length: int,
    num_beams: int,
) -> List[str]:
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    )
    if device == "cuda":
        inputs = inputs.to(device)
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
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return ip.postprocess_batch(decoded, lang=tgt_lang)


def evaluate_language(
    model,
    tokenizer,
    ip: IndicProcessor,
    lang_code: str,
    max_samples: int,
    batch_size: int,
    device: str,
    max_length: int,
    num_beams: int,
) -> Dict:
    ds_stream = load_dataset(
        "HuggingFaceFW/finetranslations",
        name=lang_code,
        split="train",
        streaming=True,
    )
    sources, targets = collect_parallel_samples(
        ds_stream, max_samples=max_samples, lang_code=lang_code
    )
    if not sources:
        return {"lang": lang_code, "samples": 0}

    predictions: List[str] = []
    total_batches = (len(sources) + batch_size - 1) // batch_size
    with tqdm(
        total=total_batches,
        desc=f"[{lang_code}] Translating",
        unit="batch",
        dynamic_ncols=True,
    ) as pbar:
        for batch in batched(sources, batch_size):
            predictions.extend(
                translate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    ip=ip,
                    sentences=batch,
                    src_lang="eng_Latn",
                    tgt_lang=lang_code,
                    device=device,
                    max_length=max_length,
                    num_beams=num_beams,
                )
            )
            pbar.update(1)

    refs = [[t] for t in targets]
    bleu = evaluate.load("sacrebleu").compute(predictions=predictions, references=refs)
    chrf = evaluate.load("chrf").compute(predictions=predictions, references=refs)
    ter = evaluate.load("ter").compute(predictions=predictions, references=refs)

    return {
        "lang": lang_code,
        "samples": len(predictions),
        "bleu": bleu.get("score"),
        "chrf": chrf.get("score"),
        "ter": ter.get("score"),
    }


def parse_langs(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IndicTrans2 En->Indic on FineTranslations."
    )
    parser.add_argument(
        "--model_name",
        default="ai4bharat/indictrans2-en-indic-1B",
        help="HF model name or path.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max parallel chunks per language.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load model in 8-bit with bitsandbytes (reduces VRAM).",
    )
    parser.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load model in 4-bit with bitsandbytes (further reduces VRAM).",
    )
    parser.add_argument(
        "--max_memory_gb",
        type=int,
        default=6,
        help="GPU VRAM budget for device_map auto placement.",
    )
    parser.add_argument(
        "--langs",
        type=parse_langs,
        default=None,
        help="Comma-separated list of language codes to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "indictrans2_finetranslations_eval.json"),
        help="Where to write the JSON results.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if args.load_4bit and args.load_8bit:
        raise ValueError("Use only one of --load_4bit or --load_8bit.")

    quant_config = None
    if (args.load_4bit or args.load_8bit) and BitsAndBytesConfig is None:
        raise RuntimeError(
            "bitsandbytes/transformers quantization not available. "
            "Install bitsandbytes or run without --load_4bit/--load_8bit."
        )
    if args.load_4bit and BitsAndBytesConfig is not None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif args.load_8bit and BitsAndBytesConfig is not None:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    if device == "cuda":
        max_mem = {0: f"{args.max_memory_gb}GiB", "cpu": "30GiB"}
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_mem,
            quantization_config=quant_config,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=quant_config,
        ).to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    # Work around IndicTrans2 past_key_values=None bug in some HF module builds.
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
    ip = IndicProcessor(inference=True)

    config_names = set(get_dataset_config_names("HuggingFaceFW/finetranslations"))
    lang_list = args.langs if args.langs else DEFAULT_22_LANGS
    lang_list = [lang for lang in lang_list if lang in config_names]

    results: Dict[str, Dict] = {}
    for lang in lang_list:
        print(f"\n=== Evaluating {lang} ===")
        results[lang] = evaluate_language(
            model=model,
            tokenizer=tokenizer,
            ip=ip,
            lang_code=lang,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=device,
            max_length=args.max_length,
            num_beams=args.num_beams,
        )

    # Overall (macro) averages for languages with samples
    valid = [v for v in results.values() if v.get("samples", 0) > 0]
    if valid:
        results["overall"] = {
            "langs": len(valid),
            "samples": sum(v["samples"] for v in valid),
            "bleu": sum(v["bleu"] for v in valid) / len(valid),
            "chrf": sum(v["chrf"] for v in valid) / len(valid),
            "ter": sum(v["ter"] for v in valid) / len(valid),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
