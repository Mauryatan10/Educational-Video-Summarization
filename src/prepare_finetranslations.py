"""
prepare_finetranslations.py — build the IndicTrans2 fine-tuning corpus from
HuggingFaceFW/finetranslations (top-20 langs, budgeted subsample).

Pipeline per language config (source = eng_Latn):
  1. stream the `train` split,
  2. skip samples with `early_stop`,
  3. subsample to a per-language document budget (high-resource capped; low-resource
     use all available docs; optional 10k upsampling floor),
  4. carve a validation split (5%, clamped [500, 2000] docs/lang),
  5. EXPLODE each doc into (en_chunk, tgt_chunk) pairs via iter_chunk_pairs,
  6. (B2+) attach an English DOC SUMMARY as context (LED, or cheap extractive
     fallback, or none = plain B2),
  7. write per-language train/val Parquet with columns [source, target, context].

The SAME `build_source_with_prefix` prefix (translate_stage.py) is applied at
training time, so the B2+ context effect transfers to inference.

Outputs land in datasets/finetranslations_prepared/ with a manifest.json.

NOTE on `datasets` shadowing: the repo root has a top-level `datasets/` folder
that shadows the `datasets` package. We strip repo root from sys.path before
importing `datasets` (same trick as train_min_seg.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm


# Per-language document budgets (mirrors the plan §1 table).
# High-resource are capped at 40k docs; low-resource use their full availability.
BUDGETS: Dict[str, int] = {
    "hin_Deva": 40000, "ben_Beng": 40000, "mar_Deva": 40000, "tam_Taml": 40000,
    "npi_Deva": 40000, "urd_Arab": 40000, "mal_Mlym": 40000, "kan_Knda": 40000,
    "guj_Gujr": 40000, "tel_Telu": 40000, "pan_Guru": 40000, "ory_Orya": 40000,
    "snd_Arab": 40000, "asm_Beng": 40000,
    "bho_Deva": 17894, "mai_Deva": 22723, "san_Deva": 21443, "gom_Deva": 5764,
    "brx_Deva": 3260, "doi_Deva": 1170,
}

SCHEMA = pa.schema([
    ("source", pa.string()),
    ("target", pa.string()),
    ("context", pa.string()),
])

MAX_CHUNK_BYTES = 15 * 1024 * 1024 * 1024  # 15 GB ceiling per chunk file


def iter_chunk_pairs(sample: Dict) -> Iterable[Tuple[str, str]]:
    eng_chunks = sample.get("translated_chunks")
    tgt_chunks = sample.get("og_chunks")
    if isinstance(eng_chunks, list) and isinstance(tgt_chunks, list):
        for en, tgt in zip(eng_chunks, tgt_chunks):
            if isinstance(en, str) and isinstance(tgt, str) and en.strip() and tgt.strip():
                yield en, tgt
        return
    en = sample.get("translated_text")
    tgt = sample.get("og_full_text")
    if isinstance(en, str) and isinstance(tgt, str) and en.strip() and tgt.strip():
        yield en, tgt


def doc_context(
    sample: Dict,
    mode: str,
    led_model=None,
    led_tokenizer=None,
) -> str:
    """English summary context for one document, per context-mode."""
    if mode == "none":
        return ""
    text = sample.get("translated_text") or ""
    if not text.strip():
        chunks = sample.get("translated_chunks") or []
        text = " ".join(c for c in chunks if isinstance(c, str))
    if mode == "extractive":
        sents = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        return ". ".join(sents[:3]).strip()
    if mode == "led":
        if led_model is None:
            raise RuntimeError("context-mode 'led' requires a loaded LED model.")
        from inference import generate_summary
        summary = generate_summary(
            text, model=led_model, tokenizer=led_tokenizer, adapter="hier",
            batch_size=1, max_target=320,
        )
        return (summary or "").strip()
    raise ValueError(f"Unknown context-mode: {mode!r}")


def build_parquet_writer(path: str) -> pq.ParquetWriter:
    return pq.ParquetWriter(path, SCHEMA)


def write_rows(writer: pq.ParquetWriter, rows: List[Tuple[str, str, str]]) -> None:
    if not rows:
        return
    arrays = [
        pa.array([r[0] for r in rows], type=pa.string()),
        pa.array([r[1] for r in rows], type=pa.string()),
        pa.array([r[2] for r in rows], type=pa.string()),
    ]
    writer.write_table(pa.Table.from_arrays(arrays, schema=SCHEMA))


def process_language(
    lang_code: str,
    budget: int,
    output_dir: str,
    context_mode: str,
    val_frac: float,
    val_min: int,
    val_max: int,
    seed: int,
    limit_per_lang: Optional[int],
    upsample_floor: Optional[int],
    led_model=None,
    led_tokenizer=None,
) -> Dict:
    rng = random.Random(seed + hash(lang_code) % (2 ** 31))
    effective_budget = limit_per_lang if limit_per_lang else budget
    if upsample_floor and effective_budget < upsample_floor:
        effective_budget = upsample_floor

    val_target = int(round(effective_budget * val_frac))
    # Only apply the absolute floor/ceiling when the budget is large enough that
    # they make sense; otherwise the floor (e.g. 500) would swallow the whole split.
    if effective_budget > val_min * 2:
        val_target = max(val_target, min(val_max, val_min))
    val_target = max(0, min(val_target, val_max, effective_budget - 1))
    stride = max(1, effective_budget // max(1, val_target)) if val_target else 0

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"train-{lang_code}.parquet")
    val_path = os.path.join(output_dir, f"val-{lang_code}.parquet")
    train_writer = build_parquet_writer(train_path)
    val_writer = build_parquet_writer(val_path)

    doc_counter = 0
    val_docs = 0
    train_docs = 0
    train_pairs = 0
    val_pairs = 0
    train_buf: List[Tuple[str, str, str]] = []
    val_buf: List[Tuple[str, str, str]] = []
    BUF = 2000

    ds_stream = load_dataset(
        "HuggingFaceFW/finetranslations", name=lang_code, split="train", streaming=True
    )

    with tqdm(total=effective_budget, desc=f"[{lang_code}] docs", dynamic_ncols=True) as pbar:
        for sample in ds_stream:
            if sample.get("early_stop") is True:
                continue
            if doc_counter >= effective_budget:
                break

            if stride and (doc_counter % stride == 0) and val_docs < val_target:
                is_val = True
            else:
                is_val = False

            ctx = doc_context(sample, context_mode, led_model, led_tokenizer)
            writer = val_writer if is_val else train_writer
            buf = val_buf if is_val else train_buf
            added = 0
            for en, tgt in iter_chunk_pairs(sample):
                buf.append((en, tgt, ctx))
                added += 1
            if is_val:
                val_docs += 1
                val_pairs += added
            else:
                train_docs += 1
                train_pairs += added

            if len(train_buf) >= BUF:
                write_rows(train_writer, train_buf)
                train_buf.clear()
            if len(val_buf) >= BUF:
                write_rows(val_writer, val_buf)
                val_buf.clear()

            doc_counter += 1
            pbar.update(1)

    write_rows(train_writer, train_buf)
    write_rows(val_writer, val_buf)
    train_writer.close()
    val_writer.close()

    train_size = os.path.getsize(train_path)
    val_size = os.path.getsize(val_path)
    assert train_size <= MAX_CHUNK_BYTES, f"{train_path} exceeds 15 GB chunk ceiling"
    assert val_size <= MAX_CHUNK_BYTES, f"{val_path} exceeds 15 GB chunk ceiling"

    return {
        "lang": lang_code,
        "budget_docs": effective_budget,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "context_mode": context_mode,
        "train_path": train_path,
        "val_path": val_path,
        "train_size_bytes": train_size,
        "val_size_bytes": val_size,
    }


def parse_langs(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IndicTrans2 fine-tuning corpus.")
    parser.add_argument("--langs", type=parse_langs, default=None,
                        help="Comma-separated subset of lang codes; default = all in BUDGETS.")
    parser.add_argument("--output-dir",
                        default=os.path.join(REPO_ROOT, "datasets", "finetranslations_prepared"))
    parser.add_argument("--context-mode", choices=["none", "extractive", "led"], default="led",
                        help="B2 (none) vs B2+ context (led/extractive).")
    parser.add_argument("--led-base-model", default="allenai/led-base-16384")
    parser.add_argument("--led-hier-dir",
                        default=os.path.join(REPO_ROOT, "models", "led-tib-hier", "led_hier_adapter_run"))
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--val-min", type=int, default=500)
    parser.add_argument("--val-max", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-lang", type=int, default=None,
                        help="Smoke-test cap on docs/lang (ignores budget).")
    parser.add_argument("--upsample-floor", type=int, default=None,
                        help="Optional low-resource floor (e.g. 10000).")
    args = parser.parse_args()

    setattr(sys.modules[__name__], "_REPO_ROOT", REPO_ROOT)
    lang_codes = args.langs if args.langs else list(BUDGETS.keys())
    missing = [l for l in lang_codes if l not in BUDGETS]
    if missing:
        raise ValueError(f"Unknown lang codes (not in BUDGETS): {missing}")

    config_names = set(get_dataset_config_names("HuggingFaceFW/finetranslations"))
    lang_codes = [l for l in lang_codes if l in config_names]
    if not lang_codes:
        raise RuntimeError("No requested languages exist in HuggingFaceFW/finetranslations.")

    led_model = led_tokenizer = None
    if args.context_mode == "led":
        from inference import load_model
        print("Loading LED summarizer for context...")
        led_model, led_tokenizer = load_model(
            base_model_name=args.led_base_model, adapter="hier", hier_dir=args.led_hier_dir
        )

    manifest: Dict = {"languages": {}, "context_mode": args.context_mode,
                      "limit_per_lang": args.limit_per_lang}
    for lang in lang_codes:
        print(f"\n=== Preparing {lang} ===")
        info = process_language(
            lang_code=lang,
            budget=BUDGETS[lang],
            output_dir=args.output_dir,
            context_mode=args.context_mode,
            val_frac=args.val_frac,
            val_min=args.val_min,
            val_max=args.val_max,
            seed=args.seed,
            limit_per_lang=args.limit_per_lang,
            upsample_floor=args.upsample_floor,
            led_model=led_model,
            led_tokenizer=led_tokenizer,
        )
        manifest["languages"][lang] = info
        print(json.dumps(info, ensure_ascii=False, indent=2))

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
