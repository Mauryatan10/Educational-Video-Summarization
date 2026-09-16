"""
train_indictrans2.py — QLoRA fine-tune of ai4bharat/indictrans2-en-indic-1B.

Implements Decision C: train BOTH variants on the corpus from prepare_finetranslations.py
  * B2  (plain)   : (source=chunk) -> target
  * B2+ (context) : (source="Context: <doc_summary>\nSource: <chunk>") -> target,
                    where the doc summary is the context column written by prepare_finetranslations.

The source string is built with the SAME `build_source_with_prefix` used at inference
(translate_stage.py), so the B2+ context effect transfers to app.py / translate_stage.

Tokenization uses IndicProcessor exactly as at inference:
  source -> ip.preprocess_batch(src, src_lang="eng_Latn", tgt_lang=<lang>, is_target=False)
  target -> ip.preprocess_batch(tgt, src_lang=<lang>,      tgt_lang=<lang>, is_target=True)

Outputs (LoRA adapters only):
  B2  -> models/indictrans2-en-indic-1B-qlora/
  B2+ -> models/indictrans2-en-indic-1B-qlora-ctx/

NOTE: strip repo root from sys.path before importing `datasets` (top-level `datasets/`
folder shadows the package) — same trick as train_min_seg.py / prepare_finetranslations.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)

import torch
from datasets import Dataset, concatenate_datasets
import pyarrow.parquet as pq
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from IndicTransToolkit.processor import IndicProcessor

from translate_stage import build_source_with_prefix, DEFAULT_MODEL_NAME, SRC_LANG

try:
    from sacrebleu.metrics import CHRF as _CHRF
except Exception:  # pragma: no cover
    _CHRF = None

# Lazy COMET loader (cached). Requires the `comet` package + the Unbabel/wmt22-comet-da
# model; if unavailable we fall back to chrF + TA and skip COMET.
_COMET = None


def _get_comet():
    global _COMET
    if _COMET is False:
        return None
    if _COMET is None:
        try:
            import evaluate

            _COMET = evaluate.load("comet", "Unbabel/wmt22-comet-da")
        except Exception as e:  # pragma: no cover
            print(f"[compute_metrics] COMET unavailable, skipping: {e}")
            _COMET = False
    return _COMET if _COMET else None


_STOP = set(
    "the a an and or of to in for with on at by from as is are was were be been being "
    "this that these those it its we you they he she them their our your his her not have "
    "has had do does did will would can could should may might must also more most very "
    "over under between into about".split()
)


def _content_terms(text: str) -> set:
    toks = re.findall(r"[A-Za-z]+", (text or "").lower())
    return {w for w in toks if len(w) >= 4 and w not in _STOP}


def _ta_proxy(pred_texts, ref_texts) -> float:
    """Terminology Accuracy proxy: recall of reference content terms (>=4 letters,
    non-stopword) present in the prediction. No explicit glossary needed; measures
    whether key terms survive translation. Range [0, 1]."""
    scores = []
    for p, r in zip(pred_texts, ref_texts):
        rt = _content_terms(r)
        if not rt:
            scores.append(1.0)
            continue
        pt = _content_terms(p)
        scores.append(len(rt & pt) / len(rt))
    return float(sum(scores) / len(scores)) if scores else float("nan")


def make_compute_metrics(tokenizer, val_sources: Optional[List[str]], use_comet: bool, use_ta: bool):
    """Build a compute_metrics fn for Seq2SeqTrainer.

    - chrF: character-level F-score (best string-overlap metric for Indic).
    - COMET (Unbabel/wmt22-comet-da): neural reference-based quality (needs sources).
    - TA: terminology-accuracy proxy vs the reference (see _ta_proxy).
    `val_sources` aligns by position with the eval batch (Trainer evaluates in order).
    """

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if hasattr(preds, "ndim") and preds.ndim == 3:
            preds = preds[:, 0, :]
        labels = eval_pred.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        pred_texts = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        ref_texts = tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        n = len(pred_texts)
        out = {}

        if _CHRF is not None:
            chrf = _CHRF(word_order=2)  # chrF++
            out["chrf"] = float(chrf.corpus_score(pred_texts, [ref_texts]).score)
        else:
            out["chrf"] = float("nan")

        if use_comet:
            cm = _get_comet()
            if cm is not None:
                srcs = (val_sources or [""] * n)[:n]
                out["comet"] = float(
                    cm.compute(predictions=pred_texts, references=ref_texts, sources=srcs)["score"]
                )
            else:
                out["comet"] = float("nan")

        if use_ta:
            out["ta"] = _ta_proxy(pred_texts, ref_texts)

        return out

    return compute_metrics

LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
MAX_SOURCE = 256
MAX_TARGET = 256


def load_prepared(prepared_dir: str, split: str, langs: Optional[List[str]]) -> Dataset:
    files = sorted(glob.glob(os.path.join(prepared_dir, f"{split}-*.parquet")))
    if langs:
        wanted = {f"{split}-{l}.parquet" for l in langs}
        files = [f for f in files if os.path.basename(f) in wanted]
    if not files:
        raise FileNotFoundError(f"No {split}-*.parquet found in {prepared_dir}")

    shards = []
    for f in files:
        lang = os.path.basename(f)[len(split) + 1 : -len(".parquet")]
        d = Dataset.from_parquet(f)
        d = d.add_column("tgt_lang", [lang] * len(d))
        shards.append(d)
    return concatenate_datasets(shards)


def make_preprocess(ip: IndicProcessor, tokenizer, variant: str):
    def _fn(examples):
        sources = examples["source"]
        contexts = examples.get("context", [""] * len(sources))
        targets = examples["target"]
        lang_list = examples["tgt_lang"]

        src_pp, tgt_pp = [], []
        for src, ctx, tgt, lang in zip(sources, contexts, targets, lang_list):
            if variant == "b2p" and ctx:
                src_text = build_source_with_prefix(src, context=ctx)
            else:
                src_text = src
            # Per-example preprocessing (each row may have a different target language).
            src_pp += ip.preprocess_batch([src_text], src_lang=SRC_LANG, tgt_lang=lang, is_target=False)
            tgt_pp += ip.preprocess_batch([tgt], src_lang=lang, tgt_lang=lang, is_target=True)

        model_inputs = tokenizer(
            src_pp, max_length=MAX_SOURCE, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                tgt_pp, max_length=MAX_TARGET, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune IndicTrans2 (B2 / B2+).")
    parser.add_argument("--variant", choices=["b2", "b2p"], required=True,
                        help="b2 = plain; b2p = context-augmented (Decision C).")
    parser.add_argument("--prepared-dir",
                        default=os.path.join(REPO_ROOT, "datasets", "finetranslations_prepared"))
    parser.add_argument("--langs", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                        default=None, help="Subset of langs (parquet file stems).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", default=None,
                        help="Adapter output dir. Default: models/indictrans2-en-indic-1B-qlora[/ctx].")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training examples (smoke test / quick runs).")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=50,
                        help="Evaluate every N optimizer steps (not per epoch).")
    parser.add_argument("--no-comet", action="store_true",
                        help="Disable COMET in compute_metrics (chrF + TA still logged).")
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-memory-gb", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        suffix = "qlora-ctx" if args.variant == "b2p" else "qlora"
        args.output_dir = os.path.join(REPO_ROOT, "models", f"indictrans2-en-indic-1B-{suffix}")

    print(f"Variant={args.variant}  prepared={args.prepared_dir}  out={args.output_dir}", flush=True)

    train_ds = load_prepared(args.prepared_dir, "train", args.langs)
    try:
        val_ds = load_prepared(args.prepared_dir, "val", args.langs)
    except FileNotFoundError:
        val_ds = None
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if val_ds is not None and args.max_val_samples:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))
    # Keep raw sources (position-aligned with eval batches) for COMET/TA reference.
    val_sources = list(val_ds["source"]) if val_ds is not None else None
    print(f"Train examples: {len(train_ds)}  Val examples: {len(val_ds) if val_ds else 0}", flush=True)

    ip = IndicProcessor(inference=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    compute_metrics = (
        make_compute_metrics(tokenizer, val_sources, use_comet=not args.no_comet, use_ta=True)
        if val_ds is not None else None
    )

    train_ds = train_ds.map(
        make_preprocess(ip, tokenizer, args.variant),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    if val_ds is not None:
        val_ds = val_ds.map(
            make_preprocess(ip, tokenizer, args.variant),
            batched=True,
            remove_columns=val_ds.column_names,
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: f"{args.max_memory_gb}GiB", "cpu": "30GiB"},
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=args.eval_steps,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET,
        report_to="none",
        seed=args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
