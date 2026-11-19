#!/usr/bin/env python3
"""
scripts/evaluate_script_constrained_model.py

Evaluates the new Script-Constrained QLoRA model (trained on diversified
10-15 sentence paragraphs with forbidden Indic logit suppression)
on all 35 validation subtopics in data/subtopic_diversified/val_subtopics.parquet.

Measures:
  - Flores200 & 13a SacreBLEU (Reported and Raw unpenalized)
  - chrF++ (word_order=2)
  - Brevity Penalty (BP) & System-to-Reference Length Ratio
  - 1-gram to 4-gram Precisions (Overall and STEM-specific)
  - Script Purity Audit: Counts and percentages of Devanagari, Latin, and Forbidden Indic glyphs
  - Side-by-side delta against the previous unconstrained QLoRA checkpoint

Outputs:
  - data/subtopic_translation_eval/script_constrained_val_translations.json
  - data/subtopic_translation_eval/script_constrained_val_metrics.json
"""

import os
import sys
import time
import json
import torch
import pyarrow.parquet as pq
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Strip local dir from sys.path to avoid shadowing third-party 'datasets'
sys.path = [p for p in sys.path if p not in ('', '.', os.path.dirname(os.path.abspath('.')))]

BASE_MODEL_ID = "sarvamai/sarvam-1"
ADAPTER_PATH = "models/sarvam1_subtopic_script_constrained/best_checkpoint"
VAL_SUBTOPICS_PATH = "data/subtopic_diversified/val_subtopics.parquet"
OUTPUT_DIR = "data/subtopic_translation_eval"
OUTPUT_RECORDS_FILE = os.path.join(OUTPUT_DIR, "script_constrained_val_translations.json")
OUTPUT_METRICS_FILE = os.path.join(OUTPUT_DIR, "script_constrained_val_metrics.json")
PREV_METRICS_FILE = os.path.join(OUTPUT_DIR, "finetuned_sarvam1_val_metrics.json")


def build_translation_prompt(en_text: str) -> str:
    return f"अंग्रेजी:\n{en_text}\n\nहिन्दी (देवनागरी लिपि में अनुवाद):\n"


def audit_script_composition(text: str) -> dict:
    """Audits the character composition of generated text across Unicode blocks."""
    counts = {
        "devanagari": 0,
        "latin": 0,
        "forbidden_indic": 0,
        "digits": 0,
        "punctuation_whitespace": 0,
        "total": len(text)
    }

    for c in text:
        cp = ord(c)
        if 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif (
            (0x0A00 <= cp <= 0x0A7F) or  # Gurmukhi
            (0x0B80 <= cp <= 0x0BFF) or  # Tamil
            (0x0C00 <= cp <= 0x0C7F) or  # Telugu
            (0x0C80 <= cp <= 0x0CFF) or  # Kannada
            (0x0D00 <= cp <= 0x0D7F) or  # Malayalam
            (0x0980 <= cp <= 0x09FF) or  # Bengali / Assamese
            (0x0A80 <= cp <= 0x0AFF) or  # Gujarati
            (0x0B00 <= cp <= 0x0B7F)     # Oriya
        ):
            counts["forbidden_indic"] += 1
        elif ('a' <= c <= 'z') or ('A' <= c <= 'Z'):
            counts["latin"] += 1
        elif c.isdigit():
            counts["digits"] += 1
        else:
            counts["punctuation_whitespace"] += 1

    return counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=str, default=ADAPTER_PATH, help="Path to adapter checkpoint")
    parser.add_argument("--output-tag", type=str, default="best_checkpoint", help="Tag for output files")
    args = parser.parse_args()

    adapter_path = args.adapter_path
    output_records_file = os.path.join(OUTPUT_DIR, f"script_constrained_{args.output_tag}_translations.json")
    output_metrics_file = os.path.join(OUTPUT_DIR, f"script_constrained_{args.output_tag}_metrics.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"=== EVALUATING SCRIPT-CONSTRAINED QLORA ({args.output_tag}) ON 35 VALIDATION SUBTOPICS ===")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loading checkpoint from: {adapter_path}")

    # 1. Load Validation Subtopics
    table = pq.read_table(VAL_SUBTOPICS_PATH)
    val_subtopics = table.to_pylist()
    print(f"Loaded {len(val_subtopics)} validation subtopics from {VAL_SUBTOPICS_PATH}.")

    # 2. Load Model in 4-bit NF4 with Adapter
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"Model loaded in {time.time() - t0_load:.2f}s | Active VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    # 3. Run Inference on All 35 Subtopics
    t0_infer = time.time()
    records = []
    total_forbidden_chars = 0
    total_deva_chars = 0
    total_latin_chars = 0

    for idx, subtopic in enumerate(val_subtopics):
        en_text = subtopic["src_en"]
        gold_hi = subtopic["tgt_hi"]
        subtopic_id = subtopic.get("subtopic_id", f"val_{idx:06d}")
        is_stem = subtopic.get("is_stem", False)
        stem_terms = subtopic.get("stem_terms", [])
        num_sentences = subtopic.get("num_sentences", 15)

        prompt = build_translation_prompt(en_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = outputs[0][prompt_len:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean prompt markers if any
        for stop_marker in [
            "\nअंग्रेजी:", "\n\nअंग्रेजी:", "\nEnglish:", "\n\nEnglish:",
            "\nहिन्दी", "\n\nहिन्दी", "\nTranslate", "\n\nTranslate",
            "\nTask:", "\nExplanation:", "\nNote:", "\n###"
        ]:
            if stop_marker in decoded:
                decoded = decoded.split(stop_marker)[0].strip()

        # Audit script composition
        script_comp = audit_script_composition(decoded)
        total_forbidden_chars += script_comp["forbidden_indic"]
        total_deva_chars += script_comp["devanagari"]
        total_latin_chars += script_comp["latin"]

        # Sentence / word level stats
        hyp_words = len(decoded.split())
        gold_words = len(gold_hi.split())
        en_words = len(en_text.split())

        records.append({
            "idx": idx,
            "subtopic_id": subtopic_id,
            "is_stem": is_stem,
            "stem_terms": stem_terms,
            "num_sentences": num_sentences,
            "word_count_en": en_words,
            "word_count_gold_hi": gold_words,
            "word_count_hyp_hi": hyp_words,
            "length_ratio": round(hyp_words / max(1, gold_words), 3),
            "script_composition": script_comp,
            "src_en": en_text,
            "gold_hi": gold_hi,
            "finetuned_hi": decoded
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(val_subtopics):
            print(f"[{idx+1}/{len(val_subtopics)}] Done (Length ratio: {hyp_words}/{gold_words} = {hyp_words/max(1,gold_words):.2f} | Forbidden chars: {script_comp['forbidden_indic']})")

    t_infer = time.time() - t0_infer
    print(f"\nInference on 35 subtopics finished in {t_infer:.2f}s ({len(val_subtopics)/t_infer:.2f} subtopics/s).")

    # 4. Compute Metrics across Validation Set
    all_hyp = [r["finetuned_hi"] for r in records]
    all_ref = [[r["gold_hi"] for r in records]]

    bleu_flores = sacrebleu.corpus_bleu(all_hyp, all_ref, tokenize="flores200")
    bleu_13a = sacrebleu.corpus_bleu(all_hyp, all_ref, tokenize="13a")
    chrf_pp = sacrebleu.corpus_chrf(all_hyp, all_ref, word_order=2)
    raw_bleu_flores = (bleu_flores.score / bleu_flores.bp) if bleu_flores.bp > 0 else 0.0

    stem_hyp = [r["finetuned_hi"] for r in records if r["is_stem"]]
    stem_ref = [[r["gold_hi"] for r in records if r["is_stem"]]]
    stem_bleu_flores = sacrebleu.corpus_bleu(stem_hyp, stem_ref, tokenize="flores200") if stem_hyp else None
    stem_chrf = sacrebleu.corpus_chrf(stem_hyp, stem_ref, word_order=2) if stem_hyp else None
    stem_raw_bleu = (stem_bleu_flores.score / stem_bleu_flores.bp) if (stem_bleu_flores and stem_bleu_flores.bp > 0) else 0.0

    # Script Purity
    total_audited_chars = total_deva_chars + total_forbidden_chars + total_latin_chars
    script_purity = {
        "total_devanagari_chars": total_deva_chars,
        "total_latin_chars": total_latin_chars,
        "total_forbidden_indic_chars": total_forbidden_chars,
        "forbidden_indic_char_rate": round(total_forbidden_chars / max(1, total_audited_chars) * 100, 4),
        "devanagari_purity_rate": round(total_deva_chars / max(1, total_audited_chars) * 100, 2)
    }

    metrics_summary = {
        "val_samples_count": len(records),
        "stem_samples_count": len(stem_hyp),
        "inference_time_sec": round(t_infer, 2),
        "overall": {
            "reported_bleu_flores200": round(bleu_flores.score, 2),
            "raw_bleu_flores200": round(raw_bleu_flores, 2),
            "bleu_13a": round(bleu_13a.score, 2),
            "chrf_plus_plus": round(chrf_pp.score, 2),
            "brevity_penalty": round(bleu_flores.bp, 3),
            "sys_len": bleu_flores.sys_len,
            "ref_len": bleu_flores.ref_len,
            "len_ratio": round(bleu_flores.sys_len / max(1, bleu_flores.ref_len), 3),
            "precisions_1_to_4_gram": [round(p, 2) for p in bleu_flores.precisions]
        },
        "stem": {
            "reported_bleu_flores200": round(stem_bleu_flores.score, 2) if stem_bleu_flores else None,
            "raw_bleu_flores200": round(stem_raw_bleu, 2) if stem_bleu_flores else None,
            "chrf_plus_plus": round(stem_chrf.score, 2) if stem_chrf else None,
            "brevity_penalty": round(stem_bleu_flores.bp, 3) if stem_bleu_flores else None,
            "precisions_1_to_4_gram": [round(p, 2) for p in stem_bleu_flores.precisions] if stem_bleu_flores else None
        },
        "script_purity": script_purity
    }

    # Load Previous Metrics for Direct Comparison
    prev_metrics = None
    if os.path.exists(PREV_METRICS_FILE):
        try:
            with open(PREV_METRICS_FILE) as f:
                prev_metrics = json.load(f)
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("=== QUANTITATIVE BENCHMARK RESULTS (NEW SCRIPT-CONSTRAINED MODEL) ===")
    print("=" * 80)
    print(f"Reported Flores200 BLEU: {metrics_summary['overall']['reported_bleu_flores200']:.2f}")
    print(f"Raw Flores200 BLEU:      {metrics_summary['overall']['raw_bleu_flores200']:.2f}")
    print(f"chrF++ (w=2):            {metrics_summary['overall']['chrf_plus_plus']:.2f}")
    print(f"Brevity Penalty (BP):    {metrics_summary['overall']['brevity_penalty']:.3f}")
    print(f"Length Ratio (Sys/Ref):  {metrics_summary['overall']['len_ratio']:.3f} ({bleu_flores.sys_len}/{bleu_flores.ref_len} tokens)")
    print(f"1-Gram Precision:        {bleu_flores.precisions[0]:.2f}%")
    print(f"Forbidden Indic Chars:   {total_forbidden_chars} ({script_purity['forbidden_indic_char_rate']}%)")
    print(f"Devanagari Char Purity:  {script_purity['devanagari_purity_rate']}%")

    if prev_metrics:
        prev_rep_bleu = prev_metrics["overall"]["bleu_flores200"]
        prev_raw_bleu = prev_metrics["overall"]["raw_bleu_flores200"]
        prev_chrf = prev_metrics["overall"]["chrf_plus_plus"]
        prev_bp = prev_metrics["overall"]["bp"]
        prev_1gram = prev_metrics["overall"]["precisions"][0]

        print("\n--- Comparative Progression vs Previous Model ---")
        print(f"Reported BLEU:   {prev_rep_bleu:.2f} -> {metrics_summary['overall']['reported_bleu_flores200']:.2f} (Delta: {metrics_summary['overall']['reported_bleu_flores200'] - prev_rep_bleu:+.2f})")
        print(f"Raw BLEU:        {prev_raw_bleu:.2f} -> {metrics_summary['overall']['raw_bleu_flores200']:.2f} (Delta: {metrics_summary['overall']['raw_bleu_flores200'] - prev_raw_bleu:+.2f})")
        print(f"chrF++ (w=2):    {prev_chrf:.2f} -> {metrics_summary['overall']['chrf_plus_plus']:.2f} (Delta: {metrics_summary['overall']['chrf_plus_plus'] - prev_chrf:+.2f})")
        print(f"Brevity Penalty: {prev_bp:.3f} -> {metrics_summary['overall']['brevity_penalty']:.3f} (Delta: {metrics_summary['overall']['brevity_penalty'] - prev_bp:+.3f})")
        print(f"1-Gram Prec:     {prev_1gram:.2f}% -> {bleu_flores.precisions[0]:.2f}% (Delta: {bleu_flores.precisions[0] - prev_1gram:+.2f}%)")

    # Save outputs
    with open(output_records_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Detailed validation translations to {output_records_file}")

    with open(output_metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Full validation metrics summary to {output_metrics_file}")


if __name__ == "__main__":
    main()
