#!/usr/bin/env python3
"""
scripts/translate_val_subtopics.py

Runs inference on all 35 validation subtopics in data/subtopic_engineered/val_subtopics.parquet
using the fine-tuned Sarvam-1 QLoRA model.
Saves predictions, gold references, and error diagnostic metrics to
data/subtopic_translation_eval/finetuned_sarvam1_val_translations.json
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
ADAPTER_PATH = "models/sarvam1_subtopic_qlora/best_checkpoint"
VAL_SUBTOPICS_PATH = "data/subtopic_engineered/val_subtopics.parquet"
OUTPUT_FILE = "data/subtopic_translation_eval/finetuned_sarvam1_val_translations.json"
METRICS_OUTPUT_FILE = "data/subtopic_translation_eval/finetuned_sarvam1_val_metrics.json"

def build_translation_prompt(en_text: str) -> str:
    return f"अंग्रेजी:\n{en_text}\n\nहिन्दी (देवनागरी लिपि में अनुवाद):\n"

def main():
    print(f"=== Translating Validation Subtopics with Fine-Tuned Sarvam-1 QLoRA ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print(f"Model loaded in {time.time() - t0_load:.2f}s | Active VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    # 3. Run Inference on All 35 Subtopics
    t0_infer = time.time()
    records = []

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
            "src_en": en_text,
            "gold_hi": gold_hi,
            "finetuned_hi": decoded
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(val_subtopics):
            print(f"[{idx+1}/{len(val_subtopics)}] Done (last length ratio: {hyp_words}/{gold_words} = {hyp_words/max(1,gold_words):.2f})")

    t_infer = time.time() - t0_infer
    print(f"Validation inference finished in {t_infer:.2f}s ({len(val_subtopics)/t_infer:.2f} subtopics/s).")

    # 4. Compute Metrics across Val
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

    metrics_summary = {
        "val_samples_count": len(records),
        "stem_samples_count": len(stem_hyp),
        "overall": {
            "bleu_flores200": round(bleu_flores.score, 2),
            "bleu_13a": round(bleu_13a.score, 2),
            "chrf_plus_plus": round(chrf_pp.score, 2),
            "bp": round(bleu_flores.bp, 3),
            "sys_len": bleu_flores.sys_len,
            "ref_len": bleu_flores.ref_len,
            "raw_bleu_flores200": round(raw_bleu_flores, 2),
            "precisions": [round(p, 2) for p in bleu_flores.precisions]
        },
        "stem": {
            "bleu_flores200": round(stem_bleu_flores.score, 2) if stem_bleu_flores else None,
            "chrf_plus_plus": round(stem_chrf.score, 2) if stem_chrf else None,
            "bp": round(stem_bleu_flores.bp, 3) if stem_bleu_flores else None,
            "raw_bleu_flores200": round(stem_raw_bleu, 2) if stem_bleu_flores else None,
            "precisions": [round(p, 2) for p in stem_bleu_flores.precisions] if stem_bleu_flores else None
        }
    }

    print("\n=== Validation Metrics Summary ===")
    print(f"Overall Raw BLEU: {round(raw_bleu_flores, 2)} | BP: {round(bleu_flores.bp, 3)} | Reported BLEU: {round(bleu_flores.score, 2)} | chrF++: {round(chrf_pp.score, 2)}")
    print(f"Precisions (1-4 gram): {[round(p, 2) for p in bleu_flores.precisions]}")
    if stem_bleu_flores:
        print(f"STEM Raw BLEU: {round(stem_raw_bleu, 2)} | BP: {round(stem_bleu_flores.bp, 3)} | Reported BLEU: {round(stem_bleu_flores.score, 2)} | chrF++: {round(stem_chrf.score, 2)}")
        print(f"STEM Precisions (1-4 gram): {[round(p, 2) for p in stem_bleu_flores.precisions]}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Full validation records saved to {OUTPUT_FILE}")

    with open(METRICS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Validation metrics saved to {METRICS_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
