#!/usr/bin/env python3
"""
prepare_subtopic_dataset.py

Engineers subtopic / paragraph-level training, validation, and test datasets
from the raw IIT Bombay Parallel Corpus (cfilt/iitb-english-hindi).

Instead of isolated single sentences, this script aggregates consecutive sequential
sentences into coherent subtopic paragraphs (target ~4 to 8 sentences, 100 to 250 words)
representing lecture subtopic blocks.

Saves:
  - data/subtopic_engineered/train_subtopics.parquet
  - data/subtopic_engineered/test_subtopics.parquet
  - data/subtopic_engineered/val_subtopics.parquet
  - data/subtopic_engineered/dataset_statistics.json
"""

import os
import sys
import json
import re
from typing import List, Dict, Any

# Strip local dir from sys.path to avoid shadowing third-party 'datasets'
sys.path = [p for p in sys.path if p not in ('', '.', os.path.dirname(os.path.abspath(__file__)))]

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

# Technical STEM keywords for domain tagging
STEM_KEYWORDS = [
    "algorithm", "matrix", "vector", "neural", "network", "function", "variable",
    "constant", "linear", "gradient", "model", "computation", "compute", "database",
    "query", "binary", "byte", "server", "protocol", "circuit", "voltage", "current",
    "frequency", "signal", "energy", "force", "velocity", "acceleration", "quantum",
    "equation", "derivative", "integral", "polynomial", "dimension", "entropy",
    "software", "hardware", "interface", "compiler", "memory", "cache", "processor",
    "architecture", "instruction", "bandwidth", "packet", "router", "ethernet",
    "device", "system", "transmission", "storage", "sensor", "actuator", "feedback",
    "optimization", "parameter", "distribution", "eigenvalue", "eigenvector"
]

OUTPUT_DIR = "dat   a/subtopic_engineered"
RAW_DIR = "data/iitb_raw/data"


def is_valid_sentence(en: str, hi: str) -> bool:
    en_clean = en.strip()
    hi_clean = hi.strip()
    # Skip empty or trivial 1-word UI fragments
    if len(en_clean.split()) < 3 or len(hi_clean.split()) < 3:
        return False
    # Skip UI placeholders like %s or __
    if "% s" in en_clean or "%s" in en_clean:
        return False
    return True


def detect_stem_terms(text: str) -> List[str]:
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return [k for k in STEM_KEYWORDS if k in words]


def pack_sentences_into_subtopics(
    raw_data: List[Dict[str, str]],
    split_name: str,
    target_sentences: int = 15,
    max_words: int = 700,
    min_sentences: int = 5
) -> List[Dict[str, Any]]:
    subtopics = []
    curr_en_sents = []
    curr_hi_sents = []
    curr_word_count = 0
    subtopic_id = 0

    for idx, row in enumerate(raw_data):
        en = row.get("en", "").strip()
        hi = row.get("hi", "").strip()

        if not is_valid_sentence(en, hi):
            # If we hit an invalid row (like a UI divider or heading), flush current buffer if substantial
            if len(curr_en_sents) >= min_sentences:
                en_block = " ".join(curr_en_sents)
                hi_block = " ".join(curr_hi_sents)
                stem_terms = detect_stem_terms(en_block)
                subtopics.append({
                    "subtopic_id": f"{split_name}_{subtopic_id:06d}",
                    "src_en": en_block,
                    "tgt_hi": hi_block,
                    "num_sentences": len(curr_en_sents),
                    "word_count_en": len(en_block.split()),
                    "word_count_hi": len(hi_block.split()),
                    "is_stem": len(stem_terms) > 0,
                    "stem_terms": stem_terms
                })
                subtopic_id += 1
            curr_en_sents = []
            curr_hi_sents = []
            curr_word_count = 0
            continue

        en_words = len(en.split())
        curr_en_sents.append(en)
        curr_hi_sents.append(hi)
        curr_word_count += en_words

        # Check if subtopic block is ready (target sentence count reached or word threshold met)
        if len(curr_en_sents) >= target_sentences or curr_word_count >= max_words:
            en_block = " ".join(curr_en_sents)
            hi_block = " ".join(curr_hi_sents)
            stem_terms = detect_stem_terms(en_block)
            subtopics.append({
                "subtopic_id": f"{split_name}_{subtopic_id:06d}",
                "src_en": en_block,
                "tgt_hi": hi_block,
                "num_sentences": len(curr_en_sents),
                "word_count_en": len(en_block.split()),
                "word_count_hi": len(hi_block.split()),
                "is_stem": len(stem_terms) > 0,
                "stem_terms": stem_terms
            })
            subtopic_id += 1
            curr_en_sents = []
            curr_hi_sents = []
            curr_word_count = 0

    # Flush any remaining valid buffer
    if len(curr_en_sents) >= min_sentences:
        en_block = " ".join(curr_en_sents)
        hi_block = " ".join(curr_hi_sents)
        stem_terms = detect_stem_terms(en_block)
        subtopics.append({
            "subtopic_id": f"{split_name}_{subtopic_id:06d}",
            "src_en": en_block,
            "tgt_hi": hi_block,
            "num_sentences": len(curr_en_sents),
            "word_count_en": len(en_block.split()),
            "word_count_hi": len(hi_block.split()),
            "is_stem": len(stem_terms) > 0,
            "stem_terms": stem_terms
        })

    return subtopics


def process_and_save_split(raw_file: str, split_name: str, tokenizer=None) -> Dict[str, Any]:
    print(f"\nProcessing {split_name} from {raw_file}...")
    table = pq.read_table(raw_file)
    raw_pairs = []
    for r in table.to_pylist():
        t = r.get("translation", {})
        en = t.get("en", "")
        hi = t.get("hi", "")
        if en and hi:
            raw_pairs.append({"en": en, "hi": hi})

    print(f"Loaded {len(raw_pairs)} raw sentence pairs.")
    subtopics = pack_sentences_into_subtopics(raw_pairs, split_name)
    print(f"Packaged into {len(subtopics)} coherent 15-sentence subtopic paragraphs.")

    # Save to Parquet
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}_subtopics.parquet")
    pa_table = pa.Table.from_pylist(subtopics)
    pq.write_table(pa_table, out_path)
    print(f"Saved {split_name} subtopics to {out_path}")

    # Compute word and sentence statistics
    stem_count = sum(1 for s in subtopics if s["is_stem"])
    avg_words_en = sum(s["word_count_en"] for s in subtopics) / max(1, len(subtopics))
    avg_words_hi = sum(s["word_count_hi"] for s in subtopics) / max(1, len(subtopics))
    avg_sents = sum(s["num_sentences"] for s in subtopics) / max(1, len(subtopics))

    # Compute Sarvam-1 token statistics
    token_stats = {}
    if tokenizer is not None and len(subtopics) > 0:
        sample_size = min(3000, len(subtopics))
        sample_subs = subtopics[:sample_size]
        src_tokens = [len(tokenizer.encode(s["src_en"])) for s in sample_subs]
        tgt_tokens = [len(tokenizer.encode(s["tgt_hi"])) for s in sample_subs]
        token_stats = {
            "tokenized_sample_count": sample_size,
            "avg_tokens_src_en": round(sum(src_tokens) / len(src_tokens), 1),
            "avg_tokens_tgt_hi": round(sum(tgt_tokens) / len(tgt_tokens), 1),
            "max_tokens_src_en": max(src_tokens),
            "max_tokens_tgt_hi": max(tgt_tokens),
            "avg_combined_tokens": round((sum(src_tokens) + sum(tgt_tokens)) / len(src_tokens), 1)
        }
        print(f"Token Stats (Sarvam-1): Avg EN={token_stats['avg_tokens_src_en']} tok | Avg HI={token_stats['avg_tokens_tgt_hi']} tok | Avg Combined={token_stats['avg_combined_tokens']} tok")

    return {
        "split": split_name,
        "total_raw_sentences": len(raw_pairs),
        "total_subtopics": len(subtopics),
        "stem_subtopics": stem_count,
        "stem_percentage": round(stem_count / max(1, len(subtopics)) * 100, 2),
        "avg_sentences_per_subtopic": round(avg_sents, 2),
        "avg_words_en": round(avg_words_en, 2),
        "avg_words_hi": round(avg_words_hi, 2),
        **token_stats
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Engineering 15-Sentence Subtopic Paragraph Dataset ===")

    print("Loading Sarvam-1 tokenizer to benchmark context window token usage...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")
    except Exception as e:
        print(f"Warning: Could not load tokenizer ({e}). Proceeding without token counts.")
        tokenizer = None

    stats = {}
    # 1. Test Split
    test_path = os.path.join(RAW_DIR, "test-00000-of-00001.parquet")
    stats["test"] = process_and_save_split(test_path, "test", tokenizer=tokenizer)

    # 2. Validation Split
    val_path = os.path.join(RAW_DIR, "validation-00000-of-00001.parquet")
    stats["validation"] = process_and_save_split(val_path, "val", tokenizer=tokenizer)

    # 3. Training Split
    train_path = os.path.join(RAW_DIR, "train-00000-of-00001.parquet")
    stats["train"] = process_and_save_split(train_path, "train", tokenizer=tokenizer)

    stats_file = os.path.join(OUTPUT_DIR, "dataset_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Complete] Summary dataset statistics written to {stats_file}")


if __name__ == "__main__":
    main()
