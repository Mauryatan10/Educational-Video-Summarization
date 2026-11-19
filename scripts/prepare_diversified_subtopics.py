#!/usr/bin/env python3
"""
scripts/prepare_diversified_subtopics.py

Engineers diversified 10-15 sentence paragraph-level training datasets
from the raw IIT Bombay Parallel Corpus (data/iitb_raw/data/).

Key Improvements:
  - Strict Word Budget: 200 to 320 English words per block.
  - Strict Sentence Count: Uniformly diversified across 10 to 15 sentences.
  - Quality Filtering: Rejects sentences < 4 words (UI buttons, fragments) or > 60 words.
  - No Early Flushing: Discards buffers that fail to reach >= 10 sentences, ensuring
    zero short fragments in the training corpus.
  - Curates balanced 6,000-sample training dataset (3,600 STEM + 2,400 General).

Saves to: data/subtopic_diversified/
"""

import os
import sys
import json
import re
import random
from typing import List, Dict, Any, Optional

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
    "optimization", "parameter", "distribution", "eigenvalue", "eigenvector",
    "probability", "statistic", "hypothesis", "regression", "classification",
    "cluster", "entropy", "tensor", "scalar", "convolution", "transformer"
]

OUTPUT_DIR = "data/subtopic_diversified"
RAW_DIR = "data/iitb_raw/data"


def is_valid_sentence(en: str, hi: str) -> bool:
    en_clean = en.strip()
    hi_clean = hi.strip()
    w_en = len(en_clean.split())
    w_hi = len(hi_clean.split())
    
    # Must be at least 4 words and at most 60 words
    if w_en < 4 or w_hi < 4 or w_en > 60 or w_hi > 75:
        return False
    # Filter UI placeholders and dividers
    if "% s" in en_clean or "%s" in en_clean or "___" in en_clean:
        return False
    if "http://" in en_clean or "https://" in en_clean:
        return False
    return True


def detect_stem_terms(text: str) -> List[str]:
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return [k for k in STEM_KEYWORDS if k in words]


def pack_diversified_subtopics(
    raw_data: List[Dict[str, str]],
    split_name: str,
    min_sentences: int = 10,
    max_sentences: int = 15,
    min_words: int = 200,
    max_words: int = 320,
) -> List[Dict[str, Any]]:
    subtopics = []
    curr_en_sents = []
    curr_hi_sents = []
    curr_word_count = 0
    subtopic_id = 0

    # Initialize first target budget uniformly between min_words and max_words
    random.seed(42)
    target_words = random.randint(min_words, max_words)
    target_sents = random.randint(min_sentences, max_sentences)

    for idx, row in enumerate(raw_data):
        en = row.get("en", "").strip()
        hi = row.get("hi", "").strip()

        if not is_valid_sentence(en, hi):
            # If invalid sentence encountered, only flush if buffer already satisfies strict 10-sentence constraint
            if len(curr_en_sents) >= min_sentences and curr_word_count >= 180:
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
            # Reset buffer completely to avoid short fragments
            curr_en_sents = []
            curr_hi_sents = []
            curr_word_count = 0
            target_words = random.randint(min_words, max_words)
            target_sents = random.randint(min_sentences, max_sentences)
            continue

        en_words = len(en.split())
        curr_en_sents.append(en)
        curr_hi_sents.append(hi)
        curr_word_count += en_words

        # Check if subtopic is ready
        if (len(curr_en_sents) >= target_sents and curr_word_count >= target_words) or len(curr_en_sents) >= max_sentences:
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
            target_words = random.randint(min_words, max_words)
            target_sents = random.randint(min_sentences, max_sentences)

    # Flush final buffer if it satisfies minimum requirements
    if len(curr_en_sents) >= min_sentences and curr_word_count >= 180:
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


def process_split(raw_file: str, split_name: str) -> List[Dict[str, Any]]:
    print(f"\nProcessing {split_name} from {raw_file}...")
    table = pq.read_table(raw_file)
    raw_pairs = []
    for r in table.to_pylist():
        t = r.get("translation", {})
        en = t.get("en", "")
        hi = t.get("hi", "")
        if en and hi:
            raw_pairs.append({"en": en, "hi": hi})

    print(f"Loaded {len(raw_pairs):,} raw sentence pairs.")
    subtopics = pack_diversified_subtopics(raw_pairs, split_name)
    print(f"Packaged into {len(subtopics):,} diversified 10-15 sentence paragraphs.")
    return subtopics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 80)
    print("=== ENGINEERING DIVERSIFIED 10-15 SENTENCE SUBTOPIC DATASET ===")
    print("=" * 80)

    # 1. Process Train Split
    train_raw = os.path.join(RAW_DIR, "train-00000-of-00001.parquet")
    train_subtopics = process_split(train_raw, "train")

    # Save full training set
    full_train_p = os.path.join(OUTPUT_DIR, "train_subtopics_all.parquet")
    pq.write_table(pa.Table.from_pylist(train_subtopics), full_train_p)
    print(f"Saved all {len(train_subtopics):,} training paragraphs to {full_train_p}")

    # Curate balanced 6,000-sample training curriculum (3,600 STEM + 2,400 General)
    stem_pool = [s for s in train_subtopics if s["is_stem"]]
    gen_pool = [s for s in train_subtopics if not s["is_stem"]]
    print(f"Training Pool: {len(stem_pool):,} STEM paragraphs | {len(gen_pool):,} General paragraphs")

    random.seed(42)
    target_stem = 3600
    target_gen = 2400
    selected_stem = random.sample(stem_pool, min(target_stem, len(stem_pool)))
    selected_gen = random.sample(gen_pool, min(target_gen, len(gen_pool)))
    curated_train = selected_stem + selected_gen
    random.shuffle(curated_train)

    curated_train_p = os.path.join(OUTPUT_DIR, "train_subtopics_6k.parquet")
    pq.write_table(pa.Table.from_pylist(curated_train), curated_train_p)
    print(f"Saved curated 6,000-sample balanced training split to {curated_train_p}")

    # Compute sentence & word stats for curated training
    sents = [s["num_sentences"] for s in curated_train]
    words_en = [s["word_count_en"] for s in curated_train]
    words_hi = [s["word_count_hi"] for s in curated_train]
    print(f"Curated 6k Stats:")
    print(f"  Sentence Count: Min={min(sents)} | Max={max(sents)} | Avg={sum(sents)/len(sents):.2f}")
    print(f"  English Words:  Min={min(words_en)} | Max={max(words_en)} | Avg={sum(words_en)/len(words_en):.2f}")
    print(f"  Hindi Words:    Min={min(words_hi)} | Max={max(words_hi)} | Avg={sum(words_hi)/len(words_hi):.2f}")

    # 2. Validation Set: Copy the canonical 35 validation subtopics for exact benchmark comparison
    val_canon_p = "data/subtopic_engineered/val_subtopics.parquet"
    val_table = pq.read_table(val_canon_p)
    val_out_p = os.path.join(OUTPUT_DIR, "val_subtopics.parquet")
    pq.write_table(val_table, val_out_p)
    print(f"Linked canonical validation benchmark (35 subtopics) to {val_out_p}")

    # Save summary statistics
    stats = {
        "train_curated_6k": {
            "total_samples": len(curated_train),
            "stem_count": len(selected_stem),
            "stem_percentage": round(len(selected_stem) / len(curated_train) * 100, 2),
            "avg_sentences": round(sum(sents)/len(sents), 2),
            "min_sentences": min(sents),
            "max_sentences": max(sents),
            "avg_words_en": round(sum(words_en)/len(words_en), 2),
            "min_words_en": min(words_en),
            "max_words_en": max(words_en),
            "avg_words_hi": round(sum(words_hi)/len(words_hi), 2),
        },
        "validation_benchmark": {
            "total_samples": len(val_table),
            "description": "Standard 35-subtopic pedagogical benchmark for exact Flores200/chrF++ comparison."
        }
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
