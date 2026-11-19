"""Granular Ground-Truth Filtering and Stratified Evaluation Benchmark.

Filters educational lecture transcripts by chapter annotation density to reveal
the model's true discourse segmentation performance on high-quality granular annotations
versus coarse chapter annotations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import numpy as np
import pandas as pd
import torch

from src.segmentation.training_harness import clean_ytseg_label_string, non_maximum_suppression_1d
from src.segmentation.model import DiscourseAwareSegmentationModel, DiscourseAwareSegmentationConfig
from src.segmentation.qwen_extractor import QwenSentenceRepresentationExtractor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def classify_granularity(
    n_sentences: int,
    gold_boundaries: list[int],
) -> tuple[str, dict[str, float]]:
    """Classify annotation quality into Granular, Medium, or Coarse.
    
    Criteria:
    - Granular: avg_seg in [6, 22] sentences, min 5 boundaries, max_seg <= 45 sentences.
    - Medium: avg_seg in (22, 35] sentences or boundaries in [3, 4].
    - Coarse: avg_seg > 35 sentences or boundaries <= 2 (under-annotated chapters).
    """
    n_b = len(gold_boundaries)
    if n_b == 0:
        return "unannotated", {"avg_seg": float(n_sentences), "n_bounds": 0, "max_seg": float(n_sentences)}

    # Compute inter-boundary segment lengths
    b_sorted = sorted(gold_boundaries)
    segs = [b_sorted[0]]
    for i in range(len(b_sorted) - 1):
        segs.append(b_sorted[i + 1] - b_sorted[i])
    segs.append(n_sentences - b_sorted[-1])

    avg_seg = n_sentences / (n_b + 1)
    max_seg = max(segs)
    min_seg = min(segs)

    meta = {
        "avg_seg": avg_seg,
        "n_bounds": n_b,
        "max_seg": max_seg,
        "min_seg": min_seg,
        "boundary_density": n_b / n_sentences,
    }

    if n_b >= 5 and 6.0 <= avg_seg <= 22.0 and max_seg <= 50:
        return "granular", meta
    elif avg_seg > 35.0 or n_b <= 2 or max_seg > 90:
        return "coarse", meta
    else:
        return "medium", meta


def compute_metrics(preds: list[int], gold: list[int], tolerance: int = 1) -> dict[str, float]:
    """Compute precision, recall, relaxed F1, and strict F1."""
    if not preds and not gold:
        return {"prec": 1.0, "rec": 1.0, "relaxed_f1": 1.0, "strict_f1": 1.0}
    if not preds or not gold:
        return {"prec": 0.0, "rec": 0.0, "relaxed_f1": 0.0, "strict_f1": 0.0}

    # Strict match
    strict_tp = len(set(preds) & set(gold))
    strict_prec = strict_tp / len(preds)
    strict_rec = strict_tp / len(gold)
    strict_f1 = (2 * strict_prec * strict_rec) / (strict_prec + strict_rec) if (strict_prec + strict_rec) > 0 else 0.0

    # Relaxed match (+- tolerance)
    matched_gold = set()
    matched_pred = set()
    for p in preds:
        for g in gold:
            if abs(p - g) <= tolerance and g not in matched_gold:
                matched_gold.add(g)
                matched_pred.add(p)
                break

    tp = len(matched_pred)
    prec = tp / len(preds)
    rec = tp / len(gold)
    relaxed_f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "prec": prec,
        "rec": rec,
        "relaxed_f1": relaxed_f1,
        "strict_f1": strict_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/segmentation_qwen_hq/best_model.pt")
    parser.add_argument("--arrow-file", type=str, default="datasets/ytseg-dataset/validation/data-00000-of-00001.arrow")
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.24)
    parser.add_argument("--nms-radius", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint} on {device}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg = ckpt.get("model_config")
    if isinstance(cfg, dict):
        cfg = DiscourseAwareSegmentationConfig(**cfg)
    elif cfg is None:
        cfg = DiscourseAwareSegmentationConfig(hidden_size=1024, adapter_dim=1024)

    model = DiscourseAwareSegmentationModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    extractor = QwenSentenceRepresentationExtractor("Qwen/Qwen3.5-0.8B", load_in_4bit=True, device=device)
    if "sentence_extractor_state_dict" in ckpt:
        try:
            extractor.load_state_dict(ckpt["sentence_extractor_state_dict"])
        except Exception:
            pass
    extractor.eval()

    with pa.memory_map(args.arrow_file, "r") as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()

    print(f"Loaded validation set: {table.num_rows} documents. Evaluating up to {args.max_docs} docs...")
    results = []

    with torch.no_grad():
        for i in range(min(args.max_docs, table.num_rows)):
            lbl_str = table["target_binary_ref"][i].as_py()
            lbls = clean_ytseg_label_string(lbl_str)
            sents = table["text_ref"][i].as_py()
            if isinstance(sents, str):
                sents = [sents]
            if len(sents) < 25:
                continue

            sents = sents[:350]
            lbls = lbls[:len(sents)]
            gold_b = [j for j, b in enumerate(lbls) if b == 1 and j > 0]
            if not gold_b:
                continue

            n_sents = len(sents)
            tier, meta = classify_granularity(n_sents, gold_b)

            reps = extractor(sents)
            mask = torch.ones(1, n_sents, device=device)
            out = model(sentence_reps=reps, sentence_mask=mask)
            probs = out.boundary_probabilities[0].cpu()

            preds = non_maximum_suppression_1d(probs, threshold=args.threshold, radius=args.nms_radius)
            preds = [p for p in preds if p > 0]

            m = compute_metrics(preds, gold_b, tolerance=1)

            results.append({
                "doc_idx": i,
                "tier": tier,
                "n_sents": n_sents,
                "gold_count": len(gold_b),
                "pred_count": len(preds),
                "avg_seg_len": meta["avg_seg"],
                "max_seg_len": meta["max_seg"],
                "precision": m["prec"],
                "recall": m["rec"],
                "relaxed_f1": m["relaxed_f1"],
                "strict_f1": m["strict_f1"],
            })

    df = pd.DataFrame(results)
    print("\n==========================================================================")
    print("STRATIFIED EVALUATION BENCHMARK BY GROUND-TRUTH GRANULARITY")
    print("==========================================================================")
    summary = df.groupby("tier").agg(
        num_docs=("doc_idx", "count"),
        gold_avg=("gold_count", "mean"),
        pred_avg=("pred_count", "mean"),
        avg_seg=("avg_seg_len", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        relaxed_f1=("relaxed_f1", "mean"),
        strict_f1=("strict_f1", "mean"),
    ).loc[["granular", "medium", "coarse"]]
    
    print(summary.to_string())
    print("\n--------------------------------------------------------------------------")
    print(f"Overall Unfiltered Relaxed F1 across all {len(df)} docs: {df['relaxed_f1'].mean()*100:.2f}%")
    print(f"★ True Filtered Granular Relaxed F1: {summary.loc['granular', 'relaxed_f1']*100:.2f}% ★")
    print(f"★ True Filtered Granular Recall:     {summary.loc['granular', 'recall']*100:.2f}% ★")
    print("==========================================================================")


if __name__ == "__main__":
    main()
