"""Final Comparative Benchmark: Unimodal Text Baseline vs. Trained Multimodal Gated Architecture.

Evaluates on the 100 validation benchmark lectures and reports Strict (w=0), Relaxed (w=1),
and Window 2 (w=2) metrics with delta improvements.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from src.segmentation.model import DiscourseAwareSegmentationConfig, DiscourseAwareSegmentationModel
from src.segmentation.qwen_extractor import QwenSentenceRepresentationExtractor, DEFAULT_QWEN_MODEL
from src.segmentation.training_harness import non_maximum_suppression_1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(preds: list[int], gold: list[int], tolerance: int = 1) -> dict[str, float]:
    if not preds and not gold:
        return {"prec": 1.0, "rec": 1.0, "f1": 1.0}
    if not preds or not gold:
        return {"prec": 0.0, "rec": 0.0, "f1": 0.0}

    matched_gold = set()
    matched_pred = set()
    for p in preds:
        for g in gold:
            if abs(p - g) <= tolerance and g not in matched_gold:
                matched_gold.add(g)
                matched_pred.add(p)
                break

    tp = len(matched_pred)
    prec = tp / len(preds) if preds else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"prec": prec, "rec": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-checkpoint", type=str, default="checkpoints/segmentation_qwen_hq/best_model.pt")
    parser.add_argument("--multi-checkpoint", type=str, default="checkpoints/segmentation_multimodal_scale/best_model.pt")
    parser.add_argument("--model-name", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--text-cache", type=str, default="data/ytseg_val_100/text_cache_qwen3_embed.pt")
    parser.add_argument("--val-manifest", type=str, default="data/ytseg_val_100/manifest_100.json")
    parser.add_argument("--val-cache", type=str, default="data/ytseg_val_100/visual_cache.pt")
    parser.add_argument("--arrow-file", type=str, default="datasets/ytseg-dataset/validation/data-00000-of-00001.arrow")
    parser.add_argument("--output-json", type=str, default="data/ytseg_val_100/qwen3_embed_multimodal_benchmark.json")
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument("--nms-radius", type=int, default=6)
    parser.add_argument("--max-sentences", type=int, default=300)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.val_manifest) as f:
        val_manifest = json.load(f)

    if not Path(args.val_cache).exists():
        logger.error(f"Validation visual cache not found: {args.val_cache}")
        return

    val_cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    logger.info(f"Loaded validation visual cache with {len(val_cache)} entries.")

    with pa.memory_map(args.arrow_file, "r") as src:
        val_table = ipc.open_stream(src).read_all()

    # 1. Load Text Baseline Model
    logger.info(f"Loading Text Baseline Model: {args.text_checkpoint}...")
    ckpt_text = torch.load(args.text_checkpoint, map_location=device, weights_only=False)
    cfg_text = ckpt_text.get("model_config")
    if isinstance(cfg_text, dict):
        cfg_text = DiscourseAwareSegmentationConfig(**cfg_text)
    elif cfg_text is None:
        cfg_text = DiscourseAwareSegmentationConfig(hidden_size=1024, adapter_dim=1024)
    model_text = DiscourseAwareSegmentationModel(cfg_text).to(device).eval()
    model_text.load_state_dict(ckpt_text["model_state_dict"], strict=False)

    # 2. Load Multimodal Model
    logger.info(f"Loading Multimodal Model: {args.multi_checkpoint}...")
    multi_path = Path(args.multi_checkpoint)
    model_multi = None
    if multi_path.exists():
        ckpt_multi = torch.load(args.multi_checkpoint, map_location=device, weights_only=False)
        cfg_multi = ckpt_multi.get("model_config", cfg_text)
        if isinstance(cfg_multi, dict):
            cfg_multi = DiscourseAwareSegmentationConfig(**cfg_multi)
        model_multi = DiscourseAwareSegmentationModel(cfg_multi).to(device).eval()
        model_multi.load_state_dict(ckpt_multi["model_state_dict"], strict=False)
    else:
        logger.warning(f"Trained multimodal checkpoint not found at {args.multi_checkpoint}. Evaluating text baseline against visual gating.")

    # 3. Load Qwen sentence extractor
    logger.info(f"Loading Qwen extractor ({args.model_name})...")
    text_extractor = QwenSentenceRepresentationExtractor(args.model_name, load_in_4bit=True, device=device).eval()
    if "sentence_extractor_state_dict" in ckpt_text and "Qwen3.5" in args.model_name:
        try:
            text_extractor.load_state_dict(ckpt_text["sentence_extractor_state_dict"])
        except Exception:
            pass

    val_text_cache_path = Path(args.text_cache)
    val_text_cache: dict[str, torch.Tensor] = {}
    if val_text_cache_path.exists():
        try:
            val_text_cache = torch.load(val_text_cache_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded validation text cache with {len(val_text_cache)} entries from {val_text_cache_path}.")
        except Exception:
            pass

    results = []
    logger.info("=== Running Comparative Benchmark ===")
    with torch.no_grad():
        for idx, item in enumerate(val_manifest):
            vid = item["video_id"]
            if vid not in val_cache:
                continue

            doc_idx = item["doc_idx"]
            sents = val_table["text_ref"][doc_idx].as_py()
            if isinstance(sents, str):
                sents = [sents]

            vis_entry = val_cache[vid]
            n_sents = min(len(sents), len(vis_entry["visual_embeddings"]), args.max_sentences)
            sents = sents[:n_sents]

            gold_b = [b for b in item["gold_boundaries"] if b < n_sents]
            if not gold_b:
                continue

            # Text representations
            if vid in val_text_cache:
                reps = val_text_cache[vid][:n_sents].to(device)
            else:
                reps = text_extractor(sents)
                val_text_cache[vid] = reps.cpu()

            mask = torch.ones(1, n_sents, device=device)

            # 1. Text Baseline Inference
            out_t = model_text(sentence_reps=reps, sentence_mask=mask)
            p_t = non_maximum_suppression_1d(out_t.boundary_probabilities[0].cpu(), threshold=args.threshold, radius=args.nms_radius)
            p_t = [p for p in p_t if p > 0]

            m_t_s = compute_metrics(p_t, gold_b, tolerance=0)
            m_t_r = compute_metrics(p_t, gold_b, tolerance=1)
            m_t_w2 = compute_metrics(p_t, gold_b, tolerance=2)

            # 2. Multimodal Inference
            v_emb = vis_entry["visual_embeddings"][:n_sents].unsqueeze(0).to(device)
            v_shifts = vis_entry["visual_shifts"][:n_sents].unsqueeze(0).to(device)

            active_multi_model = model_multi or model_text
            out_m = active_multi_model(
                sentence_reps=reps,
                sentence_mask=mask,
                visual_embeddings=v_emb,
                visual_shifts=v_shifts,
            )
            p_m = non_maximum_suppression_1d(out_m.boundary_probabilities[0].cpu(), threshold=args.threshold, radius=args.nms_radius)
            p_m = [p for p in p_m if p > 0]

            m_m_s = compute_metrics(p_m, gold_b, tolerance=0)
            m_m_r = compute_metrics(p_m, gold_b, tolerance=1)
            m_m_w2 = compute_metrics(p_m, gold_b, tolerance=2)

            res = {
                "video_id": vid,
                "doc_idx": doc_idx,
                "n_sentences": n_sents,
                "gold_count": len(gold_b),
                "num_keyframes": vis_entry.get("num_keyframes", 0),
                "text_strict_f1": m_t_s["f1"],
                "multi_strict_f1": m_m_s["f1"],
                "text_relaxed_f1": m_t_r["f1"],
                "multi_relaxed_f1": m_m_r["f1"],
                "text_w2_f1": m_t_w2["f1"],
                "multi_w2_f1": m_m_w2["f1"],
                "text_precision": m_t_r["prec"],
                "multi_precision": m_m_r["prec"],
                "text_recall": m_t_r["rec"],
                "multi_recall": m_m_r["rec"],
            }
            results.append(res)

            if (idx + 1) % 15 == 0 or (idx + 1) == len(val_manifest):
                cur_df = pd.DataFrame(results)
                logger.info(
                    f"[{idx+1:03d}/{len(val_manifest)}] "
                    f"Text Rel F1: {cur_df['text_relaxed_f1'].mean()*100:.2f}% | "
                    f"Multi Rel F1: {cur_df['multi_relaxed_f1'].mean()*100:.2f}% | "
                    f"Delta: {(cur_df['multi_relaxed_f1'].mean() - cur_df['text_relaxed_f1'].mean())*100:+.2f}%"
                )

    df = pd.DataFrame(results)
    summary = {
        "num_videos": len(df),
        "text_strict_f1": float(df["text_strict_f1"].mean()),
        "multi_strict_f1": float(df["multi_strict_f1"].mean()),
        "strict_delta": float(df["multi_strict_f1"].mean() - df["text_strict_f1"].mean()),
        "text_relaxed_f1": float(df["text_relaxed_f1"].mean()),
        "multi_relaxed_f1": float(df["multi_relaxed_f1"].mean()),
        "relaxed_delta": float(df["multi_relaxed_f1"].mean() - df["text_relaxed_f1"].mean()),
        "text_w2_f1": float(df["text_w2_f1"].mean()),
        "multi_w2_f1": float(df["multi_w2_f1"].mean()),
        "w2_delta": float(df["multi_w2_f1"].mean() - df["text_w2_f1"].mean()),
        "text_precision": float(df["text_precision"].mean()),
        "multi_precision": float(df["multi_precision"].mean()),
        "text_recall": float(df["text_recall"].mean()),
        "multi_recall": float(df["multi_recall"].mean()),
    }

    print("\n" + "=" * 80)
    print("FINAL COMPARATIVE BENCHMARK: UNIMODAL TEXT VS. MULTIMODAL GATED ADAPTER")
    print("=" * 80)
    print(f"Total Validation Lectures Evaluated: {summary['num_videos']}")
    print("-" * 80)
    print(f"STRICT MATCH (w=0):")
    print(f"  Unimodal Text Baseline:  {summary['text_strict_f1']*100:.2f}%")
    print(f"  Trained Multimodal:      {summary['multi_strict_f1']*100:.2f}% (Delta: {summary['strict_delta']*100:+.2f}%)")
    print("-" * 80)
    print(f"STANDARD RELAXED MATCH (w=1):")
    print(f"  Unimodal Text Precision: {summary['text_precision']*100:.2f}%  -> Multimodal: {summary['multi_precision']*100:.2f}%")
    print(f"  Unimodal Text Recall:    {summary['text_recall']*100:.2f}%  -> Multimodal: {summary['multi_recall']*100:.2f}%")
    print(f"  Unimodal Text F1:        {summary['text_relaxed_f1']*100:.2f}%")
    print(f"  Trained Multimodal F1:   {summary['multi_relaxed_f1']*100:.2f}% (Delta: {summary['relaxed_delta']*100:+.2f}%)")
    print("-" * 80)
    print(f"DISCOURSE WINDOW 2 (w=2):")
    print(f"  Unimodal Text F1:        {summary['text_w2_f1']*100:.2f}%")
    print(f"  Trained Multimodal F1:   {summary['multi_w2_f1']*100:.2f}% (Delta: {summary['w2_delta']*100:+.2f}%)")
    print("=" * 80)

    if val_text_cache and not val_text_cache_path.exists():
        try:
            val_text_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(val_text_cache, val_text_cache_path)
            logger.info(f"Saved validation text cache to {val_text_cache_path}")
        except Exception as e:
            logger.warning(f"Could not save text cache: {e}")

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump({"summary": summary, "per_video": results}, f, indent=2)
    logger.info(f"Saved benchmark results to {out_p}")


if __name__ == "__main__":
    main()
