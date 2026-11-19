#!/usr/bin/env python3
"""
scripts/train_qwen3_embed_3stage_multimodal.py

Full 3-Stage Multimodal (Text + Vision) Discourse Segmentation Training Harness
using Qwen/Qwen3-Embedding-0.6B as the sentence backbone.

Three-Stage Schedule on Scaled Tier-1 Dataset (592 Educational Lectures):
  - Stage 1 (Epochs 1-4): Frozen backbone, train DBCA + Multimodal Gating from scratch (LR = 2e-4).
  - Stage 2 (Epochs 5-7): Differential DBCA Layer 2 + Cross-Modal Refinement (LR = 2e-5).
  - Stage 3 (Epochs 8-9): End-to-End LoRA on Qwen3-Embedding + DBCA + Gating (LoRA LR = 5e-5, DBCA LR = 1e-4).

Evaluates on 94 Tier-1 validation benchmark lectures after each epoch.
Saves checkpoints and full benchmark report to checkpoints/segmentation_qwen3_embed_multimodal/.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from src.segmentation.losses import SoftMarginFocalLoss
from src.segmentation.model import DiscourseAwareSegmentationConfig, DiscourseAwareSegmentationModel
from src.segmentation.qwen_extractor import Qwen3EmbeddingSentenceRepresentationExtractor
from src.segmentation.training_harness import non_maximum_suppression_1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_qwen3_embed_3stage.log", mode="w"),
    ],
)
logger = logging.getLogger("TrainQwen3Embed3Stage")


def compute_metrics(preds: list[int], gold: list[int], tolerance: int = 1) -> dict[str, float]:
    """Compute precision, recall, and F1 for a given tolerance window."""
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


def evaluate(
    model: nn.Module,
    val_manifest: list[dict],
    val_cache: dict[str, dict],
    val_text_cache: dict[str, torch.Tensor],
    device: str,
    threshold: float = 0.20,
    nms_radius: int = 6,
    max_sentences: int = 300,
) -> dict[str, float]:
    """Evaluate model on the 94 validation benchmark lectures."""
    model.eval()
    results = []

    with torch.no_grad():
        for item in val_manifest:
            vid = item["video_id"]
            if vid not in val_cache or vid not in val_text_cache:
                continue

            vis_entry = val_cache[vid]
            reps = val_text_cache[vid].to(device)
            n_sents = min(reps.shape[1], len(vis_entry["visual_embeddings"]), max_sentences)
            reps = reps[:, :n_sents]

            gold_b = [b for b in item["gold_boundaries"] if b < n_sents]
            if not gold_b:
                continue

            mask = torch.ones(1, n_sents, device=device)
            v_emb = vis_entry["visual_embeddings"][:n_sents].unsqueeze(0).to(device)
            v_shifts = vis_entry["visual_shifts"][:n_sents].unsqueeze(0).to(device)

            out = model(
                sentence_reps=reps,
                sentence_mask=mask,
                visual_embeddings=v_emb,
                visual_shifts=v_shifts,
            )
            probs = out.boundary_probabilities[0].cpu()
            preds = non_maximum_suppression_1d(probs, threshold=threshold, radius=nms_radius)
            preds = [p for p in preds if p > 0]

            m_s = compute_metrics(preds, gold_b, tolerance=0)
            m_r = compute_metrics(preds, gold_b, tolerance=1)
            m_w2 = compute_metrics(preds, gold_b, tolerance=2)

            results.append({
                "strict_f1": m_s["f1"],
                "relaxed_f1": m_r["f1"],
                "w2_f1": m_w2["f1"],
                "prec": m_r["prec"],
                "rec": m_r["rec"],
            })

    if not results:
        return {"strict_f1": 0.0, "relaxed_f1": 0.0, "w2_f1": 0.0, "prec": 0.0, "rec": 0.0}

    return {
        "strict_f1": float(np.mean([r["strict_f1"] for r in results])),
        "relaxed_f1": float(np.mean([r["relaxed_f1"] for r in results])),
        "w2_f1": float(np.mean([r["w2_f1"] for r in results])),
        "prec": float(np.mean([r["prec"] for r in results])),
        "rec": float(np.mean([r["rec"] for r in results])),
        "evaluated_docs": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="checkpoints/segmentation_qwen3_embed_multimodal")
    parser.add_argument("--train-manifest", type=str, default="data/ytseg_train_scale/manifest_1600.json")
    parser.add_argument("--train-visual-cache", type=str, default="data/ytseg_train_scale/visual_cache.pt")
    parser.add_argument("--train-text-cache", type=str, default="data/ytseg_train_scale/text_cache_qwen3_embed.pt")
    parser.add_argument("--fallback-text-cache", type=str, default="data/ytseg_train_400/text_cache_qwen3_embed.pt")
    parser.add_argument("--val-manifest", type=str, default="data/ytseg_val_100/manifest_100.json")
    parser.add_argument("--val-visual-cache", type=str, default="data/ytseg_val_100/visual_cache.pt")
    parser.add_argument("--val-text-cache", type=str, default="data/ytseg_val_100/text_cache_qwen3_embed.pt")
    parser.add_argument("--stage1-epochs", type=int, default=4)
    parser.add_argument("--stage2-epochs", type=int, default=3)
    parser.add_argument("--stage3-epochs", type=int, default=2)
    parser.add_argument("--stage1-lr", type=float, default=2e-4)
    parser.add_argument("--stage2-lr", type=float, default=2e-5)
    parser.add_argument("--stage3-lora-lr", type=float, default=5e-5)
    parser.add_argument("--stage3-dbca-lr", type=float, default=1e-4)
    parser.add_argument("--max-sentences", type=int, default=250)
    parser.add_argument("--val-threshold", type=float, default=0.20)
    parser.add_argument("--chunk-size", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    logger.info("=" * 80)
    logger.info("STARTING 3-STAGE MULTIMODAL TRAINING: QWEN3-EMBEDDING-0.6B + DBCA")
    logger.info("=" * 80)
    logger.info(f"Target Output: {out_dir}")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Schedule: Stage 1 ({args.stage1_epochs} eps) -> Stage 2 ({args.stage2_epochs} eps) -> Stage 3 ({args.stage3_epochs} eps) = {total_epochs} Total Epochs")

    # 1. Initialize Fresh DiscourseAwareSegmentationModel with DBCA (H=1024)
    cfg = DiscourseAwareSegmentationConfig(
        hidden_size=1024,
        adapter_dim=1024,
        num_heads=8,
        num_layers=2,
        vision_dim=1024,
        vision_fusion_alpha=0.35,
        use_vision=True,
    )
    model = DiscourseAwareSegmentationModel(cfg).to(device)
    logger.info(f"Initialized Model. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Initialize Qwen3-Embedding Extractor
    logger.info("Loading Qwen3-Embedding-0.6B extractor in 4-bit NF4...")
    text_extractor = Qwen3EmbeddingSentenceRepresentationExtractor(
        chunk_size=args.chunk_size,
        load_in_4bit=True,
        device=device
    ).eval()

    # 3. Load Arrow Shards
    train_shards = sorted(glob.glob("datasets/ytseg-dataset/train/*.arrow"))
    train_tables = []
    for tf in train_shards:
        with pa.memory_map(tf, "r") as src:
            train_tables.append(ipc.open_stream(src).read_all())

    val_file = "datasets/ytseg-dataset/validation/data-00000-of-00001.arrow"
    with pa.memory_map(val_file, "r") as src:
        val_table = ipc.open_stream(src).read_all()

    # 4. Load Manifests & Visual Caches
    with open(args.train_manifest) as f:
        train_manifest = json.load(f)
    with open(args.val_manifest) as f:
        val_manifest = json.load(f)

    train_visual_cache = torch.load(args.train_visual_cache, map_location="cpu", weights_only=False)
    val_visual_cache = torch.load(args.val_visual_cache, map_location="cpu", weights_only=False)

    train_items = [it for it in train_manifest if it["video_id"] in train_visual_cache]
    logger.info(f"Loaded Scaled Dataset: {len(train_items)} eligible training items | {len(val_visual_cache)} validation items.")

    # 5. Load or Precompute Complete Text Cache (592 items)
    train_text_cache_p = Path(args.train_text_cache)
    train_text_cache: dict[str, torch.Tensor] = {}
    if train_text_cache_p.exists():
        train_text_cache = torch.load(train_text_cache_p, map_location="cpu", weights_only=False)
        logger.info(f"Loaded existing scaled train text cache with {len(train_text_cache)} items.")
    elif Path(args.fallback_text_cache).exists():
        train_text_cache = torch.load(args.fallback_text_cache, map_location="cpu", weights_only=False)
        logger.info(f"Loaded base train text cache with {len(train_text_cache)} items from fallback.")

    missing_items = [it for it in train_items if it["video_id"] not in train_text_cache]
    if missing_items:
        logger.info(f"Extracting Qwen3-Embedding representations for {len(missing_items)} remaining scaled training lectures...")
        t0_extract = time.time()
        for idx, item in enumerate(missing_items):
            vid = item["video_id"]
            s_idx = item["shard_idx"]
            r_idx = item["row_idx"]
            table = train_tables[s_idx]
            sents = table["text_ref"][r_idx].as_py()
            if isinstance(sents, str):
                sents = [sents]
            vis_entry = train_visual_cache[vid]
            n_sents = min(len(sents), len(vis_entry["visual_embeddings"]), args.max_sentences)
            sents = sents[:n_sents]

            with torch.no_grad():
                reps = text_extractor(sents)
                train_text_cache[vid] = reps.cpu()

            if (idx + 1) % 40 == 0 or (idx + 1) == len(missing_items):
                logger.info(f"Extracted {idx+1}/{len(missing_items)} ({time.time()-t0_extract:.1f}s)")

        torch.save(train_text_cache, train_text_cache_p)
        logger.info(f"Saved complete 592-lecture training text cache to {train_text_cache_p}")

    val_text_cache = torch.load(args.val_text_cache, map_location="cpu", weights_only=False)
    logger.info(f"Loaded validation text cache with {len(val_text_cache)} items.")

    loss_fn = SoftMarginFocalLoss(sigma=0.5, gamma=2.0, alpha=0.25)
    best_relaxed_f1 = -1.0
    history = []

    # =========================================================================
    # STAGE 1: Frozen Backbone, Train DBCA + Multimodal Gating from Scratch
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"=== STAGE 1: DBCA + GATING WARM-UP (Epochs 1 to {args.stage1_epochs}) ===")
    logger.info("=" * 80)

    # All DBCA layers, Transition Head, and Multimodal Gated Fusion are trainable
    for param in model.parameters():
        param.requires_grad = True

    opt_stage1 = AdamW(model.parameters(), lr=args.stage1_lr, weight_decay=0.01)
    sched_stage1 = CosineAnnealingLR(opt_stage1, T_max=args.stage1_epochs * len(train_items), eta_min=1e-6)

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        total_loss = 0.0
        t_epoch = time.time()
        np.random.shuffle(train_items)

        for step, item in enumerate(train_items):
            vid = item["video_id"]
            vis_entry = train_visual_cache[vid]
            gold_b = item["gold_boundaries"]
            text_reps = train_text_cache[vid].to(device)
            n_sents = min(text_reps.shape[1], len(vis_entry["visual_embeddings"]), args.max_sentences)
            text_reps = text_reps[:, :n_sents]
            gold_b = [b for b in gold_b if b < n_sents]
            if not gold_b:
                continue

            v_emb = vis_entry["visual_embeddings"][:n_sents].unsqueeze(0).to(device)
            v_shifts = vis_entry["visual_shifts"][:n_sents].unsqueeze(0).to(device)

            mask = torch.ones(1, n_sents, device=device)
            mask[:, 0] = 0.0
            gold_tensor = torch.tensor(gold_b, device=device)

            opt_stage1.zero_grad()
            out = model(
                sentence_reps=text_reps,
                sentence_mask=mask,
                visual_embeddings=v_emb,
                visual_shifts=v_shifts,
            )
            loss = loss_fn(out.boundary_logits, gold_tensor, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_stage1.step()
            sched_stage1.step()

            total_loss += loss.item()

        t_epoch_dur = time.time() - t_epoch
        avg_loss = total_loss / max(1, len(train_items))
        val_m = evaluate(model, val_manifest, val_visual_cache, val_text_cache, device, threshold=args.val_threshold, max_sentences=args.max_sentences)

        logger.info(
            f"[Stage 1 | Epoch {epoch}/{args.stage1_epochs}] ({t_epoch_dur:.1f}s) - Loss: {avg_loss:.4f} | "
            f"Strict F1: {val_m['strict_f1']*100:.2f}% | Relaxed F1: {val_m['relaxed_f1']*100:.2f}% | "
            f"W2 F1: {val_m['w2_f1']*100:.2f}% | Prec: {val_m['prec']*100:.2f}% | Rec: {val_m['rec']*100:.2f}%"
        )
        history.append({"stage": 1, "epoch": epoch, "loss": avg_loss, "val_metrics": val_m, "dur": t_epoch_dur})

        if val_m["relaxed_f1"] > best_relaxed_f1:
            best_relaxed_f1 = val_m["relaxed_f1"]
            torch.save({"epoch": epoch, "stage": 1, "model_state_dict": model.state_dict(), "model_config": cfg.__dict__, "best_relaxed_f1": best_relaxed_f1, "val_metrics": val_m}, out_dir / "best_model.pt")
            torch.save(model.state_dict(), out_dir / "stage1_best.pt")
            logger.info(f"--> [Saved Best Stage 1] Relaxed F1: {best_relaxed_f1*100:.2f}%")

    # =========================================================================
    # STAGE 2: Differential Refinement (DBCA Layer 2 + Final Norm + Gating)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"=== STAGE 2: DIFFERENTIAL CROSS-MODAL REFINEMENT (Epochs 1 to {args.stage2_epochs}) ===")
    logger.info("=" * 80)

    # Freeze DBCA Layer 1, keep Layer 2, final_norm, and trimodal_fusion trainable
    for name, param in model.named_parameters():
        if "dbca.layers.0." in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    stage2_trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Stage 2 Configured. Trainable params: {sum(p.numel() for p in stage2_trainable):,}")

    opt_stage2 = AdamW(stage2_trainable, lr=args.stage2_lr, weight_decay=0.01)
    sched_stage2 = CosineAnnealingLR(opt_stage2, T_max=args.stage2_epochs * len(train_items), eta_min=1e-6)

    for epoch in range(1, args.stage2_epochs + 1):
        global_ep = args.stage1_epochs + epoch
        model.train()
        total_loss = 0.0
        t_epoch = time.time()
        np.random.shuffle(train_items)

        for step, item in enumerate(train_items):
            vid = item["video_id"]
            vis_entry = train_visual_cache[vid]
            gold_b = item["gold_boundaries"]
            text_reps = train_text_cache[vid].to(device)
            n_sents = min(text_reps.shape[1], len(vis_entry["visual_embeddings"]), args.max_sentences)
            text_reps = text_reps[:, :n_sents]
            gold_b = [b for b in gold_b if b < n_sents]
            if not gold_b:
                continue

            v_emb = vis_entry["visual_embeddings"][:n_sents].unsqueeze(0).to(device)
            v_shifts = vis_entry["visual_shifts"][:n_sents].unsqueeze(0).to(device)

            mask = torch.ones(1, n_sents, device=device)
            mask[:, 0] = 0.0
            gold_tensor = torch.tensor(gold_b, device=device)

            opt_stage2.zero_grad()
            out = model(
                sentence_reps=text_reps,
                sentence_mask=mask,
                visual_embeddings=v_emb,
                visual_shifts=v_shifts,
            )
            loss = loss_fn(out.boundary_logits, gold_tensor, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage2_trainable, max_norm=1.0)
            opt_stage2.step()
            sched_stage2.step()

            total_loss += loss.item()

        t_epoch_dur = time.time() - t_epoch
        avg_loss = total_loss / max(1, len(train_items))
        val_m = evaluate(model, val_manifest, val_visual_cache, val_text_cache, device, threshold=args.val_threshold, max_sentences=args.max_sentences)

        logger.info(
            f"[Stage 2 | Epoch {epoch}/{args.stage2_epochs} (Total Ep {global_ep})] ({t_epoch_dur:.1f}s) - Loss: {avg_loss:.4f} | "
            f"Strict F1: {val_m['strict_f1']*100:.2f}% | Relaxed F1: {val_m['relaxed_f1']*100:.2f}% | "
            f"W2 F1: {val_m['w2_f1']*100:.2f}% | Prec: {val_m['prec']*100:.2f}% | Rec: {val_m['rec']*100:.2f}%"
        )
        history.append({"stage": 2, "epoch": epoch, "loss": avg_loss, "val_metrics": val_m, "dur": t_epoch_dur})

        if val_m["relaxed_f1"] > best_relaxed_f1:
            best_relaxed_f1 = val_m["relaxed_f1"]
            torch.save({"epoch": global_ep, "stage": 2, "model_state_dict": model.state_dict(), "model_config": cfg.__dict__, "best_relaxed_f1": best_relaxed_f1, "val_metrics": val_m}, out_dir / "best_model.pt")
            torch.save(model.state_dict(), out_dir / "stage2_best.pt")
            logger.info(f"--> [Saved Best Stage 2] Relaxed F1: {best_relaxed_f1*100:.2f}%")

    # =========================================================================
    # STAGE 3: End-to-End LoRA on Qwen3-Embedding + DBCA + Gating
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info(f"=== STAGE 3: END-TO-END LORA FINE-TUNING (Epochs 1 to {args.stage3_epochs}) ===")
    logger.info("=" * 80)

    # Attach PEFT LoRA to Qwen3-Embedding
    text_extractor.apply_lora(r=16, alpha=32)
    text_extractor.unfreeze_lora_only()

    # Unfreeze full DBCA and multimodal model
    for param in model.parameters():
        param.requires_grad = True

    lora_params = [p for p in text_extractor.parameters() if p.requires_grad]
    dbca_params = list(model.parameters())
    logger.info(f"Stage 3 Configured: LoRA params: {sum(p.numel() for p in lora_params):,}, DBCA params: {sum(p.numel() for p in dbca_params):,}")

    param_groups = [
        {"params": lora_params, "lr": args.stage3_lora_lr, "weight_decay": 0.01},
        {"params": dbca_params, "lr": args.stage3_dbca_lr, "weight_decay": 0.01},
    ]

    if HAS_BNB:
        opt_stage3 = bnb.optim.PagedAdamW8bit(param_groups)
        logger.info("Using bitsandbytes PagedAdamW8bit for Stage 3 autograd.")
    else:
        opt_stage3 = AdamW(param_groups)
        logger.info("Using standard AdamW for Stage 3 autograd.")

    sched_stage3 = CosineAnnealingLR(opt_stage3, T_max=args.stage3_epochs * len(train_items), eta_min=1e-6)

    for epoch in range(1, args.stage3_epochs + 1):
        global_ep = args.stage1_epochs + args.stage2_epochs + epoch
        model.train()
        text_extractor.train()
        total_loss = 0.0
        t_epoch = time.time()
        np.random.shuffle(train_items)

        for step, item in enumerate(train_items):
            vid = item["video_id"]
            s_idx = item["shard_idx"]
            r_idx = item["row_idx"]
            table = train_tables[s_idx]
            sents = table["text_ref"][r_idx].as_py()
            if isinstance(sents, str):
                sents = [sents]

            vis_entry = train_visual_cache[vid]
            n_sents = min(len(sents), len(vis_entry["visual_embeddings"]), args.max_sentences)
            sents = sents[:n_sents]
            gold_b = [b for b in item["gold_boundaries"] if b < n_sents]
            if not gold_b:
                continue

            v_emb = vis_entry["visual_embeddings"][:n_sents].unsqueeze(0).to(device)
            v_shifts = vis_entry["visual_shifts"][:n_sents].unsqueeze(0).to(device)

            mask = torch.ones(1, n_sents, device=device)
            mask[:, 0] = 0.0
            gold_tensor = torch.tensor(gold_b, device=device)

            opt_stage3.zero_grad()

            # End-to-end forward pass with chunked checkpointing through LoRA
            text_reps = text_extractor(sents)  # [1, N, 1024]
            out = model(
                sentence_reps=text_reps,
                sentence_mask=mask,
                visual_embeddings=v_emb,
                visual_shifts=v_shifts,
            )
            loss = loss_fn(out.boundary_logits, gold_tensor, mask=mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(lora_params + dbca_params, max_norm=1.0)
            opt_stage3.step()
            sched_stage3.step()

            total_loss += loss.item()

            if (step + 1) % 150 == 0 or (step + 1) == len(train_items):
                cur_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
                logger.info(f"Stage 3 Step [{step+1}/{len(train_items)}] - Loss: {loss.item():.4f} | Peak VRAM: {cur_vram:.1f} MB")

        t_epoch_dur = time.time() - t_epoch
        avg_loss = total_loss / max(1, len(train_items))

        # Recompute validation text representations with updated LoRA weights
        text_extractor.eval()
        val_text_cache_lora = {}
        with torch.no_grad():
            for it in val_manifest:
                vid = it["video_id"]
                if vid not in val_visual_cache:
                    continue
                d_idx = it["doc_idx"]
                sents_val = val_table["text_ref"][d_idx].as_py()
                if isinstance(sents_val, str):
                    sents_val = [sents_val]
                n_sents_val = min(len(sents_val), len(val_visual_cache[vid]["visual_embeddings"]), args.max_sentences)
                val_text_cache_lora[vid] = text_extractor(sents_val[:n_sents_val]).cpu()

        val_m = evaluate(model, val_manifest, val_visual_cache, val_text_cache_lora, device, threshold=args.val_threshold, max_sentences=args.max_sentences)

        logger.info(
            f"[Stage 3 | Epoch {epoch}/{args.stage3_epochs} (Total Ep {global_ep})] ({t_epoch_dur:.1f}s) - Loss: {avg_loss:.4f} | "
            f"Strict F1: {val_m['strict_f1']*100:.2f}% | Relaxed F1: {val_m['relaxed_f1']*100:.2f}% | "
            f"W2 F1: {val_m['w2_f1']*100:.2f}% | Prec: {val_m['prec']*100:.2f}% | Rec: {val_m['rec']*100:.2f}%"
        )
        history.append({"stage": 3, "epoch": epoch, "loss": avg_loss, "val_metrics": val_m, "dur": t_epoch_dur})

        if val_m["relaxed_f1"] > best_relaxed_f1:
            best_relaxed_f1 = val_m["relaxed_f1"]
            torch.save({
                "epoch": global_ep,
                "stage": 3,
                "model_state_dict": model.state_dict(),
                "model_config": cfg.__dict__,
                "best_relaxed_f1": best_relaxed_f1,
                "val_metrics": val_m,
                "extractor_lora_state_dict": {k: v.cpu() for k, v in text_extractor.backbone.state_dict().items() if "lora_" in k},
            }, out_dir / "best_model.pt")
            logger.info(f"--> [Saved Best Stage 3 Record] Relaxed F1: {best_relaxed_f1*100:.2f}%")

    # =========================================================================
    # FINAL THRESHOLD SWEEP ON BEST MODEL
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("=== RUNNING FINAL THRESHOLD SWEEP ON BEST CHECKPOINT ===")
    logger.info("=" * 80)

    best_ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    active_val_text = val_text_cache_lora if (best_ckpt.get("stage", 1) == 3 and "val_text_cache_lora" in locals()) else val_text_cache

    thresholds = [0.16, 0.18, 0.20, 0.21, 0.22, 0.23, 0.24, 0.26]
    sweep_results = {}
    for tau in thresholds:
        m = evaluate(model, val_manifest, val_visual_cache, active_val_text, device, threshold=tau, max_sentences=args.max_sentences)
        sweep_results[str(tau)] = m
        logger.info(
            f"Tau {tau:.2f} -> Strict F1: {m['strict_f1']*100:.2f}% | Relaxed F1: {m['relaxed_f1']*100:.2f}% | "
            f"W2 F1: {m['w2_f1']*100:.2f}% | Prec: {m['prec']*100:.2f}% | Rec: {m['rec']*100:.2f}%"
        )

    final_report = {
        "best_overall_relaxed_f1": best_relaxed_f1,
        "best_checkpoint_stage": best_ckpt.get("stage", 3),
        "best_checkpoint_epoch": best_ckpt.get("epoch", 9),
        "threshold_sweep": sweep_results,
        "training_history": history,
    }

    report_p = out_dir / "final_benchmark_report.json"
    with open(report_p, "w") as f:
        json.dump(final_report, f, indent=2)
    logger.info(f"[Saved] Final Benchmark Report saved to {report_p}")
    logger.info("=== 3-Stage Multimodal Training Complete ===")


if __name__ == "__main__":
    main()
