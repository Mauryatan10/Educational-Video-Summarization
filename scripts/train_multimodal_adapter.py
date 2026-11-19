"""Supervised Training Harness for Multimodal (Text + Vision) Discourse Segmentation.

Trains the learnable Gated Cross-Modal Adapter on 400 Tier-1 Granular training lectures
and validates on the 100 validation benchmark using Soft-Margin Focal Loss.
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

from src.segmentation.losses import SoftMarginFocalLoss
from src.segmentation.model import DiscourseAwareSegmentationConfig, DiscourseAwareSegmentationModel
from src.segmentation.qwen_extractor import QwenSentenceRepresentationExtractor
from src.segmentation.training_harness import non_maximum_suppression_1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/segmentation_qwen_hq/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints/segmentation_multimodal")
    parser.add_argument("--train-manifest", type=str, default="data/ytseg_train_400/manifest_400.json")
    parser.add_argument("--train-cache", type=str, default="data/ytseg_train_400/visual_cache.pt")
    parser.add_argument("--train-text-cache", type=str, default=None, help="Path to train text cache (defaults to same dir as train-cache)")
    parser.add_argument("--val-manifest", type=str, default="data/ytseg_val_100/manifest_100.json")
    parser.add_argument("--val-cache", type=str, default="data/ytseg_val_100/visual_cache.pt")
    parser.add_argument("--val-text-cache", type=str, default=None, help="Path to val text cache (defaults to same dir as val-cache)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dbca-lr", type=float, default=2e-5, help="Learning rate for unfrozen DBCA Layer 2")
    parser.add_argument("--unfreeze-dbca-l2", action="store_true", help="Unfreeze DBCA Layer 2 for cross-modal context fine-tuning")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sentences", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.24)
    parser.add_argument("--nms-radius", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base checkpoint: {args.base_checkpoint} on {device}...")
    ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)

    cfg = ckpt.get("model_config")
    if isinstance(cfg, dict):
        cfg = DiscourseAwareSegmentationConfig(**cfg)
    elif cfg is None:
        cfg = DiscourseAwareSegmentationConfig(hidden_size=1024, adapter_dim=1024)

    model = DiscourseAwareSegmentationModel(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    logger.info(f"Model loaded (missing keys initialized: {len(missing)}, unexpected: {len(unexpected)}).")
    model.to(device)

    # Parameter selection with differential learning rate
    adapter_params = []
    dbca_l2_params = []
    for name, param in model.named_parameters():
        if "visual_proj" in name or "gate" in name or "boundary_head" in name:
            param.requires_grad = True
            adapter_params.append(param)
        elif args.unfreeze_dbca_l2 and ("dbca.layers.1." in name or "dbca.final_norm" in name):
            param.requires_grad = True
            dbca_l2_params.append(param)
        else:
            param.requires_grad = False

    trainable_params = adapter_params + dbca_l2_params
    logger.info(
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} "
        f"(Adapter: {sum(p.numel() for p in adapter_params):,}, "
        f"DBCA Layer 2: {sum(p.numel() for p in dbca_l2_params):,})"
    )

    logger.info("Loading text sentence extractor (Qwen 3.5 0.8B 4-bit)...")
    text_extractor = QwenSentenceRepresentationExtractor("Qwen/Qwen3.5-0.8B", load_in_4bit=True, device=device).eval()
    if "sentence_extractor_state_dict" in ckpt:
        try:
            text_extractor.load_state_dict(ckpt["sentence_extractor_state_dict"])
        except Exception:
            pass

    # Load Arrow shards
    train_shards = sorted(glob.glob("datasets/ytseg-dataset/train/*.arrow"))
    train_tables = []
    for tf in train_shards:
        with pa.memory_map(tf, "r") as src:
            train_tables.append(ipc.open_stream(src).read_all())

    val_table = None
    val_file = "datasets/ytseg-dataset/validation/data-00000-of-00001.arrow"
    with pa.memory_map(val_file, "r") as src:
        val_table = ipc.open_stream(src).read_all()

    # Load manifests & visual caches
    with open(args.train_manifest) as f:
        train_manifest = json.load(f)
    with open(args.val_manifest) as f:
        val_manifest = json.load(f)

    if not Path(args.train_cache).exists():
        logger.warning(f"Train cache {args.train_cache} not found yet. Ensure precompute_visual_cache.py runs first.")
        return

    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    logger.info(f"Loaded train visual cache: {len(train_cache)} entries.")

    val_cache = {}
    if Path(args.val_cache).exists():
        val_cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
        logger.info(f"Loaded val visual cache: {len(val_cache)} entries.")

    # Filter training items present in cache
    train_items = [it for it in train_manifest if it["video_id"] in train_cache]
    logger.info(f"Eligible cached training samples: {len(train_items)}")

    # Loss function (Soft-Margin Focal Loss)
    loss_fn = SoftMarginFocalLoss(sigma=0.5, gamma=2.0, alpha=0.25)

    param_groups = [
        {"params": adapter_params, "lr": args.lr, "weight_decay": 0.01}
    ]
    if dbca_l2_params:
        param_groups.append({
            "params": dbca_l2_params,
            "lr": args.dbca_lr,
            "weight_decay": 0.01
        })

    optimizer = AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_items), eta_min=1e-6)

    best_val_f1 = -1.0
    train_text_cache_path = Path(args.train_text_cache) if args.train_text_cache else Path(args.train_cache).parent / "text_cache.pt"
    val_text_cache_path = Path(args.val_text_cache) if args.val_text_cache else Path(args.val_cache).parent / "text_cache.pt"
    text_cache: dict[str, torch.Tensor] = {}
    if train_text_cache_path.exists():
        try:
            text_cache = torch.load(train_text_cache_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded existing train text cache: {len(text_cache)} entries.")
        except Exception:
            pass

    val_text_cache: dict[str, torch.Tensor] = {}
    if val_text_cache_path.exists():
        try:
            val_text_cache = torch.load(val_text_cache_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded existing val text cache: {len(val_text_cache)} entries.")
        except Exception:
            pass

    logger.info("=== Starting Multimodal Supervised Training Loop ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
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

            # Harmonize length with visual cache
            vis_entry = train_cache[vid]
            n_sents = min(len(sents), len(vis_entry["visual_embeddings"]), args.max_sentences)
            sents = sents[:n_sents]

            gold_b = [b for b in item["gold_boundaries"] if b < n_sents]
            if not gold_b:
                continue

            # Visual cache inputs
            v_emb = vis_entry["visual_embeddings"][:n_sents].unsqueeze(0).to(device)  # [1, N, 1024]
            v_shifts = vis_entry["visual_shifts"][:n_sents].unsqueeze(0).to(device)    # [1, N]

            # Text representations with memory caching
            if vid in text_cache:
                text_reps = text_cache[vid][:n_sents].to(device)
            else:
                with torch.no_grad():
                    text_reps = text_extractor(sents)
                    text_cache[vid] = text_reps.cpu()

            mask = torch.ones(1, n_sents, device=device)
            mask[:, 0] = 0.0  # Position 0 is never a topic boundary
            gold_tensor = torch.tensor(gold_b, device=device)

            optimizer.zero_grad()
            out = model(
                sentence_reps=text_reps,
                sentence_mask=mask,
                visual_embeddings=v_emb,
                visual_shifts=v_shifts,
            )

            loss = 10.0 * loss_fn(
                logits=out.boundary_logits,
                gold_boundaries=gold_tensor,
                mask=mask,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_items):
                if len(text_cache) > 0 and (step + 1) % 100 == 0:
                    torch.save(text_cache, train_text_cache_path)
                logger.info(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_items)}] "
                    f"Loss: {loss.item():.4f} (Avg: {total_loss/(step+1):.4f}) | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        # Save complete text cache
        torch.save(text_cache, train_text_cache_path)
        epoch_loss = total_loss / max(1, len(train_items))
        logger.info(f"--- Epoch {epoch} Complete. Train Loss: {epoch_loss:.4f} (Time: {time.time()-t_epoch:.1f}s) ---")

        # Evaluate on validation cache
        if val_cache and val_table is not None:
            model.eval()
            val_results = []
            with torch.no_grad():
                for v_item in val_manifest:
                    v_vid = v_item["video_id"]
                    if v_vid not in val_cache:
                        continue
                    v_doc_idx = v_item["doc_idx"]
                    v_sents = val_table["text_ref"][v_doc_idx].as_py()
                    if isinstance(v_sents, str):
                        v_sents = [v_sents]

                    v_entry = val_cache[v_vid]
                    v_n_sents = min(len(v_sents), len(v_entry["visual_embeddings"]), args.max_sentences)
                    v_sents = v_sents[:v_n_sents]
                    v_gold_b = [b for b in v_item["gold_boundaries"] if b < v_n_sents]
                    if not v_gold_b:
                        continue

                    if v_vid in val_text_cache:
                        v_reps = val_text_cache[v_vid][:v_n_sents].to(device)
                    else:
                        v_reps = text_extractor(v_sents)
                        val_text_cache[v_vid] = v_reps.cpu()

                    v_mask = torch.ones(1, v_n_sents, device=device)
                    v_emb_val = v_entry["visual_embeddings"][:v_n_sents].unsqueeze(0).to(device)
                    v_shift_val = v_entry["visual_shifts"][:v_n_sents].unsqueeze(0).to(device)

                    v_out = model(
                        sentence_reps=v_reps,
                        sentence_mask=v_mask,
                        visual_embeddings=v_emb_val,
                        visual_shifts=v_shift_val,
                    )
                    v_preds = non_maximum_suppression_1d(
                        v_out.boundary_probabilities[0].cpu(),
                        threshold=args.threshold,
                        radius=args.nms_radius,
                    )
                    v_preds = [p for p in v_preds if p > 0]
                    m = compute_metrics(v_preds, v_gold_b, tolerance=1)
                    val_results.append(m["f1"])

            if val_text_cache:
                torch.save(val_text_cache, val_text_cache_path)

            if val_results:
                mean_val_f1 = float(np.mean(val_results)) * 100.0
                logger.info(f"★ Epoch {epoch} Validation Relaxed F1 (+-1): {mean_val_f1:.2f}% (over {len(val_results)} docs) ★")
                if mean_val_f1 > best_val_f1:
                    best_val_f1 = mean_val_f1
                    ckpt_save_path = out_dir / "best_model.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "model_config": cfg,
                        "epoch": epoch,
                        "val_f1": best_val_f1,
                    }, ckpt_save_path)
                    logger.info(f"Saved new best model checkpoint to {ckpt_save_path}")

        # Always save last model checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": cfg,
            "epoch": epoch,
            "val_f1": mean_val_f1 if val_results else 0.0,
        }, out_dir / "last_model.pt")

    logger.info(f"=== Training Completed. Best Validation Relaxed F1: {best_val_f1:.2f}% ===")


if __name__ == "__main__":
    main()
