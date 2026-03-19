from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from miniseg_core import (
    BoundaryCollator,
    BoundaryDataset,
    collect_predictions,
    evaluate_segmentation,
    find_best_checkpoint,
    load_checkpoint,
    tune_threshold,
)


def _import_load_from_disk():
    """
    Avoid local `datasets/` directory shadowing Hugging Face `datasets` package.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))

    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The `datasets` package is required for evaluation. Install with `pip install datasets`."
        ) from exc
    return load_from_disk


def _build_validation_loader(
    dataset_dir: str,
    tokenizer,
    max_sentences: int,
    max_tokens: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    load_from_disk = _import_load_from_disk()
    dataset = load_from_disk(dataset_dir)
    if "validation" not in dataset:
        raise ValueError("Validation split not found. Expected split name: 'validation'.")

    val_dataset = BoundaryDataset(dataset["validation"], max_sentences=max_sentences)
    collator = BoundaryCollator(tokenizer=tokenizer, max_tokens=max_tokens)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


def _resolve_checkpoint(model_dir: str, checkpoint: str | None) -> Path:
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    model_dir_path = Path(model_dir)
    if model_dir_path.is_file():
        return model_dir_path

    best = find_best_checkpoint(model_dir_path)
    if best is None:
        raise FileNotFoundError(f"No checkpoint found under: {model_dir_path}")
    return best


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate MiniSeg checkpoint on validation split.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(repo_root / "datasets" / "ytseg-dataset"),
        help="Path to saved dataset directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(repo_root / "models" / "miniseg"),
        help="Directory containing MiniSeg checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional direct checkpoint path. If omitted, uses best checkpoint in model-dir.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Output JSON file path. Defaults to <model-dir>/validation_metrics_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = _resolve_checkpoint(args.model_dir, args.checkpoint)
    model, tokenizer, saved_threshold, config, checkpoint_metrics = load_checkpoint(checkpoint_path, device=device)

    val_loader = _build_validation_loader(
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        max_sentences=config.max_sentences,
        max_tokens=config.max_tokens,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    probs_by_doc, labels_by_doc = collect_predictions(model, val_loader, device, desc="eval-valid")
    tuned_threshold, tuned_threshold_meta = tune_threshold(probs_by_doc, labels_by_doc, objective="f1")

    metrics_saved_threshold = evaluate_segmentation(probs_by_doc, labels_by_doc, saved_threshold)
    metrics_tuned_threshold = evaluate_segmentation(probs_by_doc, labels_by_doc, tuned_threshold)
    metrics_default_05 = evaluate_segmentation(probs_by_doc, labels_by_doc, 0.5)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_metrics": checkpoint_metrics,
        "model_config": config.__dict__,
        "num_validation_docs": len(probs_by_doc),
        "saved_threshold": float(saved_threshold),
        "tuned_threshold_from_validation": float(tuned_threshold),
        "tuned_threshold_meta": tuned_threshold_meta,
        "metrics_saved_threshold": metrics_saved_threshold,
        "metrics_tuned_threshold": metrics_tuned_threshold,
        "metrics_threshold_0_5": metrics_default_05,
    }

    if args.metrics_out:
        out_path = Path(args.metrics_out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.model_dir) / f"validation_metrics_{stamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation docs: {len(probs_by_doc)}")
    print(
        "Saved threshold metrics | "
        f"f1={metrics_saved_threshold['f1']:.4f} "
        f"pk={metrics_saved_threshold['pk']:.4f} "
        f"wd={metrics_saved_threshold['window_diff']:.4f}"
    )
    print(
        "Tuned threshold metrics | "
        f"thr={tuned_threshold:.2f} "
        f"f1={metrics_tuned_threshold['f1']:.4f} "
        f"pk={metrics_tuned_threshold['pk']:.4f} "
        f"wd={metrics_tuned_threshold['window_diff']:.4f}"
    )
    print(f"Wrote metrics to: {out_path}")


if __name__ == "__main__":
    main()

