from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from deberta_seg_core import (
    build_doc_level_windows,
    collect_doc_probabilities,
    import_load_from_disk,
    load_deberta_checkpoint,
    resolve_best_checkpoint,
)
from miniseg_core import evaluate_segmentation, tune_threshold


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate DeBERTa boundary classifier on validation split.")
    parser.add_argument("--dataset-dir", type=str, default=str(repo_root / "datasets" / "ytseg-dataset"))
    parser.add_argument("--model-dir", type=str, default=str(repo_root / "models" / "deberta-seg"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--metrics-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_best_checkpoint(args.model_dir, args.checkpoint)
    model, tokenizer, saved_threshold, config, checkpoint_metrics = load_deberta_checkpoint(checkpoint_path, device)

    load_from_disk = import_load_from_disk()
    dataset = load_from_disk(args.dataset_dir)
    docs = build_doc_level_windows(
        split=dataset["validation"],
        max_sentences=config.max_sentences,
        left_window=config.left_window,
        right_window=config.right_window,
    )

    probs_by_doc, labels_by_doc = collect_doc_probabilities(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        max_tokens=config.max_tokens,
        batch_size=args.eval_batch_size,
        device=device,
        desc="deberta-valid",
    )
    tuned_threshold, tuned_meta = tune_threshold(probs_by_doc, labels_by_doc, objective="f1")
    metrics_saved = evaluate_segmentation(probs_by_doc, labels_by_doc, saved_threshold)
    metrics_tuned = evaluate_segmentation(probs_by_doc, labels_by_doc, tuned_threshold)
    metrics_default = evaluate_segmentation(probs_by_doc, labels_by_doc, 0.5)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_metrics": checkpoint_metrics,
        "model_config": config.__dict__,
        "num_validation_docs": len(docs),
        "saved_threshold": saved_threshold,
        "tuned_threshold_from_validation": tuned_threshold,
        "tuned_threshold_meta": tuned_meta,
        "metrics_saved_threshold": metrics_saved,
        "metrics_tuned_threshold": metrics_tuned,
        "metrics_threshold_0_5": metrics_default,
    }

    if args.metrics_out:
        out_path = Path(args.metrics_out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.model_dir) / f"validation_metrics_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation docs: {len(docs)}")
    print(
        "Saved threshold metrics | "
        f"f1={metrics_saved['f1']:.4f} pk={metrics_saved['pk']:.4f} wd={metrics_saved['window_diff']:.4f}"
    )
    print(
        "Tuned threshold metrics | "
        f"thr={tuned_threshold:.2f} f1={metrics_tuned['f1']:.4f} "
        f"pk={metrics_tuned['pk']:.4f} wd={metrics_tuned['window_diff']:.4f}"
    )
    print(f"Wrote metrics to: {out_path}")


if __name__ == "__main__":
    main()

