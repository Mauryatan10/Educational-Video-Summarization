from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from miniseg_core import evaluate_segmentation, set_seed, tune_threshold
from seg_benchmark_pipeline import (
    MODEL_SPECS,
    build_doc_windows,
    collect_doc_probabilities,
    create_model,
    import_load_from_disk,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoints for BERT/DeBERTa/RoBERTa on one split and select the best model."
    )
    parser.add_argument("--dataset-dir", type=str, default=str(repo_root / "datasets" / "ytseg-dataset"))
    parser.add_argument("--models-root", type=str, default=str(repo_root / "models" / "seg-benchmarks"))
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--models", nargs="+", default=["bert", "deberta", "roberta"], choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--retune-threshold", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=str, default=None)
    return parser.parse_args()


def _load_best_checkpoint(models_root: Path, model_key: str) -> Tuple[Dict[str, Any], Path]:
    ckpt_path = models_root / model_key / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint for '{model_key}': {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu")
    return payload, ckpt_path


def _load_split(dataset_dir: str, split: str):
    load_from_disk = import_load_from_disk()
    dataset = load_from_disk(dataset_dir)
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(dataset.keys())}")
    return dataset[split]


def _evaluate_model(
    model_key: str,
    payload: Dict[str, Any],
    split_data,
    device: torch.device,
    eval_batch_size: int,
    retune_threshold: bool,
) -> Dict[str, Any]:
    train_config = payload.get("train_config", {})
    model_config = payload.get("model_config", {})
    model_spec = payload.get("model_spec", {})

    backbone_name = model_spec.get("backbone_name") or model_config.get("backbone_name") or MODEL_SPECS[model_key].backbone_name
    max_sentences = int(train_config.get("max_sentences", 256))
    max_tokens = int(train_config.get("max_tokens", 256))
    left_window = int(train_config.get("left_window", 3))
    right_window = int(train_config.get("right_window", 3))
    dropout = float(model_config.get("dropout", train_config.get("dropout", 0.1)))
    classifier_hidden = int(model_config.get("classifier_hidden", train_config.get("classifier_hidden", 256)))

    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.cls_token

    docs = build_doc_windows(
        split=split_data,
        max_sentences=max_sentences,
        left_window=left_window,
        right_window=right_window,
    )
    model, _ = create_model(
        model_key=model_key,
        backbone_name=backbone_name,
        dropout=dropout,
        classifier_hidden=classifier_hidden,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    probs_by_doc, labels_by_doc = collect_doc_probabilities(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        max_tokens=max_tokens,
        batch_size=eval_batch_size,
        device=device,
        desc=f"eval-{model_key}",
    )

    saved_threshold = float(payload.get("metrics", {}).get("threshold", 0.5))
    threshold = saved_threshold
    if retune_threshold:
        threshold, _ = tune_threshold(probs_by_doc, labels_by_doc, objective="f1")

    metrics = evaluate_segmentation(probs_by_doc, labels_by_doc, threshold=threshold)
    metrics["threshold"] = float(threshold)
    return {
        "model_key": model_key,
        "model_name": backbone_name,
        "model_spec": model_spec,
        "model_config": model_config,
        "train_config": train_config,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_timestamp": payload.get("timestamp"),
        "checkpoint_training_metrics": payload.get("metrics", {}),
        "checkpoint_threshold": saved_threshold,
        "threshold_used": float(threshold),
        "num_docs": len(docs),
        "metrics": metrics,
    }


def _sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    metrics = row["metrics"]
    return (
        float(metrics.get("f1", float("-inf"))),
        -float(metrics.get("pk", float("inf"))),
        -float(metrics.get("window_diff", float("inf"))),
    )


def _sort_key_relaxed(row: Dict[str, Any]) -> Tuple[float, float, float]:
    metrics = row["metrics"]
    return (
        float(metrics.get("relaxed_f1", float("-inf"))),
        -float(metrics.get("pk", float("inf"))),
        -float(metrics.get("window_diff", float("inf"))),
    )


def _print_table(results: List[Dict[str, Any]]) -> None:
    print("\nModel ranking:")
    print(
        f"{'Rank':<4} {'Model':<12} {'F1':>8} {'RelaxF1':>9} {'Precision':>10} {'Recall':>8} {'Pk':>8} {'WinDiff':>10} {'Thr':>6}"
    )
    for idx, row in enumerate(results, start=1):
        metrics = row["metrics"]
        print(
            f"{idx:<4} {row['model_key']:<12} {metrics['f1']:>8.4f} {metrics.get('relaxed_f1', 0.0):>9.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>8.4f} {metrics['pk']:>8.4f} {metrics['window_diff']:>10.4f} {metrics['threshold']:>6.2f}"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_root = Path(args.models_root)
    split_data = _load_split(args.dataset_dir, args.split)

    results: List[Dict[str, Any]] = []
    for model_key in args.models:
        payload, ckpt_path = _load_best_checkpoint(models_root, model_key)
        print(f"Evaluating {model_key} from {ckpt_path}")
        row = _evaluate_model(
            model_key=model_key,
            payload=payload,
            split_data=split_data,
            device=device,
            eval_batch_size=args.eval_batch_size,
            retune_threshold=args.retune_threshold,
        )
        row["checkpoint"] = str(ckpt_path)
        results.append(row)

    ranked = sorted(results, key=_sort_key, reverse=True)
    ranked_relaxed = sorted(results, key=_sort_key_relaxed, reverse=True)
    _print_table(ranked)

    winner = ranked[0]
    winner_relaxed = ranked_relaxed[0]
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "retune_threshold": bool(args.retune_threshold),
        "evaluation_config": {
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
            "models": list(args.models),
        },
        "winner": winner,
        "winner_by_relaxed_f1": winner_relaxed,
        "results": ranked,
    }

    output_file = Path(args.output_file) if args.output_file else models_root / f"best_model_eval_{args.split}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWinner: {winner['model_key']} ({winner['model_name']}) | F1={winner['metrics']['f1']:.4f}")
    print(
        f"Winner by Relaxed-F1: {winner_relaxed['model_key']} ({winner_relaxed['model_name']}) | "
        f"Relaxed-F1={winner_relaxed['metrics'].get('relaxed_f1', 0.0):.4f}"
    )
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
