from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from seg_benchmark_pipeline import MODEL_SPECS, TrainConfig, compare_models


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train and compare transformer segmentation models on YT-Seg.")
    parser.add_argument("--models", nargs="+", default=["bert", "deberta", "roberta"], choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--dataset-dir", type=str, default=str(repo_root / "datasets" / "ytseg-dataset"))
    parser.add_argument("--output-root", type=str, default=str(repo_root / "models" / "seg-benchmarks"))
    parser.add_argument("--baseline-metrics-file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-sentences", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--left-window", type=int, default=3)
    parser.add_argument("--right-window", type=int, default=3)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier-hidden", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_metrics = _load_json(args.baseline_metrics_file)
    config = TrainConfig(
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_sentences=args.max_sentences,
        max_tokens=args.max_tokens,
        left_window=args.left_window,
        right_window=args.right_window,
        negative_ratio=args.negative_ratio,
        max_train_examples=args.max_train_examples,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        dropout=args.dropout,
        classifier_hidden=args.classifier_hidden,
    )

    comparison = compare_models(model_keys=args.models, config=config, baseline_metrics=baseline_metrics)
    print("Comparison complete.")
    print(f"JSON: {Path(args.output_root) / 'comparison.json'}")
    print(f"Markdown: {Path(args.output_root) / 'comparison.md'}")
    top = comparison["runs"][0]
    print(f"Top model: {top['model_key']} ({top['model_name']}) | F1={top['best_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()
