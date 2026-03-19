from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader

from miniseg_core import (
    BoundaryCollator,
    BoundaryDataset,
    align_sentences_and_labels,
    collect_predictions,
    evaluate_segmentation,
    extract_labels,
    extract_sentences,
    find_best_checkpoint,
    load_checkpoint,
)


def _import_load_from_disk():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))

    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The `datasets` package is required for error analysis. Install with `pip install datasets`."
        ) from exc
    return load_from_disk


def _resolve_checkpoint(model_dir: str, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    model_dir_path = Path(model_dir)
    if model_dir_path.is_file():
        return model_dir_path

    best = find_best_checkpoint(model_dir_path)
    if best is None:
        raise FileNotFoundError(f"No checkpoint found under: {model_dir_path}")
    return best


def _load_validation_split(dataset_dir: str):
    load_from_disk = _import_load_from_disk()
    dataset = load_from_disk(dataset_dir)
    if "validation" not in dataset:
        raise ValueError("Validation split not found. Expected split name: 'validation'.")
    return dataset["validation"]


def _build_loader(
    validation_split,
    tokenizer,
    max_sentences: int,
    max_tokens: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    val_dataset = BoundaryDataset(validation_split, max_sentences=max_sentences)
    collator = BoundaryCollator(tokenizer=tokenizer, max_tokens=max_tokens)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )


def _prepare_docs(validation_split, max_sentences: int) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for idx, example in enumerate(validation_split):
        sentences = extract_sentences(example)
        labels = extract_labels(example)
        sentences, labels = align_sentences_and_labels(sentences, labels, max_sentences)
        if not sentences or not labels:
            continue
        docs.append(
            {
                "doc_index": idx,
                "video_id": example.get("video_id"),
                "channel_id": example.get("channel_id"),
                "sentences": sentences,
                "labels": labels,
            }
        )
    return docs


def _collect_error_cases(
    docs: Sequence[Dict[str, Any]],
    probs_by_doc: Sequence[Sequence[float]],
    threshold: float,
    context_window: int,
    max_cases_per_doc: int,
) -> List[Dict[str, Any]]:
    if len(docs) != len(probs_by_doc):
        raise ValueError(
            f"Doc count mismatch between validation docs ({len(docs)}) and predictions ({len(probs_by_doc)})."
        )

    results: List[Dict[str, Any]] = []
    for doc, probs in zip(docs, probs_by_doc):
        labels = list(doc["labels"])
        if not probs or not labels:
            continue

        n = min(len(labels), len(probs), len(doc["sentences"]))
        labels = labels[:n]
        probs = list(probs[:n])
        sentences = doc["sentences"][:n]

        preds = [1 if p >= threshold else 0 for p in probs]
        preds[0] = 1
        labels[0] = 1

        fp_positions: List[int] = []
        fn_positions: List[int] = []
        for i, (pred, gold) in enumerate(zip(preds, labels)):
            if i == 0:
                continue
            if pred == 1 and gold == 0:
                fp_positions.append(i)
            elif pred == 0 and gold == 1:
                fn_positions.append(i)

        errors: List[Dict[str, Any]] = []
        keep_positions = fp_positions[:max_cases_per_doc] + fn_positions[:max_cases_per_doc]
        for pos in keep_positions:
            left = max(0, pos - context_window)
            right = min(n - 1, pos + context_window)
            context_sentences = []
            for j in range(left, right + 1):
                context_sentences.append(
                    {
                        "sentence_index": j,
                        "text": sentences[j],
                        "gold_boundary": int(labels[j]),
                        "pred_boundary": int(preds[j]),
                        "boundary_prob": float(probs[j]),
                    }
                )

            errors.append(
                {
                    "error_type": "FP" if pos in fp_positions else "FN",
                    "position": pos,
                    "boundary_prob": float(probs[pos]),
                    "gold_boundary": int(labels[pos]),
                    "pred_boundary": int(preds[pos]),
                    "context": context_sentences,
                }
            )

        if errors:
            doc_error_rate = (len(fp_positions) + len(fn_positions)) / max(n - 1, 1)
            results.append(
                {
                    "doc_index": doc["doc_index"],
                    "video_id": doc.get("video_id"),
                    "channel_id": doc.get("channel_id"),
                    "num_sentences": n,
                    "num_fp": len(fp_positions),
                    "num_fn": len(fn_positions),
                    "error_rate": doc_error_rate,
                    "examples": errors,
                }
            )

    results.sort(key=lambda item: (item["error_rate"], item["num_fn"], item["num_fp"]), reverse=True)
    return results


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Analyze MiniSeg boundary errors on validation split.")
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
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=2, help="Neighboring sentences on each side.")
    parser.add_argument(
        "--max-cases-per-doc",
        type=int,
        default=4,
        help="Maximum FP + FN positions to save per document.",
    )
    parser.add_argument(
        "--top-docs",
        type=int,
        default=40,
        help="Number of highest-error validation documents to keep.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <model-dir>/error_analysis_<timestamp>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = _resolve_checkpoint(args.model_dir, args.checkpoint)
    model, tokenizer, saved_threshold, config, checkpoint_metrics = load_checkpoint(checkpoint_path, device=device)
    threshold = float(args.threshold) if args.threshold is not None else float(saved_threshold)

    validation_split = _load_validation_split(args.dataset_dir)
    val_loader = _build_loader(
        validation_split=validation_split,
        tokenizer=tokenizer,
        max_sentences=config.max_sentences,
        max_tokens=config.max_tokens,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    probs_by_doc, labels_by_doc = collect_predictions(model, val_loader, device, desc="error-valid")
    metrics = evaluate_segmentation(probs_by_doc, labels_by_doc, threshold)
    docs = _prepare_docs(validation_split, config.max_sentences)

    error_docs = _collect_error_cases(
        docs=docs,
        probs_by_doc=probs_by_doc,
        threshold=threshold,
        context_window=max(0, args.context_window),
        max_cases_per_doc=max(1, args.max_cases_per_doc),
    )
    error_docs = error_docs[: max(1, args.top_docs)]

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_metrics": checkpoint_metrics,
        "model_config": config.__dict__,
        "threshold_used": threshold,
        "validation_metrics_at_threshold": metrics,
        "num_validation_docs_predicted": len(probs_by_doc),
        "num_error_docs_saved": len(error_docs),
        "error_docs": error_docs,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.model_dir) / f"error_analysis_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Metrics | "
        f"f1={metrics['f1']:.4f} "
        f"pk={metrics['pk']:.4f} "
        f"wd={metrics['window_diff']:.4f} "
        f"thr={threshold:.2f}"
    )
    print(f"Saved {len(error_docs)} docs of FP/FN examples to: {out_path}")


if __name__ == "__main__":
    main()

