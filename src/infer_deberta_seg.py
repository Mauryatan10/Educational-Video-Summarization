from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from deberta_seg_core import (
    build_boundary_example,
    load_deberta_checkpoint,
    resolve_best_checkpoint,
)
from miniseg_core import boundaries_to_chunks, simple_sentence_split


def segment_document(
    text: str,
    model,
    tokenizer,
    threshold: float,
    max_sentences: int,
    max_tokens: int,
    left_window: int,
    right_window: int,
    batch_size: int,
    min_chunk_sentences: int,
) -> Dict[str, Any]:
    sentences = simple_sentence_split(text)
    if not sentences:
        return {"sentences": [], "probabilities": [], "boundaries": [], "chunks": []}

    sentences = sentences[:max_sentences]
    probs: List[float] = [1.0]
    device = next(model.parameters()).device

    pairs = [
        build_boundary_example(sentences, boundary_index, left_window, right_window)
        for boundary_index in range(1, len(sentences))
    ]

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = tokenizer(
                [left for left, _ in batch_pairs],
                [right for _, right in batch_pairs],
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            batch_probs = torch.sigmoid(logits).detach().cpu().tolist()
            if isinstance(batch_probs, float):
                batch_probs = [batch_probs]
            probs.extend(float(prob) for prob in batch_probs)

    boundaries = [1] + [1 if prob >= threshold else 0 for prob in probs[1: len(sentences)]]
    chunks = boundaries_to_chunks(
        sentences=sentences,
        probs=probs,
        threshold=threshold,
        min_chunk_sentences=min_chunk_sentences,
    )
    return {
        "sentences": sentences,
        "probabilities": probs,
        "boundaries": boundaries,
        "chunks": [chunk.__dict__ for chunk in chunks],
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Infer topic segments with DeBERTa boundary classifier.")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=str(repo_root / "models" / "deberta-seg"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-chunk-sentences", type=int, default=1)
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.text and args.text.strip():
        text = args.text.strip()
    elif args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        raise ValueError("Pass either --text or --file.")

    checkpoint_path = resolve_best_checkpoint(args.model_dir, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, saved_threshold, config, metrics = load_deberta_checkpoint(checkpoint_path, device=device)
    threshold = args.threshold if args.threshold is not None else saved_threshold

    result = segment_document(
        text=text,
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        max_sentences=config.max_sentences,
        max_tokens=config.max_tokens,
        left_window=config.left_window,
        right_window=config.right_window,
        batch_size=args.batch_size,
        min_chunk_sentences=args.min_chunk_sentences,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    if metrics:
        print(f"Validation metrics: {json.dumps(metrics, indent=2)}")
    for idx, chunk in enumerate(result["chunks"], start=1):
        print(
            f"\n[{idx}] sentences {chunk['start_sentence']} - {chunk['end_sentence']} "
            f"(score={chunk['boundary_score']:.3f})"
        )
        print(chunk["text"])

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "threshold": threshold,
            "config": config.__dict__,
            "metrics": metrics,
            "result": result,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON output to: {out_path}")


if __name__ == "__main__":
    main()

