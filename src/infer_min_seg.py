from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer

from miniseg_core import (
    Chunk,
    MiniSegConfig,
    boundaries_to_chunks,
    find_best_checkpoint,
    load_checkpoint,
    simple_sentence_split,
)


def _load_text_from_input(file_path: str | None, inline_text: str | None) -> str:
    if inline_text and inline_text.strip():
        return inline_text.strip()
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    return (
        "Topic segmentation helps us break a long transcript into meaningful sections. "
        "That makes downstream summarization more accurate. "
        "When the topic changes, the model should create a new chunk. "
        "The final summary can then be generated per topic and merged into a global summary."
    )


def _load_model(
    model_dir: str,
    checkpoint: str | None = None,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = None

    if checkpoint:
        checkpoint_path = Path(checkpoint)
    else:
        model_dir_path = Path(model_dir)
        if model_dir_path.is_file():
            checkpoint_path = model_dir_path
        else:
            checkpoint_path = find_best_checkpoint(model_dir_path)

    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    model, tokenizer, threshold, config, metrics = load_checkpoint(checkpoint_path, device=device)
    return model, tokenizer, threshold, config, metrics, checkpoint_path


def segment_document(
    text: str,
    model,
    tokenizer,
    threshold: float,
    max_sentences: int,
    max_tokens: int,
    min_chunk_sentences: int = 1,
) -> Dict[str, Any]:
    sentences = simple_sentence_split(text)
    if not sentences:
        return {
            "sentences": [],
            "probabilities": [],
            "boundaries": [],
            "chunks": [],
        }

    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].unsqueeze(0).to(device)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(device)
    sentence_mask = torch.ones(1, len(sentences), dtype=torch.bool, device=device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_mask=sentence_mask,
        )
        probs = torch.sigmoid(logits)[0][: len(sentences)].detach().cpu().tolist()

    boundaries = [1]
    boundaries.extend(1 if p >= threshold else 0 for p in probs[1:])
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


def _format_result(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Predicted topic chunks:")
    if not result["chunks"]:
        lines.append("  (no chunks found)")
        return "\n".join(lines)

    for idx, chunk in enumerate(result["chunks"], start=1):
        lines.append(
            f"\n[{idx}] sentences {chunk['start_sentence']} - {chunk['end_sentence']} "
            f"(score={chunk['boundary_score']:.3f})"
        )
        lines.append(chunk["text"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Infer topic segments from a transcript.")
    parser.add_argument("--file", type=str, default=None, help="Path to a transcript text file.")
    parser.add_argument("--text", type=str, default=None, help="Inline transcript text.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(repo_root / "models" / "miniseg"),
        help="Directory containing MiniSeg checkpoints.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to a checkpoint.")
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved threshold.")
    parser.add_argument("--max-sentences", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-chunk-sentences", type=int, default=1)
    parser.add_argument("--json-out", type=str, default=None, help="Write the full segmentation result as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = _load_text_from_input(args.file, args.text)

    model, tokenizer, saved_threshold, config, metrics, checkpoint_path = _load_model(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
    )
    threshold = args.threshold if args.threshold is not None else saved_threshold
    max_sentences = min(args.max_sentences, config.max_sentences)

    result = segment_document(
        text=text,
        model=model,
        tokenizer=tokenizer,
        threshold=threshold,
        max_sentences=max_sentences,
        max_tokens=args.max_tokens,
        min_chunk_sentences=args.min_chunk_sentences,
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    if metrics:
        print(f"Validation metrics: {json.dumps(metrics, indent=2)}")
    print(_format_result(result))

    if args.json_out:
        payload = {
            "checkpoint": str(checkpoint_path),
            "threshold": threshold,
            "config": config.__dict__,
            "metrics": metrics,
            "result": result,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON output to {args.json_out}")


if __name__ == "__main__":
    main()
