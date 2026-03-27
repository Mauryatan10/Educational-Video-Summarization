from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from infer_deberta_seg import segment_document as segment_with_deberta
from miniseg_core import simple_sentence_split


def load_text(file_path: str | None, inline_text: str | None) -> str:
    if inline_text and inline_text.strip():
        return inline_text.strip()
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    raise ValueError("Pass either --file or --text.")


def load_deberta_segmenter(model_dir: str, checkpoint: str | None):
    from deberta_seg_core import load_deberta_checkpoint, resolve_best_checkpoint

    checkpoint_path = resolve_best_checkpoint(model_dir, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, saved_threshold, config, metrics = load_deberta_checkpoint(checkpoint_path, device=device)
    return model, tokenizer, saved_threshold, config, metrics, checkpoint_path


def simple_size_chunks(text: str, max_sentences: int) -> Dict[str, Any]:
    sentences = simple_sentence_split(text)
    chunks: List[Dict[str, Any]] = []
    for start in range(0, len(sentences), max(1, max_sentences)):
        end = min(len(sentences), start + max(1, max_sentences))
        chunk_sentences = sentences[start:end]
        if not chunk_sentences:
            continue
        chunks.append(
            {
                "start_sentence": start,
                "end_sentence": end - 1,
                "boundary_score": 1.0 if start == 0 else 0.0,
                "text": " ".join(chunk_sentences).strip(),
            }
        )
    return {
        "sentences": sentences,
        "probabilities": [1.0] * len(sentences),
        "boundaries": [1 if i % max(1, max_sentences) == 0 else 0 for i in range(len(sentences))],
        "chunks": chunks,
    }


def load_summarizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def chunk_by_tokens(
    text: str,
    tokenizer,
    max_source_tokens: int,
) -> List[str]:
    sentences = simple_sentence_split(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        token_count = len(tokenizer(candidate, truncation=False)["input_ids"])
        if current and token_count > max_source_tokens:
            chunks.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def generate_text(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_source_tokens: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_source_tokens,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def build_detail_prompt(text: str) -> str:
    return (
        "You are summarizing one topic segment from a long educational video transcript.\n"
        "Write a detailed explanatory summary that keeps the important concepts, reasoning, and examples.\n"
        "Then provide a short topic title.\n\n"
        "Return exactly in this format:\n"
        "Title: <short topic title>\n"
        "Summary: <detailed explanatory summary>\n\n"
        f"Transcript segment:\n{text}"
    )


def build_reduce_prompt(partials: Sequence[str]) -> str:
    joined = "\n\n".join(f"- {item}" for item in partials if item.strip())
    return (
        "You are combining partial summaries of the same topic from an educational video transcript.\n"
        "Merge them into one coherent, detailed explanation without repetition.\n"
        "Then provide a short topic title.\n\n"
        "Return exactly in this format:\n"
        "Title: <short topic title>\n"
        "Summary: <detailed explanatory summary>\n\n"
        f"Partial summaries:\n{joined}"
    )


def build_global_prompt(topic_items: Sequence[Dict[str, Any]]) -> str:
    topic_lines = []
    for idx, item in enumerate(topic_items, start=1):
        topic_lines.append(f"Topic {idx}: {item['title']}\n{item['summary']}")
    joined = "\n\n".join(topic_lines)
    return (
        "You are summarizing a full educational video using topic summaries.\n"
        "Write a global summary that explains the main progression of ideas across the video.\n"
        "Keep it concise but informative, and mention the major topics in order.\n\n"
        f"Topic summaries:\n{joined}"
    )


def parse_title_and_summary(text: str) -> Tuple[str, str]:
    title_match = re.search(r"Title:\s*(.+)", text, flags=re.IGNORECASE)
    summary_match = re.search(r"Summary:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else ""
    summary = summary_match.group(1).strip() if summary_match else text.strip()
    if not title:
        title = summary.split(".")[0][:80].strip() or "Untitled Topic"
    return title, summary


def summarize_topic_chunk(
    chunk_text: str,
    model,
    tokenizer,
    device: torch.device,
    max_source_tokens: int,
    max_new_tokens: int,
    num_beams: int,
) -> Dict[str, Any]:
    text_parts = chunk_by_tokens(chunk_text, tokenizer, max_source_tokens=max(64, max_source_tokens - 128))
    partials: List[str] = []

    if not text_parts:
        return {"title": "Empty Topic", "summary": ""}

    if len(text_parts) == 1:
        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=build_detail_prompt(text_parts[0]),
            max_source_tokens=max_source_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        title, summary = parse_title_and_summary(raw)
        return {"title": title, "summary": summary, "partials": [raw]}

    progress = tqdm(text_parts, desc="topic-parts", leave=False, dynamic_ncols=True)
    for part in progress:
        partial = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=build_detail_prompt(part),
            max_source_tokens=max_source_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        partials.append(partial)
    progress.close()

    combined_raw = generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=build_reduce_prompt(partials),
        max_source_tokens=max_source_tokens,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    title, summary = parse_title_and_summary(combined_raw)
    return {"title": title, "summary": summary, "partials": partials, "combined_raw": combined_raw}


def format_markdown(result: Dict[str, Any]) -> str:
    lines = ["# Topic-Aware Summary", "", "## Global Summary", result["global_summary"], "", "## Topics"]
    for idx, topic in enumerate(result["topics"], start=1):
        lines.append(f"### {idx}. {topic['title']}")
        lines.append(
            f"Sentences: {topic['start_sentence']} - {topic['end_sentence']} | "
            f"Boundary score: {topic['boundary_score']:.3f}"
        )
        lines.append("")
        lines.append(topic["summary"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate topic-aware summaries from a long transcript.")
    parser.add_argument("--file", type=str, default=None, help="Path to a transcript text file.")
    parser.add_argument("--text", type=str, default=None, help="Inline transcript text.")
    parser.add_argument(
        "--segmentation-mode",
        type=str,
        choices=("deberta", "none"),
        default="deberta",
        help="Use a trained DeBERTa segmenter or simple fixed-size chunks.",
    )
    parser.add_argument("--segmenter-model-dir", type=str, default=str(repo_root / "models" / "deberta-seg"))
    parser.add_argument("--segmenter-checkpoint", type=str, default=None)
    parser.add_argument("--segmenter-threshold", type=float, default=None)
    parser.add_argument("--segmenter-batch-size", type=int, default=16)
    parser.add_argument("--min-chunk-sentences", type=int, default=2)
    parser.add_argument("--fallback-chunk-sentences", type=int, default=12)
    parser.add_argument("--summarizer-model", type=str, default="google/flan-t5-base")
    parser.add_argument("--max-source-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens-topic", type=int, default=220)
    parser.add_argument("--max-new-tokens-global", type=int, default=220)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--md-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = load_text(args.file, args.text)

    if args.segmentation_mode == "deberta":
        seg_model, seg_tokenizer, saved_threshold, seg_config, seg_metrics, seg_checkpoint = load_deberta_segmenter(
            args.segmenter_model_dir,
            args.segmenter_checkpoint,
        )
        threshold = args.segmenter_threshold if args.segmenter_threshold is not None else saved_threshold
        segmentation = segment_with_deberta(
            text=text,
            model=seg_model,
            tokenizer=seg_tokenizer,
            threshold=threshold,
            max_sentences=seg_config.max_sentences,
            max_tokens=seg_config.max_tokens,
            left_window=seg_config.left_window,
            right_window=seg_config.right_window,
            batch_size=args.segmenter_batch_size,
            min_chunk_sentences=args.min_chunk_sentences,
        )
        segmentation_meta = {
            "mode": "deberta",
            "checkpoint": str(seg_checkpoint),
            "threshold": threshold,
            "metrics": seg_metrics,
            "config": seg_config.__dict__,
        }
    else:
        segmentation = simple_size_chunks(text, args.fallback_chunk_sentences)
        segmentation_meta = {
            "mode": "none",
            "strategy": "fixed_size",
            "fallback_chunk_sentences": args.fallback_chunk_sentences,
        }

    summary_model, summary_tokenizer, summary_device = load_summarizer(args.summarizer_model)
    topic_results: List[Dict[str, Any]] = []

    topic_progress = tqdm(segmentation["chunks"], desc="topics", dynamic_ncols=True)
    for idx, chunk in enumerate(topic_progress, start=1):
        topic_progress.set_postfix(topic=f"{idx}/{len(segmentation['chunks'])}")
        topic_summary = summarize_topic_chunk(
            chunk_text=chunk["text"],
            model=summary_model,
            tokenizer=summary_tokenizer,
            device=summary_device,
            max_source_tokens=args.max_source_tokens,
            max_new_tokens=args.max_new_tokens_topic,
            num_beams=args.num_beams,
        )
        topic_results.append(
            {
                "topic_index": idx,
                "title": topic_summary["title"],
                "summary": topic_summary["summary"],
                "start_sentence": chunk["start_sentence"],
                "end_sentence": chunk["end_sentence"],
                "boundary_score": float(chunk["boundary_score"]),
                "text": chunk["text"],
            }
        )
    topic_progress.close()

    global_summary = generate_text(
        model=summary_model,
        tokenizer=summary_tokenizer,
        device=summary_device,
        prompt=build_global_prompt(topic_results),
        max_source_tokens=args.max_source_tokens,
        max_new_tokens=args.max_new_tokens_global,
        num_beams=args.num_beams,
    )

    result = {
        "transcript_length_chars": len(text),
        "num_topics": len(topic_results),
        "segmentation": segmentation_meta,
        "global_summary": global_summary,
        "topics": topic_results,
    }

    print("\nGlobal summary:\n")
    print(global_summary)
    print("\nTopics:\n")
    for topic in topic_results:
        print(f"[{topic['topic_index']}] {topic['title']}")
        print(topic["summary"])
        print("")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote JSON output to: {out_path}")

    if args.md_out:
        out_path = Path(args.md_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_markdown(result), encoding="utf-8")
        print(f"Wrote Markdown output to: {out_path}")


if __name__ == "__main__":
    main()
