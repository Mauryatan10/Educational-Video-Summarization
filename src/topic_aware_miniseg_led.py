from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from infer_min_seg import segment_document
from miniseg_core import find_best_checkpoint, load_checkpoint, simple_sentence_split


def load_text(file_path: str | None, inline_text: str | None) -> str:
    if inline_text and inline_text.strip():
        return inline_text.strip()
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    raise ValueError("Pass either --file or --text.")


def load_miniseg_model(model_dir: str, checkpoint: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint:
        ckpt = Path(checkpoint)
    else:
        model_dir_path = Path(model_dir)
        if model_dir_path.is_file():
            ckpt = model_dir_path
        else:
            ckpt = find_best_checkpoint(model_dir_path)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found under: {model_dir}")
    model, tokenizer, threshold, config, metrics = load_checkpoint(ckpt, device=device)
    return model, tokenizer, threshold, config, metrics, ckpt


def load_led_hier_model(
    base_model_name: str,
    hier_dir: str,
    device: str | None = None,
):
    from inference import load_model as load_led_model

    adapter = "hier"
    model, tokenizer = load_led_model(
        base_model_name=base_model_name,
        adapter=adapter,
        peft_dir="",
        hier_dir=hier_dir,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return model, tokenizer


def generate_led_summary(
    text: str,
    model,
    tokenizer,
    min_length: int,
    max_target: int,
) -> str:
    from inference import generate_summary as led_generate_summary

    return led_generate_summary(
        text,
        model=model,
        tokenizer=tokenizer,
        adapter="hier",
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        min_length=min_length,
        max_target=max_target,
    ).strip()


def load_keybert_model(embedding_model: str):
    try:
        from keybert import KeyBERT
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "KeyBERT is required for topic title generation. Install with `pip install keybert`."
        ) from exc
    return KeyBERT(model=embedding_model)


def load_title_t5_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def build_extractive_title(text: str, max_words: int = 8) -> str:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about", "have", "has", "had", "was", "were",
        "are", "but", "not", "can", "will", "their", "there", "they", "them", "then", "than", "also", "just", "very",
        "more", "most", "what", "when", "where", "which", "who", "how", "why", "all", "any", "one", "two", "three",
        "video", "talk", "paper", "discussion", "introduction", "first", "second", "third", "these", "those", "over",
        "you", "your", "yours", "saw", "see", "seen", "kind", "actually", "particular", "here", "there", "lecture",
    }
    tokens = [t.lower() for t in simple_sentence_split(text)]
    flat = []
    for t in tokens:
        flat.extend([w for w in t.split() if w.isalpha()])
    filtered = [w for w in flat if len(w) > 2 and w not in stopwords]
    if not filtered:
        return "Topic Overview"
    counts = Counter(filtered)
    order: List[str] = []
    for token in filtered:
        if token not in order:
            order.append(token)
    ranked = sorted(order, key=lambda token: (-counts[token], order.index(token)))
    picked = ranked[:max(3, min(max_words, len(ranked)))]
    return " ".join(word.capitalize() for word in picked[:max_words]) or "Topic Overview"


def generate_title_keybert(
    chunk_text: str,
    kw_model,
    top_n: int = 3,
    ngram_range: Tuple[int, int] = (1, 3),
) -> str:
    text = normalize_text(chunk_text)
    if not text:
        return "Topic Overview"
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words="english",
        top_n=top_n,
    )
    phrases = [phrase for phrase, _ in keywords if phrase]
    if not phrases:
        return build_extractive_title(chunk_text)
    title = " ".join(phrase.title() for phrase in phrases[:top_n]).strip()
    return title or build_extractive_title(chunk_text)


def generate_title_t5(
    summary: str,
    chunk_text: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    context = " ".join(simple_sentence_split(chunk_text)[:2]).strip()
    prompt = (
        "Generate a concise topic title (5 to 10 words) in third person.\n"
        f"Summary: {summary}\n"
        f"Context: {context}\n"
        "Title:"
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return normalize_text(tokenizer.decode(output_ids[0], skip_special_tokens=True)) or "Topic Overview"


def format_text_output(global_summary: str, topics: List[Dict[str, Any]]) -> str:
    lines = ["Global summary:", "", global_summary, "", "Topics:", ""]
    for topic in topics:
        lines.append(f"[{topic['topic_index']}] {topic['title']}")
        lines.append(topic["summary"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def generate_topic_aware_summaries(
    text: str,
    miniseg_model,
    miniseg_tokenizer,
    miniseg_threshold: float,
    miniseg_max_sentences: int,
    miniseg_max_tokens: int,
    min_chunk_sentences: int,
    led_model,
    led_tokenizer,
    led_min_length: int = 50,
    led_max_target_topic: int = 256,
    led_max_target_global: int = 320,
    title_backend: str = "extractive",
    title_model: str = "google/flan-t5-small",
    title_max_new_tokens: int = 16,
    title_num_beams: int = 4,
    keybert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    keybert_top_n: int = 3,
) -> Dict[str, Any]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return {"global_summary": "", "topics": [], "segmentation": {"chunks": []}}

    segmentation = segment_document(
        text=normalized_text,
        model=miniseg_model,
        tokenizer=miniseg_tokenizer,
        threshold=miniseg_threshold,
        max_sentences=miniseg_max_sentences,
        max_tokens=miniseg_max_tokens,
        min_chunk_sentences=min_chunk_sentences,
    )

    kw_model = None
    title_model_instance = None
    title_tokenizer = None
    title_device = None
    if title_backend == "keybert":
        kw_model = load_keybert_model(keybert_model)
    elif title_backend == "t5":
        title_model_instance, title_tokenizer, title_device = load_title_t5_model(title_model)

    topics: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(segmentation["chunks"], start=1):
        topic_summary = generate_led_summary(
            text=chunk["text"],
            model=led_model,
            tokenizer=led_tokenizer,
            min_length=led_min_length,
            max_target=led_max_target_topic,
        )
        if title_backend == "t5" and title_model_instance is not None:
            topic_title = generate_title_t5(
                summary=topic_summary,
                chunk_text=chunk["text"],
                model=title_model_instance,
                tokenizer=title_tokenizer,
                device=title_device,
                max_new_tokens=title_max_new_tokens,
                num_beams=title_num_beams,
            )
        elif title_backend == "keybert" and kw_model is not None:
            topic_title = generate_title_keybert(
                chunk_text=chunk["text"],
                kw_model=kw_model,
                top_n=keybert_top_n,
            )
        else:
            topic_title = build_extractive_title(chunk["text"])
        topics.append(
            {
                "topic_index": idx,
                "title": topic_title,
                "summary": topic_summary,
                "start_sentence": chunk["start_sentence"],
                "end_sentence": chunk["end_sentence"],
                "boundary_score": float(chunk["boundary_score"]),
            }
        )

    global_summary = generate_led_summary(
        text=normalized_text,
        model=led_model,
        tokenizer=led_tokenizer,
        min_length=max(40, led_min_length),
        max_target=led_max_target_global,
    )

    return {
        "global_summary": global_summary,
        "topics": topics,
        "segmentation": segmentation,
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate topic summaries using MiniSeg + LED-Hier.")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--miniseg-model-dir", type=str, default=str(repo_root / "models" / "miniseg"))
    parser.add_argument("--miniseg-checkpoint", type=str, default=None)
    parser.add_argument("--miniseg-threshold", type=float, default=None)
    parser.add_argument("--miniseg-max-sentences", type=int, default=512)
    parser.add_argument("--miniseg-max-tokens", type=int, default=128)
    parser.add_argument("--min-chunk-sentences", type=int, default=2)
    parser.add_argument("--led-base-model-name", type=str, default="allenai/led-base-16384")
    parser.add_argument(
        "--led-hier-dir",
        type=str,
        default=str(repo_root / "models" / "led-tib-hier" / "led_hier_adapter_run"),
    )
    parser.add_argument("--led-min-length", type=int, default=50)
    parser.add_argument("--led-max-target-topic", type=int, default=256)
    parser.add_argument("--led-max-target-global", type=int, default=320)
    parser.add_argument("--title-backend", type=str, choices=["t5", "keybert", "extractive"], default="t5")
    parser.add_argument("--title-model", type=str, default="google/flan-t5-small")
    parser.add_argument("--title-max-new-tokens", type=int, default=16)
    parser.add_argument("--title-num-beams", type=int, default=4)
    parser.add_argument("--keybert-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--keybert-top-n", type=int, default=3)
    parser.add_argument(
        "--txt-out",
        type=str,
        default=None,
        help="Optional explicit .txt output path. Default: summaries/miniseg_led_<timestamp>.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = load_text(args.file, args.text)

    miniseg_model, miniseg_tokenizer, saved_threshold, config, metrics, ckpt = load_miniseg_model(
        model_dir=args.miniseg_model_dir,
        checkpoint=args.miniseg_checkpoint,
    )
    threshold = args.miniseg_threshold if args.miniseg_threshold is not None else saved_threshold
    led_model, led_tokenizer = load_led_hier_model(
        base_model_name=args.led_base_model_name,
        hier_dir=args.led_hier_dir,
    )

    result = generate_topic_aware_summaries(
        text=text,
        miniseg_model=miniseg_model,
        miniseg_tokenizer=miniseg_tokenizer,
        miniseg_threshold=threshold,
        miniseg_max_sentences=min(args.miniseg_max_sentences, config.max_sentences),
        miniseg_max_tokens=args.miniseg_max_tokens,
        min_chunk_sentences=args.min_chunk_sentences,
        led_model=led_model,
        led_tokenizer=led_tokenizer,
        led_min_length=args.led_min_length,
        led_max_target_topic=args.led_max_target_topic,
        led_max_target_global=args.led_max_target_global,
        title_backend=args.title_backend,
        title_model=args.title_model,
        title_max_new_tokens=args.title_max_new_tokens,
        title_num_beams=args.title_num_beams,
        keybert_model=args.keybert_model,
        keybert_top_n=args.keybert_top_n,
    )

    output = format_text_output(result["global_summary"], result["topics"])

    repo_root = Path(__file__).resolve().parent.parent
    if args.txt_out:
        out_path = Path(args.txt_out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = repo_root / "summaries" / f"miniseg_led_{stamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")

    print(f"MiniSeg checkpoint: {ckpt}")
    if metrics:
        print(f"MiniSeg metrics: {metrics}")
    print(f"Wrote text output to: {out_path}")


if __name__ == "__main__":
    main()
