from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from miniseg_core import (
    align_sentences_and_labels,
    evaluate_segmentation,
    extract_labels,
    extract_sentences,
    find_best_checkpoint,
    set_seed,
    tune_threshold,
)


DEFAULT_DEBERTA_MODEL = "microsoft/deberta-v3-base"


@dataclass
class DebertaSegConfig:
    model_name: str = DEFAULT_DEBERTA_MODEL
    max_sentences: int = 256
    max_tokens: int = 256
    left_window: int = 3
    right_window: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    dropout: float = 0.1


def import_load_from_disk():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))

    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The `datasets` package is required for DeBERTa segmentation. Install it with `pip install datasets`."
        ) from exc
    return load_from_disk


def _join_window(sentences: Sequence[str]) -> str:
    text = " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip()).strip()
    return text if text else "[EMPTY]"


def build_boundary_example(
    sentences: Sequence[str],
    boundary_index: int,
    left_window: int,
    right_window: int,
) -> Tuple[str, str]:
    left_start = max(0, boundary_index - left_window)
    left_text = _join_window(sentences[left_start:boundary_index])

    right_end = min(len(sentences), boundary_index + right_window + 1)
    right_text = _join_window(sentences[boundary_index:right_end])
    return left_text, right_text


class WindowBoundaryDataset(Dataset):
    def __init__(
        self,
        split: Sequence[Dict[str, Any]],
        max_sentences: int,
        left_window: int,
        right_window: int,
        skip_first: bool = True,
    ) -> None:
        self.items: List[Dict[str, Any]] = []
        self.doc_count = 0

        for example in split:
            sentences = extract_sentences(example)
            labels = extract_labels(example)
            sentences, labels = align_sentences_and_labels(sentences, labels, max_sentences)
            if len(sentences) <= 1 or len(labels) <= 1:
                continue

            self.doc_count += 1
            start_idx = 1 if skip_first else 0
            for boundary_index in range(start_idx, len(sentences)):
                left_text, right_text = build_boundary_example(
                    sentences=sentences,
                    boundary_index=boundary_index,
                    left_window=left_window,
                    right_window=right_window,
                )
                self.items.append(
                    {
                        "left_text": left_text,
                        "right_text": right_text,
                        "label": int(labels[boundary_index]),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class WindowBoundaryCollator:
    def __init__(self, tokenizer, max_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.float32),
            }

        encoded = self.tokenizer(
            [item["left_text"] for item in batch],
            [item["right_text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        return encoded


def build_doc_level_windows(
    split: Sequence[Dict[str, Any]],
    max_sentences: int,
    left_window: int,
    right_window: int,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for doc_index, example in enumerate(split):
        sentences = extract_sentences(example)
        labels = extract_labels(example)
        sentences, labels = align_sentences_and_labels(sentences, labels, max_sentences)
        if len(sentences) <= 1 or len(labels) <= 1:
            continue

        pairs: List[Tuple[str, str]] = []
        targets: List[int] = []
        for boundary_index in range(1, len(sentences)):
            pairs.append(build_boundary_example(sentences, boundary_index, left_window, right_window))
            targets.append(int(labels[boundary_index]))

        docs.append(
            {
                "doc_index": doc_index,
                "video_id": example.get("video_id"),
                "channel_id": example.get("channel_id"),
                "sentences": sentences,
                "labels": [1] + targets,
                "pairs": pairs,
            }
        )
    return docs


def collect_doc_probabilities(
    model,
    tokenizer,
    docs: Sequence[Dict[str, Any]],
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    desc: str = "deberta-eval",
) -> Tuple[List[List[float]], List[List[int]]]:
    model.eval()
    all_probs: List[List[float]] = []
    all_labels: List[List[int]] = []

    with torch.no_grad():
        progress = tqdm(docs, desc=desc, leave=False, dynamic_ncols=True)
        for doc in progress:
            pairs = doc["pairs"]
            labels = doc["labels"]
            doc_probs = [1.0]

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
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                doc_probs.extend(float(prob) for prob in probs)

            all_probs.append(doc_probs[: len(labels)])
            all_labels.append(labels[: len(doc_probs)])

        progress.close()

    return all_probs, all_labels


def save_deberta_checkpoint(
    path: str | os.PathLike[str],
    model,
    config: DebertaSegConfig,
    threshold: float,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "threshold": float(threshold),
        "metrics": metrics,
    }
    torch.save(payload, str(path))


def load_deberta_checkpoint(
    path: str | os.PathLike[str],
    device: Optional[torch.device] = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(path), map_location="cpu")
    config = DebertaSegConfig(**payload["config"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=1,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    threshold = float(payload.get("threshold", 0.5))
    metrics = payload.get("metrics", {})
    return model, tokenizer, threshold, config, metrics


def resolve_best_checkpoint(model_dir: str | os.PathLike[str], checkpoint: Optional[str] = None) -> Path:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    model_dir_path = Path(model_dir)
    if model_dir_path.is_file():
        return model_dir_path

    best = find_best_checkpoint(model_dir_path)
    if best is None:
        raise FileNotFoundError(f"No checkpoint found under: {model_dir_path}")
    return best
