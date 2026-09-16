from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MAX_SENTENCES = 512
DEFAULT_MAX_TOKENS = 128


@dataclass
class MiniSegConfig:
    encoder_name: str = DEFAULT_ENCODER_NAME
    max_sentences: int = DEFAULT_MAX_SENTENCES
    max_tokens: int = DEFAULT_MAX_TOKENS
    architecture: str = "baseline"
    context_layers: int = 2
    num_heads: int = 4
    transition_hidden: int = 256
    pairwise_hidden: int = 256
    pairwise_layers: int = 1
    dropout: float = 0.15
    freeze_sentence_encoder: bool = False
    lr_encoder: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01


@dataclass
class Chunk:
    start_sentence: int
    end_sentence: int
    boundary_score: float
    text: str


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def simple_sentence_split(text: str) -> List[str]:
    """
    Sentence split without requiring a network download.
    Uses NLTK if already installed with punkt; otherwise falls back to regex.
    """
    text = (text or "").strip()
    if not text:
        return []

    try:
        import nltk

        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            pass
    except Exception:
        pass

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _clean_label_string(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        labels = []
        for item in value:
            if isinstance(item, (int, float)):
                labels.append(int(item))
            else:
                labels.extend(_clean_label_string(item))
        return labels

    text = str(value).strip()
    if not text:
        return []

    text = text.replace("|=", "").replace("|", "")
    labels = [int(ch) for ch in text if ch in "01"]
    return labels


def _pick_text_field(example: Dict[str, Any], fields: Sequence[str]) -> List[str]:
    for field in fields:
        value = example.get(field)
        if isinstance(value, list):
            cleaned = [str(x).strip() for x in value if str(x).strip()]
            if cleaned:
                return cleaned
        elif isinstance(value, str) and value.strip():
            return simple_sentence_split(value)
    return []


def extract_sentences(example: Dict[str, Any]) -> List[str]:
    sentences = _pick_text_field(example, ("text_ref", "text_wt", "text_wl"))
    if sentences:
        return sentences
    for field in ("transcript", "text", "content"):
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return simple_sentence_split(value)
    return []


def extract_labels(example: Dict[str, Any]) -> List[int]:
    labels = _clean_label_string(example.get("target_binary_ref"))
    if labels:
        return labels
    for field in ("target_binary_wt", "target_binary_wl"):
        labels = _clean_label_string(example.get(field))
        if labels:
            return labels
    return []


def align_sentences_and_labels(
    sentences: Sequence[str],
    labels: Sequence[int],
    max_sentences: int,
) -> Tuple[List[str], List[int]]:
    sentences = [s.strip() for s in sentences if str(s).strip()]
    labels = [int(x) for x in labels if str(x).strip() != ""]
    if not sentences or not labels:
        return [], []

    n = min(len(sentences), len(labels), max_sentences)
    return list(sentences[:n]), list(labels[:n])


def masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    hidden_states: (N, T, H)
    mask: (N, T)
    """
    mask_f = mask.unsqueeze(-1).type_as(hidden_states)
    summed = (hidden_states * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return summed / denom


class BoundaryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Sequence[Dict[str, Any]],
        max_sentences: int = DEFAULT_MAX_SENTENCES,
    ) -> None:
        self.split = split
        self.max_sentences = max_sentences

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.split[idx]
        sentences = extract_sentences(example)
        labels = extract_labels(example)
        sentences, labels = align_sentences_and_labels(sentences, labels, self.max_sentences)
        return {
            "sentences": sentences,
            "labels": labels,
        }


class BoundaryCollator:
    def __init__(
        self,
        tokenizer,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = [ex for ex in batch if ex.get("sentences")]
        if not batch:
            return {
                "input_ids": torch.empty(0, 0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, 0, dtype=torch.long),
                "sentence_mask": torch.empty(0, 0, dtype=torch.bool),
                "labels": torch.empty(0, 0, dtype=torch.float32),
            }

        sentence_counts = [len(ex["sentences"]) for ex in batch]
        max_sentences = max(sentence_counts)

        flat_sentences: List[str] = []
        for ex in batch:
            flat_sentences.extend(ex["sentences"])

        encoded = self.tokenizer(
            flat_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )

        token_len = encoded["input_ids"].size(1)
        input_rows: List[torch.Tensor] = []
        mask_rows: List[torch.Tensor] = []
        label_rows: List[torch.Tensor] = []
        sentence_mask_rows: List[torch.Tensor] = []

        offset = 0
        for count, ex in zip(sentence_counts, batch):
            ids = encoded["input_ids"][offset : offset + count]
            attn = encoded["attention_mask"][offset : offset + count]
            offset += count

            pad_sentences = max_sentences - count
            if pad_sentences > 0:
                ids_pad = torch.full(
                    (pad_sentences, token_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                attn_pad = torch.zeros((pad_sentences, token_len), dtype=torch.long)
                ids = torch.cat([ids, ids_pad], dim=0)
                attn = torch.cat([attn, attn_pad], dim=0)

            labels = torch.tensor(ex["labels"], dtype=torch.float32)
            if labels.numel() < max_sentences:
                labels = torch.cat(
                    [labels, torch.zeros(max_sentences - labels.numel(), dtype=torch.float32)],
                    dim=0,
                )

            sentence_mask = torch.zeros(max_sentences, dtype=torch.bool)
            sentence_mask[:count] = True

            input_rows.append(ids)
            mask_rows.append(attn)
            label_rows.append(labels[:max_sentences])
            sentence_mask_rows.append(sentence_mask)

        return {
            "input_ids": torch.stack(input_rows, dim=0),
            "attention_mask": torch.stack(mask_rows, dim=0),
            "sentence_mask": torch.stack(sentence_mask_rows, dim=0),
            "labels": torch.stack(label_rows, dim=0),
        }


class MiniSegModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER_NAME,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
        architecture: str = "baseline",
        context_layers: int = 2,
        num_heads: int = 4,
        transition_hidden: int = 256,
        pairwise_hidden: int = 256,
        pairwise_layers: int = 1,
        dropout: float = 0.15,
        freeze_sentence_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.max_sentences = max_sentences
        self.architecture = architecture
        self.sentence_encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = int(self.sentence_encoder.config.hidden_size)

        self.pos_embedding = nn.Embedding(max_sentences, hidden_size)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=context_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 32), 1),
        )
        self.transition_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, transition_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(transition_hidden, 1),
        )
        self.pairwise_projection = nn.Linear(hidden_size * 4, pairwise_hidden)
        self.pairwise_dropout = nn.Dropout(dropout)
        self.pairwise_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=pairwise_hidden,
                nhead=max(1, min(num_heads, pairwise_hidden // 64 if pairwise_hidden >= 64 else 1)),
                dim_feedforward=pairwise_hidden * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=max(1, pairwise_layers),
        )
        self.pairwise_classifier = nn.Sequential(
            nn.LayerNorm(pairwise_hidden),
            nn.Dropout(dropout),
            nn.Linear(pairwise_hidden, max(pairwise_hidden // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(pairwise_hidden // 2, 32), 1),
        )

        if freeze_sentence_encoder:
            for param in self.sentence_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.numel() == 0:
            return input_ids.new_zeros((*sentence_mask.shape, 1), dtype=torch.float32).squeeze(-1)

        batch_size, max_sentences, token_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * max_sentences, token_len)
        flat_mask = attention_mask.view(batch_size * max_sentences, token_len)

        outputs = self.sentence_encoder(input_ids=flat_ids, attention_mask=flat_mask)
        sent_repr = masked_mean(outputs.last_hidden_state, flat_mask)
        hidden_size = sent_repr.size(-1)
        sent_repr = sent_repr.view(batch_size, max_sentences, hidden_size)

        pos_ids = torch.arange(max_sentences, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        sent_repr = sent_repr + self.pos_embedding(pos_ids)

        context = self.context_encoder(sent_repr, src_key_padding_mask=~sentence_mask.bool())
        if self.architecture == "transition":
            prev_context = torch.cat([context[:, :1, :], context[:, :-1, :]], dim=1)
            diff_context = context - prev_context
            prod_context = context * prev_context
            transition_features = torch.cat([context, prev_context, diff_context, prod_context], dim=-1)
            logits = self.transition_classifier(transition_features).squeeze(-1)
        elif self.architecture == "pairwise":
            prev_sent = torch.cat([sent_repr[:, :1, :], sent_repr[:, :-1, :]], dim=1)
            pair_features = torch.cat(
                [sent_repr, prev_sent, torch.abs(sent_repr - prev_sent), sent_repr * prev_sent], dim=-1
            )
            pair_repr = self.pairwise_projection(pair_features)
            pair_repr = F.gelu(pair_repr)
            pair_repr = self.pairwise_dropout(pair_repr)
            pair_repr = self.pairwise_encoder(pair_repr, src_key_padding_mask=~sentence_mask.bool())
            logits = self.pairwise_classifier(pair_repr).squeeze(-1)
        else:
            logits = self.classifier(context).squeeze(-1)
        return logits

    def get_head_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params.extend(list(self.pos_embedding.parameters()))
        if self.architecture == "pairwise":
            params.extend(list(self.pairwise_projection.parameters()))
            params.extend(list(self.pairwise_encoder.parameters()))
            params.extend(list(self.pairwise_classifier.parameters()))
        else:
            params.extend(list(self.context_encoder.parameters()))
        if self.architecture == "transition":
            params.extend(list(self.transition_classifier.parameters()))
        elif self.architecture == "baseline":
            params.extend(list(self.classifier.parameters()))
        return params


def boundary_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    ignore_first: bool = True,
) -> Dict[str, float]:
    if len(preds) != len(labels):
        raise ValueError("preds and labels must have the same length")

    start = 1 if ignore_first and len(labels) > 0 else 0
    tp = fp = fn = 0
    for p, y in zip(preds[start:], labels[start:]):
        if p == 1 and y == 1:
            tp += 1
        elif p == 1 and y == 0:
            fp += 1
        elif p == 0 and y == 1:
            fn += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def _boundary_counts(
    preds: Sequence[int],
    labels: Sequence[int],
    ignore_first: bool = True,
) -> Tuple[int, int, int]:
    start = 1 if ignore_first and len(labels) > 0 else 0
    tp = fp = fn = 0
    for p, y in zip(preds[start:], labels[start:]):
        if p == 1 and y == 1:
            tp += 1
        elif p == 1 and y == 0:
            fp += 1
        elif p == 0 and y == 1:
            fn += 1
    return tp, fp, fn


def relaxed_boundary_f1(
    preds: Sequence[int],
    labels: Sequence[int],
    window: int = 2,
    ignore_first: bool = True,
) -> Dict[str, float]:
    if len(preds) != len(labels):
        raise ValueError("preds and labels must have the same length")

    start = 1 if ignore_first and len(labels) > 0 else 0
    pred_indices = [i for i in range(start, len(preds)) if preds[i] == 1]
    true_indices = [i for i in range(start, len(labels)) if labels[i] == 1]

    matched_preds = set()
    matched_trues = set()
    for t_idx in true_indices:
        for p_idx in pred_indices:
            if p_idx in matched_preds:
                continue
            if abs(p_idx - t_idx) <= window:
                matched_trues.add(t_idx)
                matched_preds.add(p_idx)
                break

    tp = len(matched_trues)
    fp = max(0, len(pred_indices) - len(matched_preds))
    fn = max(0, len(true_indices) - len(matched_trues))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "window": float(window),
    }


def _boundary_to_segments(boundaries: Sequence[int]) -> List[int]:
    segment_ids: List[int] = []
    current = 0
    for idx, boundary in enumerate(boundaries):
        if idx == 0:
            current = 0
        elif boundary == 1:
            current += 1
        segment_ids.append(current)
    return segment_ids


def _segment_window_size(boundaries: Sequence[int]) -> int:
    n = len(boundaries)
    if n <= 1:
        return 1
    num_segments = 1 + sum(int(x) for x in boundaries[1:])
    avg_segment_len = n / max(num_segments, 1)
    return max(1, int(round(avg_segment_len / 2.0)))


def pk_score(ref: Sequence[int], hyp: Sequence[int]) -> float:
    if len(ref) != len(hyp):
        raise ValueError("ref and hyp must have the same length")
    if len(ref) < 2:
        return 0.0

    ref_ids = _boundary_to_segments(ref)
    hyp_ids = _boundary_to_segments(hyp)
    k = min(_segment_window_size(ref), len(ref) - 1)
    errors = 0
    total = 0
    for i in range(len(ref) - k):
        ref_same = ref_ids[i] == ref_ids[i + k]
        hyp_same = hyp_ids[i] == hyp_ids[i + k]
        if ref_same != hyp_same:
            errors += 1
        total += 1
    return errors / max(total, 1)


def window_diff(ref: Sequence[int], hyp: Sequence[int]) -> float:
    if len(ref) != len(hyp):
        raise ValueError("ref and hyp must have the same length")
    if len(ref) < 2:
        return 0.0

    k = min(_segment_window_size(ref), len(ref) - 1)
    errors = 0
    total = 0
    for i in range(len(ref) - k):
        ref_count = sum(ref[i + 1 : i + k + 1])
        hyp_count = sum(hyp[i + 1 : i + k + 1])
        if ref_count != hyp_count:
            errors += 1
        total += 1
    return errors / max(total, 1)


def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "eval",
) -> Tuple[List[List[float]], List[List[int]]]:
    model.eval()
    all_probs: List[List[float]] = []
    all_labels: List[List[int]] = []

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch in progress:
            if batch["input_ids"].numel() == 0:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sentence_mask = batch["sentence_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, sentence_mask=sentence_mask)
            probs = torch.sigmoid(logits)

            for i in range(probs.size(0)):
                valid = sentence_mask[i].bool()
                all_probs.append(probs[i][valid].detach().cpu().tolist())
                all_labels.append(labels[i][valid].int().detach().cpu().tolist())

        progress.close()

    return all_probs, all_labels


def tune_threshold(
    probs_by_doc: Sequence[Sequence[float]],
    labels_by_doc: Sequence[Sequence[int]],
    thresholds: Optional[Sequence[float]] = None,
    objective: str = "f1",
) -> Tuple[float, Dict[str, float]]:
    if thresholds is None:
        thresholds = [x / 100.0 for x in range(10, 91)]

    best_threshold = 0.5
    best_metrics = {
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "pk": float("inf"),
        "window_diff": float("inf"),
        "relaxed_f1": -1.0,
    }
    maximize = objective in {"f1", "precision", "recall", "relaxed_f1"}

    for threshold in thresholds:
        tp = fp = fn = 0
        doc_pk: List[float] = []
        doc_wd: List[float] = []
        for probs, labels in zip(probs_by_doc, labels_by_doc):
            if not probs:
                continue
            preds = [1 if p >= threshold else 0 for p in probs]
            if preds:
                preds[0] = 1
            if labels:
                labels = list(labels)
                labels[0] = 1
            c_tp, c_fp, c_fn = _boundary_counts(preds, labels, ignore_first=True)
            tp += c_tp
            fp += c_fp
            fn += c_fn
            doc_pk.append(pk_score(labels, preds))
            doc_wd.append(window_diff(labels, preds))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        relaxed_scores: List[float] = []
        for probs, labels in zip(probs_by_doc, labels_by_doc):
            if not probs:
                continue
            preds = [1 if p >= threshold else 0 for p in probs]
            if preds:
                preds[0] = 1
            if labels:
                labels = list(labels)
                labels[0] = 1
            relaxed_scores.append(relaxed_boundary_f1(preds, labels, window=2)["f1"])
        relaxed_f1 = sum(relaxed_scores) / max(len(relaxed_scores), 1)
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pk": sum(doc_pk) / max(len(doc_pk), 1),
            "window_diff": sum(doc_wd) / max(len(doc_wd), 1),
            "relaxed_f1": relaxed_f1,
        }
        score = metrics.get(objective, f1)
        best_score = best_metrics.get(objective, best_metrics["f1"])
        is_better = score > best_score if maximize else score < best_score
        if is_better:
            best_metrics = metrics
            best_threshold = float(threshold)

    return best_threshold, best_metrics


def evaluate_segmentation(
    probs_by_doc: Sequence[Sequence[float]],
    labels_by_doc: Sequence[Sequence[int]],
    threshold: float,
) -> Dict[str, float]:
    doc_f1: List[float] = []
    doc_relaxed_f1: List[float] = []
    doc_pk: List[float] = []
    doc_wd: List[float] = []
    tp = fp = fn = 0

    for probs, labels in zip(probs_by_doc, labels_by_doc):
        if not probs:
            continue
        preds = [1 if p >= threshold else 0 for p in probs]
        if preds:
            preds[0] = 1
        if labels:
            labels = list(labels)
            labels[0] = 1

        c_tp, c_fp, c_fn = _boundary_counts(preds, labels, ignore_first=True)
        tp += c_tp
        fp += c_fp
        fn += c_fn

        doc_f1.append(boundary_f1(preds, labels)["f1"])
        doc_relaxed_f1.append(relaxed_boundary_f1(preds, labels, window=2)["f1"])
        doc_pk.append(pk_score(labels, preds))
        doc_wd.append(window_diff(labels, preds))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    overall = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
    overall.update(
        {
            "threshold": threshold,
            "doc_f1": sum(doc_f1) / max(len(doc_f1), 1),
            "relaxed_f1": sum(doc_relaxed_f1) / max(len(doc_relaxed_f1), 1),
            "pk": sum(doc_pk) / max(len(doc_pk), 1),
            "window_diff": sum(doc_wd) / max(len(doc_wd), 1),
        }
    )
    return overall


def boundaries_to_chunks(
    sentences: Sequence[str],
    probs: Sequence[float],
    threshold: float,
    min_chunk_sentences: int = 1,
) -> List[Chunk]:
    if not sentences:
        return []

    boundaries = [1]
    for i in range(1, len(sentences)):
        prob = probs[i] if i < len(probs) else 0.0
        boundaries.append(1 if prob >= threshold else 0)

    chunks: List[Chunk] = []
    start = 0
    last_score = 0.0
    for idx in range(1, len(sentences)):
        last_score = probs[idx] if idx < len(probs) else last_score
        if idx < len(boundaries) and boundaries[idx] == 1:
            text = " ".join(sentences[start:idx]).strip()
            if text:
                chunks.append(
                    Chunk(
                        start_sentence=start,
                        end_sentence=idx - 1,
                        boundary_score=float(last_score),
                        text=text,
                    )
                )
            start = idx

    text = " ".join(sentences[start:]).strip()
    if text:
        chunks.append(
            Chunk(
                start_sentence=start,
                end_sentence=len(sentences) - 1,
                boundary_score=float(probs[len(sentences) - 1] if len(probs) >= len(sentences) else last_score),
                text=text,
            )
        )

    if min_chunk_sentences <= 1 or len(chunks) <= 1:
        return chunks

    merged: List[Chunk] = []
    for chunk in chunks:
        if merged and (chunk.end_sentence - chunk.start_sentence + 1) < min_chunk_sentences:
            prev = merged.pop()
            merged.append(
                Chunk(
                    start_sentence=prev.start_sentence,
                    end_sentence=chunk.end_sentence,
                    boundary_score=max(prev.boundary_score, chunk.boundary_score),
                    text=(prev.text + " " + chunk.text).strip(),
                )
            )
        else:
            merged.append(chunk)
    return merged


def save_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    config: MiniSegConfig,
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


def load_checkpoint(
    path: str | os.PathLike[str],
    device: Optional[torch.device] = None,
) -> Tuple[MiniSegModel, AutoTokenizer, float, MiniSegConfig, Dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(str(path), map_location="cpu")

    if "config" in payload:
        config = MiniSegConfig(**payload["config"])
    else:
        config = MiniSegConfig()

    model = MiniSegModel(
        encoder_name=config.encoder_name,
        max_sentences=config.max_sentences,
        architecture=config.architecture,
        context_layers=config.context_layers,
        num_heads=config.num_heads,
        transition_hidden=config.transition_hidden,
        pairwise_hidden=config.pairwise_hidden,
        pairwise_layers=config.pairwise_layers,
        dropout=config.dropout,
        freeze_sentence_encoder=config.freeze_sentence_encoder,
    )
    state_dict = payload.get("model_state_dict") or payload.get("state_dict") or payload
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    threshold = float(payload.get("threshold", 0.5))
    metrics = payload.get("metrics", {})
    return model, tokenizer, threshold, config, metrics


def find_best_checkpoint(model_dir: str | os.PathLike[str]) -> Optional[Path]:
    model_dir = Path(model_dir)
    explicit = model_dir / "best.pt"
    if explicit.exists():
        return explicit

    checkpoints = sorted(model_dir.glob("*.pt"))
    if not checkpoints:
        return None

    best_path = None
    best_score = float("-inf")
    for ckpt in checkpoints:
        try:
            payload = torch.load(str(ckpt), map_location="cpu")
            score = float(payload.get("metrics", {}).get("f1", payload.get("f1_score", -1.0)))
            if score > best_score:
                best_score = score
                best_path = ckpt
        except Exception:
            continue
    return best_path
