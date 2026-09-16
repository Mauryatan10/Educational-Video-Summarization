from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from miniseg_core import (
    align_sentences_and_labels,
    evaluate_segmentation,
    extract_labels,
    extract_sentences,
    set_seed,
    tune_threshold,
)
from seg_benchmark_models import (
    BertBoundaryConfig,
    BertBoundaryModel,
    DebertaBoundaryConfig,
    DebertaBoundaryModel,
    RobertaBoundaryConfig,
    RobertaBoundaryModel,
)


@dataclass
class TrainConfig:
    dataset_dir: str
    output_root: str
    epochs: int = 6
    batch_size: int = 8
    eval_batch_size: int = 32
    max_sentences: int = 256
    max_tokens: int = 256
    left_window: int = 3
    right_window: int = 3
    negative_ratio: float = 3.0
    max_train_examples: int = 0
    seed: int = 42
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    num_workers: int = 2
    dropout: float = 0.1
    classifier_hidden: int = 256


@dataclass
class ModelSpec:
    key: str
    display_name: str
    backbone_name: str


MODEL_SPECS: Dict[str, ModelSpec] = {
    "bert": ModelSpec("bert", "BERT Boundary Classifier", "google-bert/bert-base-uncased"),
    "deberta": ModelSpec("deberta", "DeBERTa-v3 Boundary Classifier", "microsoft/deberta-v3-base"),
    "roberta": ModelSpec("roberta", "RoBERTa Boundary Classifier", "roberta-base"),
}


def import_load_from_disk():
    try:
        from datasets import load_from_disk
    except Exception as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install with `pip install datasets`."
        ) from exc
    return load_from_disk


def _join_window(sentences: Sequence[str]) -> str:
    text = " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip()).strip()
    return text if text else "[EMPTY]"


def build_boundary_pair(
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


class WindowPairDataset(Dataset):
    def __init__(
        self,
        split: Sequence[Dict[str, Any]],
        max_sentences: int,
        left_window: int,
        right_window: int,
        negative_ratio: float = 0.0,
        max_examples: int = 0,
        seed: int = 42,
    ) -> None:
        self.items: List[Dict[str, Any]] = []
        self.doc_count = 0
        rng = random.Random(seed)

        for example in split:
            sentences = extract_sentences(example)
            labels = extract_labels(example)
            sentences, labels = align_sentences_and_labels(sentences, labels, max_sentences)
            if len(sentences) <= 1 or len(labels) <= 1:
                continue

            self.doc_count += 1
            positives: List[Dict[str, Any]] = []
            negatives: List[Dict[str, Any]] = []

            for boundary_index in range(1, len(sentences)):
                left_text, right_text = build_boundary_pair(
                    sentences=sentences,
                    boundary_index=boundary_index,
                    left_window=left_window,
                    right_window=right_window,
                )
                item = {"left_text": left_text, "right_text": right_text, "label": int(labels[boundary_index])}
                if item["label"] == 1:
                    positives.append(item)
                else:
                    negatives.append(item)

            self.items.extend(positives)
            if negative_ratio > 0 and positives:
                max_negatives = int(len(positives) * negative_ratio)
                if max_negatives > 0 and len(negatives) > max_negatives:
                    negatives = rng.sample(negatives, max_negatives)
            self.items.extend(negatives)

        if max_examples > 0 and len(self.items) > max_examples:
            self.items = rng.sample(self.items, max_examples)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class PairCollator:
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


def build_doc_windows(
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
            pairs.append(
                build_boundary_pair(
                    sentences=sentences,
                    boundary_index=boundary_index,
                    left_window=left_window,
                    right_window=right_window,
                )
            )
            targets.append(int(labels[boundary_index]))

        docs.append(
            {
                "doc_index": doc_index,
                "labels": [1] + targets,
                "pairs": pairs,
            }
        )
    return docs


def create_model(model_key: str, backbone_name: str, dropout: float, classifier_hidden: int):
    if model_key == "bert":
        model_config = BertBoundaryConfig(backbone_name=backbone_name, dropout=dropout, classifier_hidden=classifier_hidden)
        return BertBoundaryModel(model_config), model_config.to_dict()
    if model_key == "deberta":
        model_config = DebertaBoundaryConfig(
            backbone_name=backbone_name,
            dropout=dropout,
            classifier_hidden=classifier_hidden,
        )
        return DebertaBoundaryModel(model_config), model_config.to_dict()
    if model_key == "roberta":
        model_config = RobertaBoundaryConfig(
            backbone_name=backbone_name,
            dropout=dropout,
            classifier_hidden=classifier_hidden,
        )
        return RobertaBoundaryModel(model_config), model_config.to_dict()
    raise ValueError(f"Unsupported model key: {model_key}")


def estimate_pos_weight(train_loader: DataLoader) -> float:
    positives = 0.0
    negatives = 0.0
    for batch in train_loader:
        labels = batch["labels"]
        positives += float((labels == 1).sum().item())
        negatives += float((labels == 0).sum().item())
    if positives <= 0:
        return 1.0
    return max(1.0, min(50.0, negatives / positives))


def collect_doc_probabilities(
    model,
    tokenizer,
    docs: Sequence[Dict[str, Any]],
    max_tokens: int,
    batch_size: int,
    device: torch.device,
    desc: str,
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
                logits = model(**encoded)
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                doc_probs.extend(float(prob) for prob in probs)

            all_probs.append(doc_probs[: len(labels)])
            all_labels.append(labels[: len(doc_probs)])
        progress.close()
    return all_probs, all_labels


def run_validation(
    model,
    tokenizer,
    val_docs: Sequence[Dict[str, Any]],
    config: TrainConfig,
    device: torch.device,
    model_key: str,
) -> Dict[str, float]:
    probs_by_doc, labels_by_doc = collect_doc_probabilities(
        model=model,
        tokenizer=tokenizer,
        docs=val_docs,
        max_tokens=config.max_tokens,
        batch_size=config.eval_batch_size,
        device=device,
        desc=f"valid-{model_key}",
    )
    threshold, _ = tune_threshold(probs_by_doc, labels_by_doc, objective="f1")
    metrics = evaluate_segmentation(probs_by_doc, labels_by_doc, threshold=threshold)
    metrics["threshold"] = float(threshold)
    return metrics


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pos_weight: float,
    use_amp: bool,
    grad_clip: float,
    grad_accum_steps: int,
    model_key: str,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    pos_weight_tensor = torch.tensor(pos_weight, device=device)

    progress = tqdm(train_loader, desc=f"train-{model_key}", leave=False, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for batch in progress:
        if batch["input_ids"].numel() == 0:
            continue

        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")

        autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
        with autocast_ctx:
            logits = model(**batch)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_tensor)
            scaled_loss = loss / max(1, grad_accum_steps)

        scaler.scale(scaled_loss).backward()
        should_step = (steps + 1) % max(1, grad_accum_steps) == 0
        if should_step:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / max(steps, 1):.4f}")

    if steps > 0 and steps % max(1, grad_accum_steps) != 0:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    progress.close()
    return running_loss / max(steps, 1)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_single_model(model_key: str, config: TrainConfig) -> Dict[str, Any]:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from {sorted(MODEL_SPECS.keys())}.")

    spec = MODEL_SPECS[model_key]
    set_seed(config.seed)

    load_from_disk = import_load_from_disk()
    dataset = load_from_disk(config.dataset_dir)
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Expected a DatasetDict with 'train' and 'validation' splits.")

    tokenizer = AutoTokenizer.from_pretrained(spec.backbone_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.cls_token

    train_dataset = WindowPairDataset(
        split=dataset["train"],
        max_sentences=config.max_sentences,
        left_window=config.left_window,
        right_window=config.right_window,
        negative_ratio=config.negative_ratio,
        max_examples=config.max_train_examples,
        seed=config.seed,
    )
    val_docs = build_doc_windows(
        split=dataset["validation"],
        max_sentences=config.max_sentences,
        left_window=config.left_window,
        right_window=config.right_window,
    )
    collator = PairCollator(tokenizer=tokenizer, max_tokens=config.max_tokens)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model, model_config = create_model(
        model_key=model_key,
        backbone_name=spec.backbone_name,
        dropout=config.dropout,
        classifier_hidden=config.classifier_hidden,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    updates_per_epoch = max(1, (len(train_loader) + max(config.grad_accum_steps, 1) - 1) // max(config.grad_accum_steps, 1))
    total_steps = max(1, updates_per_epoch * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    pos_weight = estimate_pos_weight(train_loader)

    model_dir = Path(config.output_root) / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best.pt"
    last_path = model_dir / "last.pt"

    best_f1 = float("-inf")
    best_metrics: Optional[Dict[str, Any]] = None
    epoch_progress = tqdm(range(1, config.epochs + 1), desc=f"epochs-{model_key}", dynamic_ncols=True)
    for epoch in epoch_progress:
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            pos_weight=pos_weight,
            use_amp=use_amp,
            grad_clip=config.grad_clip,
            grad_accum_steps=config.grad_accum_steps,
            model_key=model_key,
        )
        metrics = run_validation(
            model=model,
            tokenizer=tokenizer,
            val_docs=val_docs,
            config=config,
            device=device,
            model_key=model_key,
        )

        epoch_progress.set_postfix(
            loss=f"{train_loss:.4f}",
            f1=f"{metrics['f1']:.4f}",
            pk=f"{metrics['pk']:.4f}",
            wd=f"{metrics['window_diff']:.4f}",
            thr=f"{metrics['threshold']:.2f}",
        )
        print(
            f"[{model_key}] Epoch {epoch:02d} | loss={train_loss:.4f} | f1={metrics['f1']:.4f} | "
            f"pk={metrics['pk']:.4f} | wd={metrics['window_diff']:.4f} | thr={metrics['threshold']:.2f}"
        )

        payload = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_key": model_key,
            "model_spec": asdict(spec),
            "model_config": model_config,
            "train_config": asdict(config),
            "metrics": metrics,
            "pos_weight": pos_weight,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        torch.save(payload, last_path)

        if metrics["f1"] > best_f1:
            best_f1 = float(metrics["f1"])
            best_metrics = dict(metrics)
            torch.save(payload, best_path)
            print(f"[{model_key}] Saved best checkpoint to {best_path}")

    epoch_progress.close()
    if best_metrics is None:
        raise RuntimeError(f"No metrics produced for model '{model_key}'.")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "model_key": model_key,
        "model_name": spec.backbone_name,
        "output_dir": str(model_dir),
        "num_train_windows": len(train_dataset),
        "num_train_docs": train_dataset.doc_count,
        "num_validation_docs": len(val_docs),
        "best_checkpoint": str(best_path),
        "best_metrics": best_metrics,
        "train_config": asdict(config),
        "model_config": model_config,
    }
    _write_json(model_dir / "summary.json", summary)
    return summary


def _compare_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float]:
    metrics = item.get("best_metrics", {})
    return (
        float(metrics.get("f1", float("-inf"))),
        -float(metrics.get("pk", float("inf"))),
        -float(metrics.get("window_diff", float("inf"))),
    )


def compare_models(
    model_keys: Sequence[str],
    config: TrainConfig,
    baseline_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    for model_key in model_keys:
        summaries.append(train_single_model(model_key=model_key, config=config))

    sorted_runs = sorted(summaries, key=_compare_sort_key, reverse=True)
    comparison = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": config.dataset_dir,
        "runs": sorted_runs,
    }
    if baseline_metrics:
        comparison["baseline"] = baseline_metrics

    comparison_dir = Path(config.output_root)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _write_json(comparison_dir / "comparison.json", comparison)
    _write_markdown_table(comparison_dir / "comparison.md", comparison)
    return comparison


def _write_markdown_table(path: Path, comparison: Dict[str, Any]) -> None:
    rows = [
        "| Rank | Model | F1 | Precision | Recall | Pk (lower better) | WindowDiff (lower better) | Threshold |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    runs = comparison.get("runs", [])
    for idx, run in enumerate(runs, start=1):
        metrics = run.get("best_metrics", {})
        rows.append(
            "| "
            + f"{idx} | {run.get('model_key', '-')}/{run.get('model_name', '-')} | "
            + f"{metrics.get('f1', 0.0):.4f} | {metrics.get('precision', 0.0):.4f} | {metrics.get('recall', 0.0):.4f} | "
            + f"{metrics.get('pk', 0.0):.4f} | {metrics.get('window_diff', 0.0):.4f} | {metrics.get('threshold', 0.0):.2f} |"
        )

    baseline = comparison.get("baseline")
    if baseline:
        metrics = baseline.get("metrics", baseline)
        rows.append(
            "| - | baseline/your-model | "
            + f"{metrics.get('f1', 0.0):.4f} | {metrics.get('precision', 0.0):.4f} | {metrics.get('recall', 0.0):.4f} | "
            + f"{metrics.get('pk', 0.0):.4f} | {metrics.get('window_diff', 0.0):.4f} | {metrics.get('threshold', 0.0):.2f} |"
        )

    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
