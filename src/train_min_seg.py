from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from miniseg_core import (
    BoundaryCollator,
    BoundaryDataset,
    DEFAULT_ENCODER_NAME,
    MiniSegConfig,
    MiniSegModel,
    collect_predictions,
    evaluate_segmentation,
    save_checkpoint,
    set_seed,
    tune_threshold,
)


def _import_load_from_disk():
    """
    The repo has a top-level `datasets/` folder, so we defensively remove the repo
    root from sys.path before importing the Hugging Face package.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))

    try:
        from datasets import load_from_disk
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "The `datasets` package is required to train MiniSeg. Install it with `pip install datasets`."
        ) from exc
    return load_from_disk


def _build_dataloaders(
    dataset_dir: str,
    tokenizer,
    max_sentences: int,
    max_tokens: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    load_from_disk = _import_load_from_disk()
    dataset = load_from_disk(dataset_dir)

    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Expected a DatasetDict with 'train' and 'validation' splits.")

    train_dataset = BoundaryDataset(dataset["train"], max_sentences=max_sentences)
    val_dataset = BoundaryDataset(dataset["validation"], max_sentences=max_sentences)

    collator = BoundaryCollator(tokenizer=tokenizer, max_tokens=max_tokens)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def _estimate_pos_weight(train_loader: DataLoader, device: torch.device) -> float:
    pos = 0.0
    neg = 0.0
    for batch in train_loader:
        if batch["labels"].numel() == 0:
            continue
        mask = batch["sentence_mask"]
        labels = batch["labels"][mask]
        pos += float((labels == 1).sum().item())
        neg += float((labels == 0).sum().item())
    if pos <= 0:
        return 1.0
    return max(1.0, min(30.0, neg / pos))


def _build_model(config: MiniSegConfig) -> MiniSegModel:
    return MiniSegModel(
        encoder_name=config.encoder_name,
        max_sentences=config.max_sentences,
        context_layers=config.context_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        freeze_sentence_encoder=config.freeze_sentence_encoder,
    )


def _train_one_epoch(
    model: MiniSegModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    pos_weight: float,
    use_amp: bool,
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    pos_weight_tensor = torch.tensor(pos_weight, device=device)

    for batch in train_loader:
        if batch["input_ids"].numel() == 0:
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentence_mask = batch["sentence_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
        with autocast_ctx:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_mask=sentence_mask,
            )
            valid_logits = logits[sentence_mask]
            valid_labels = labels[sentence_mask]
            loss = F.binary_cross_entropy_with_logits(
                valid_logits,
                valid_labels,
                pos_weight=pos_weight_tensor,
            )

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += float(loss.item())
        steps += 1

    return running_loss / max(steps, 1)


def _run_validation(
    model: MiniSegModel,
    val_loader: DataLoader,
    device: torch.device,
    threshold_objective: str,
) -> Tuple[float, Dict[str, float], List[List[float]], List[List[int]]]:
    probs_by_doc, labels_by_doc = collect_predictions(model, val_loader, device)
    threshold, _ = tune_threshold(probs_by_doc, labels_by_doc, objective=threshold_objective)
    metrics = evaluate_segmentation(probs_by_doc, labels_by_doc, threshold)
    return threshold, metrics, probs_by_doc, labels_by_doc


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train the MiniSeg topic segmentation model.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(repo_root / "datasets" / "ytseg-dataset"),
        help="Path to the saved yt-seg dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "models" / "miniseg"),
        help="Directory for checkpoints and metadata.",
    )
    parser.add_argument(
        "--encoder-name",
        type=str,
        default=DEFAULT_ENCODER_NAME,
        help="Sentence encoder backbone.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sentences", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-sentence-encoder", action="store_true")
    parser.add_argument("--best-metric", type=str, default="f1", choices=["f1", "window_diff", "pk"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MiniSegConfig(
        encoder_name=args.encoder_name,
        max_sentences=args.max_sentences,
        max_tokens=args.max_tokens,
        context_layers=args.context_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_sentence_encoder=args.freeze_sentence_encoder,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    train_loader, val_loader = _build_dataloaders(
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        max_sentences=config.max_sentences,
        max_tokens=config.max_tokens,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = _build_model(config).to(device)

    if config.freeze_sentence_encoder:
        encoder_params = []
    else:
        encoder_params = list(model.sentence_encoder.parameters())

    head_params = [
        *model.pos_embedding.parameters(),
        *model.context_encoder.parameters(),
        *model.classifier.parameters(),
    ]
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config.lr_encoder})
    param_groups.append({"params": head_params, "lr": config.lr_head})

    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    pos_weight = _estimate_pos_weight(train_loader, device)
    print(f"Estimated positive class weight: {pos_weight:.3f}")

    best_score = float("-inf")
    best_checkpoint = output_dir / "best.pt"
    last_checkpoint = output_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pos_weight=pos_weight,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
        )

        threshold, metrics, _, _ = _run_validation(
            model,
            val_loader,
            device,
            threshold_objective=args.best_metric,
        )
        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
            f"f1={metrics['f1']:.4f} | pk={metrics['pk']:.4f} | wd={metrics['window_diff']:.4f} | "
            f"threshold={threshold:.2f}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "config": config.__dict__,
            "threshold": threshold,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint_payload, last_checkpoint)

        score = metrics[args.best_metric]
        if args.best_metric == "window_diff":
            score = -score
        if args.best_metric == "pk":
            score = -score

        if score > best_score:
            best_score = score
            save_checkpoint(best_checkpoint, model, config, threshold, metrics)
            print(f"Saved new best checkpoint to {best_checkpoint}")

    print(f"Training complete. Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
