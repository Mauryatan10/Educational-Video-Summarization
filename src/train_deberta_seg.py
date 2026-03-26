from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from deberta_seg_core import (
    DebertaSegConfig,
    WindowBoundaryCollator,
    WindowBoundaryDataset,
    build_doc_level_windows,
    collect_doc_probabilities,
    import_load_from_disk,
    save_deberta_checkpoint,
)
from miniseg_core import evaluate_segmentation, set_seed, tune_threshold


def build_dataloaders(
    dataset_dir: str,
    tokenizer,
    config: DebertaSegConfig,
    batch_size: int,
    num_workers: int,
):
    load_from_disk = import_load_from_disk()
    dataset = load_from_disk(dataset_dir)
    if "train" not in dataset or "validation" not in dataset:
        raise ValueError("Expected a DatasetDict with 'train' and 'validation' splits.")

    train_dataset = WindowBoundaryDataset(
        split=dataset["train"],
        max_sentences=config.max_sentences,
        left_window=config.left_window,
        right_window=config.right_window,
    )
    val_docs = build_doc_level_windows(
        split=dataset["validation"],
        max_sentences=config.max_sentences,
        left_window=config.left_window,
        right_window=config.right_window,
    )
    collator = WindowBoundaryCollator(tokenizer=tokenizer, max_tokens=config.max_tokens)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_docs


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
) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    pos_weight_tensor = torch.tensor(pos_weight, device=device)

    progress = tqdm(train_loader, desc="train", leave=False, dynamic_ncols=True)
    for batch in progress:
        if batch["input_ids"].numel() == 0:
            continue

        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
        with autocast_ctx:
            logits = model(**batch).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight_tensor)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{running_loss / max(steps, 1):.4f}")

    progress.close()
    return running_loss / max(steps, 1)


def run_validation(
    model,
    tokenizer,
    val_docs,
    config: DebertaSegConfig,
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, float]:
    probs_by_doc, labels_by_doc = collect_doc_probabilities(
        model=model,
        tokenizer=tokenizer,
        docs=val_docs,
        max_tokens=config.max_tokens,
        batch_size=eval_batch_size,
        device=device,
        desc="valid",
    )
    threshold, _ = tune_threshold(probs_by_doc, labels_by_doc, objective="f1")
    metrics = evaluate_segmentation(probs_by_doc, labels_by_doc, threshold)
    metrics["threshold"] = threshold
    return metrics


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train DeBERTa windowed boundary classifier.")
    parser.add_argument("--dataset-dir", type=str, default=str(repo_root / "datasets" / "ytseg-dataset"))
    parser.add_argument("--output-dir", type=str, default=str(repo_root / "models" / "deberta-seg"))
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-sentences", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--left-window", type=int, default=3)
    parser.add_argument("--right-window", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DebertaSegConfig(
        model_name=args.model_name,
        max_sentences=args.max_sentences,
        max_tokens=args.max_tokens,
        left_window=args.left_window,
        right_window=args.right_window,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_loader, val_docs = build_dataloaders(
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=1,
        hidden_dropout_prob=config.dropout,
        attention_probs_dropout_prob=config.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    pos_weight = estimate_pos_weight(train_loader)
    print(f"Estimated positive class weight: {pos_weight:.3f}")

    best_f1 = float("-inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    epoch_progress = tqdm(range(1, args.epochs + 1), desc="epochs", dynamic_ncols=True)
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
            grad_clip=args.grad_clip,
        )

        metrics = run_validation(
            model=model,
            tokenizer=tokenizer,
            val_docs=val_docs,
            config=config,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )

        epoch_progress.set_postfix(
            loss=f"{train_loss:.4f}",
            f1=f"{metrics['f1']:.4f}",
            pk=f"{metrics['pk']:.4f}",
            wd=f"{metrics['window_diff']:.4f}",
            thr=f"{metrics['threshold']:.2f}",
        )
        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
            f"f1={metrics['f1']:.4f} | pk={metrics['pk']:.4f} | wd={metrics['window_diff']:.4f} | "
            f"threshold={metrics['threshold']:.2f}"
        )

        payload = {
            "epoch": epoch,
            "config": config.__dict__,
            "threshold": metrics["threshold"],
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        torch.save(payload, last_path)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_deberta_checkpoint(best_path, model, config, metrics["threshold"], metrics)
            print(f"Saved new best checkpoint to {best_path}")

    print(f"Training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

