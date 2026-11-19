#!/usr/bin/env python3
"""
scripts/train_subtopic_script_constrained.py

Fine-tunes the foundation model `sarvamai/sarvam-1` (2.5B, 8,192 context window)
using 4-bit QLoRA with Script-Constrained Cross-Entropy Loss on the diversified
10-15 sentence subtopic paragraph corpus.

Key Architecture & Loss:
  - Base Model: sarvamai/sarvam-1 (NF4 4-bit quantization with double quant)
  - LoRA Targets: all 7 linear projections (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
  - LoRA Rank: r=16, alpha=32, dropout=0.05
  - Training Data: data/subtopic_diversified/train_subtopics_6k.parquet (200-320 words, 10-15 sentences)
  - Script-Constrained Loss:
      L_total = L_CE + lambda_script * L_forbidden
      where L_forbidden = mean_{tokens} sum_{k in V_forbidden} softmax(z)_k
      V_forbidden: All 37,581 tokens belonging to non-target Indic scripts
      (Gurmukhi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Oriya).
      Devanagari, Latin, digits, and punctuation are completely unpenalized.
  - Memory Optimization: PagedAdamW8bit + gradient checkpointing + FP16 autocast (< 4.5 GB VRAM)
  - Output: models/sarvam1_subtopic_script_constrained/
"""

import os
import sys
import time
import json
import math
from typing import List, Dict, Any, Optional, Tuple

# Strip local dir from sys.path to avoid shadowing third-party 'datasets'
sys.path = [p for p in sys.path if p not in ('', '.', os.path.dirname(os.path.abspath(__file__)))]

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb

MODEL_ID = "sarvamai/sarvam-1"
TRAIN_PARQUET = "data/subtopic_diversified/train_subtopics_6k.parquet"
VAL_PARQUET = "data/subtopic_diversified/val_subtopics.parquet"
OUTPUT_DIR = "models/sarvam1_subtopic_script_constrained"

PROMPT_PREFIX = "अंग्रेजी:\n"
PROMPT_SUFFIX = "\n\nहिन्दी (देवनागरी लिपि में अनुवाद):\n"


def build_forbidden_vocab_mask(tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Identifies all token IDs in sarvam-1 whose string representation contains
    Unicode code points from non-target Indic scripts.
    Preserves Devanagari (0x0900-0x097F), Latin/ASCII, digits, and common punctuation.
    """
    forbidden_ids = []
    for token, tid in tokenizer.get_vocab().items():
        is_forbidden = False
        for c in token:
            cp = ord(c)
            if (
                0x0A00 <= cp <= 0x0A7F or  # Gurmukhi
                0x0B80 <= cp <= 0x0BFF or  # Tamil
                0x0C00 <= cp <= 0x0C7F or  # Telugu
                0x0C80 <= cp <= 0x0CFF or  # Kannada
                0x0D00 <= cp <= 0x0D7F or  # Malayalam
                0x0980 <= cp <= 0x09FF or  # Bengali / Assamese
                0x0A80 <= cp <= 0x0AFF or  # Gujarati
                0x0B00 <= cp <= 0x0B7F     # Oriya
            ):
                is_forbidden = True
                break
        if is_forbidden:
            forbidden_ids.append(tid)

    print(f"Total vocabulary size: {len(tokenizer):,}")
    print(f"Constructed V_forbidden partition: {len(forbidden_ids):,} non-target Indic tokens ({len(forbidden_ids)/len(tokenizer)*100:.2f}%)")
    return torch.tensor(forbidden_ids, dtype=torch.long)


class DiversifiedSubtopicDataset(Dataset):
    """
    Subtopic translation dataset loading 10-15 sentence discourse blocks
    into Devanagari-anchored instruction prompts with target-only loss masking.
    """
    def __init__(
        self,
        parquet_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1536
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading diversified subtopics from {parquet_path}...")
        table = pq.read_table(parquet_path)
        self.samples = table.to_pylist()
        print(f"Loaded {len(self.samples):,} subtopics.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        src_en = item["src_en"].strip()
        tgt_hi = item["tgt_hi"].strip()

        # Format:
        # अंग्रेजी:
        # {src_en}
        #
        # हिन्दी (देवनागरी लिपि में अनुवाद):
        # {tgt_hi}</s>
        prompt = f"{PROMPT_PREFIX}{src_en}{PROMPT_SUFFIX}"
        full_text = f"{prompt}{tgt_hi}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Mask prompt tokens with -100 so loss is strictly evaluated on Hindi target
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "subtopic_id": item.get("subtopic_id", f"sample_{idx}")
        }


def collate_fn(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pads input_ids with pad_token_id and labels with -100."""
    max_len = max(len(b["input_ids"]) for b in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []

    for b in batch:
        inp = b["input_ids"]
        lbl = b["labels"]
        pad_len = max_len - len(inp)

        if pad_len > 0:
            pad_inp = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_lbl = torch.full((pad_len,), -100, dtype=torch.long)
            mask = torch.cat([torch.ones(len(inp), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            full_inp = torch.cat([inp, pad_inp])
            full_lbl = torch.cat([lbl, pad_lbl])
        else:
            mask = torch.ones(len(inp), dtype=torch.long)
            full_inp = inp
            full_lbl = lbl

        batch_input_ids.append(full_inp)
        batch_labels.append(full_lbl)
        batch_attention_mask.append(mask)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask)
    }


def compute_script_constrained_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    forbidden_mask: torch.Tensor,
    lambda_script: float = 8.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes total loss:
      L_total = L_CE + lambda_script * L_forbidden
    Where L_forbidden is the mean probability mass assigned to V_forbidden across target tokens.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    active_mask = (shift_labels != -100)
    if not active_mask.any():
        zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return zero, zero, zero

    active_logits = shift_logits[active_mask]
    active_labels = shift_labels[active_mask]

    # 1. Standard Target Cross-Entropy Loss
    loss_ce = F.cross_entropy(active_logits, active_labels)

    # 2. Softmax Probability Mass Penalty on Forbidden Indic Scripts
    probs = F.softmax(active_logits, dim=-1)
    loss_forbidden = probs[:, forbidden_mask].sum(dim=-1).mean()

    total_loss = loss_ce + lambda_script * loss_forbidden
    return total_loss, loss_ce, loss_forbidden


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    forbidden_mask: torch.Tensor,
    lambda_script: float,
    device: str
) -> Tuple[float, float, float]:
    """Evaluates CE loss, forbidden probability, and total loss on validation subtopics."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_forbid = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, loss_ce, loss_forbid = compute_script_constrained_loss(
                    outputs.logits, labels, forbidden_mask, lambda_script=lambda_script
                )

            num_valid = (labels != -100).sum().item()
            total_loss += loss.item() * num_valid
            total_ce += loss_ce.item() * num_valid
            total_forbid += loss_forbid.item() * num_valid
            total_tokens += num_valid

    model.train()
    denom = max(1, total_tokens)
    return total_loss / denom, total_ce / denom, total_forbid / denom


def train(resume_from_checkpoint: Optional[str] = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("=== STARTING SCRIPT-CONSTRAINED QLORA TRAINING FOR SARVAM-1 ===")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Output Directory: {OUTPUT_DIR}")
    if resume_from_checkpoint:
        print(f"Resume Source: {resume_from_checkpoint}")

    # Hyperparameters
    MAX_LENGTH = 1536
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 16  # Effective batch size = 16 subtopics
    NUM_EPOCHS = 2
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05
    EVAL_EVERY_STEPS = 50
    SAVE_EVERY_STEPS = 100
    LAMBDA_SCRIPT = 8.0  # Weight for forbidden script probability mass penalty

    # 1. Load Tokenizer & Construct Forbidden Mask
    print(f"\nLoading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forbidden_mask = build_forbidden_vocab_mask(tokenizer).to(device)

    # 2. Prepare Datasets & DataLoaders
    train_dataset = DiversifiedSubtopicDataset(
        TRAIN_PARQUET,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    val_dataset = DiversifiedSubtopicDataset(
        VAL_PARQUET,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True
    )

    # 3. Load 4-Bit NF4 Base Model
    print(f"\nLoading {MODEL_ID} in 4-bit NF4 precision with BitsAndBytes...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0"
    )

    # 4. Prepare for k-bit training & activate gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    start_step = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Loading existing LoRA weights from {resume_from_checkpoint} with is_trainable=True...")
        peft_model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)
        # Parse step number if present in directory name
        base_name = os.path.basename(os.path.normpath(resume_from_checkpoint))
        if "step-" in base_name:
            try:
                start_step = int(base_name.split("step-")[-1])
            except ValueError:
                start_step = 0
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)

    peft_model.train()
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.config.use_cache = False

    print("\n--- Trainable LoRA Parameters ---")
    peft_model.print_trainable_parameters()

    # 5. Optimizer & Learning Rate Scheduler
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"Total Optimization Steps: {total_steps} | Warmup Steps: {warmup_steps} | Resuming from Step: {start_step}")

    optimizer = bnb.optim.PagedAdamW8bit(
        peft_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Fast-forward scheduler if resuming
    if start_step > 0:
        for _ in range(start_step):
            scheduler.step()
        print(f"Scheduler advanced to step {start_step}. Current LR: {scheduler.get_last_lr()[0]:.2e}")

    # Initial Validation Loss
    init_total, init_ce, init_forbid = evaluate(peft_model, val_loader, forbidden_mask, LAMBDA_SCRIPT, device)
    init_ppl = math.exp(min(20, init_ce))
    print(
        f"[Resume Validation at Step {start_step}] Total Loss: {init_total:.4f} | "
        f"CE Loss: {init_ce:.4f} (PPL: {init_ppl:.2f}) | "
        f"Forbidden P(V_forbid): {init_forbid*100:.2f}%"
    )

    # 6. Training Loop
    global_step = start_step
    # If resuming, load best_val_loss baseline from Step 200 (1.6766) or current eval
    best_val_loss = min(1.6766, init_total)
    training_history = []
    t_start = time.time()
    accum_loss = 0.0
    accum_ce = 0.0
    accum_forbid = 0.0

    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    start_epoch = (start_step // steps_per_epoch) + 1
    skip_batches_in_start_epoch = (start_step % steps_per_epoch) * GRAD_ACCUM_STEPS

    print(f"\n=== Commencing Script-Constrained QLoRA Optimization from Epoch {start_epoch} (skipping first {skip_batches_in_start_epoch} batches) ===")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch} / {NUM_EPOCHS} ---")
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < skip_batches_in_start_epoch:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
                loss, loss_ce, loss_forbid = compute_script_constrained_loss(
                    outputs.logits, labels, forbidden_mask, lambda_script=LAMBDA_SCRIPT
                )
                scaled_loss = loss / GRAD_ACCUM_STEPS

            scaled_loss.backward()
            accum_loss += loss.item()
            accum_ce += loss_ce.item()
            accum_forbid += loss_forbid.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                step_loss = accum_loss / GRAD_ACCUM_STEPS
                step_ce = accum_ce / GRAD_ACCUM_STEPS
                step_forbid = accum_forbid / GRAD_ACCUM_STEPS

                # Logging
                if global_step % 20 == 0 or global_step == 1:
                    current_lr = scheduler.get_last_lr()[0]
                    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    print(
                        f"Step {global_step:04d}/{total_steps} | "
                        f"Loss: {step_loss:.4f} (CE: {step_ce:.4f}) | "
                        f"P(V_forbid): {step_forbid*100:.3f}% | "
                        f"LR: {current_lr:.2e} | "
                        f"VRAM: {vram_mb:.1f} MB | "
                        f"Elapsed: {time.time() - t_start:.1f}s"
                    )

                training_history.append({
                    "step": global_step,
                    "epoch": round(epoch - 1 + (step + 1) / len(train_loader), 3),
                    "total_loss": round(step_loss, 4),
                    "ce_loss": round(step_ce, 4),
                    "forbidden_prob": round(step_forbid, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                accum_loss = 0.0
                accum_ce = 0.0
                accum_forbid = 0.0

                # Validation & Checkpointing
                if global_step % EVAL_EVERY_STEPS == 0:
                    val_tot, val_ce, val_forb = evaluate(peft_model, val_loader, forbidden_mask, LAMBDA_SCRIPT, device)
                    val_ppl = math.exp(min(20, val_ce))
                    print(
                        f"--> [Validation at Step {global_step:04d}] "
                        f"Total Loss: {val_tot:.4f} | CE: {val_ce:.4f} (PPL: {val_ppl:.2f}) | "
                        f"P(V_forbid): {val_forb*100:.3f}%"
                    )

                    if val_tot < best_val_loss:
                        print(f"    * New best validation loss! ({best_val_loss:.4f} -> {val_tot:.4f}). Saving checkpoint...")
                        best_val_loss = val_tot
                        best_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
                        peft_model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

                if global_step % SAVE_EVERY_STEPS == 0:
                    step_dir = os.path.join(OUTPUT_DIR, f"checkpoint-step-{global_step}")
                    peft_model.save_pretrained(step_dir)
                    tokenizer.save_pretrained(step_dir)
                    print(f"    [Checkpoint] Saved periodic adapter weights to {step_dir}")

        epoch_time = time.time() - epoch_start
        print(f"Completed Epoch {epoch} in {epoch_time:.1f}s ({epoch_time/60:.2f} mins).")

    # 7. Final Model Save
    print("\n=== Fine-Tuning Complete ===")
    final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    peft_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final QLoRA adapter saved to {final_dir}")

    # Save training metrics log
    log_file = os.path.join(OUTPUT_DIR, "training_log.json")
    summary = {
        "model_id": MODEL_ID,
        "method": "Script-Constrained QLoRA (4-bit NF4)",
        "lambda_script": LAMBDA_SCRIPT,
        "r": 16,
        "alpha": 32,
        "total_steps": global_step,
        "initial_val_loss": round(init_total, 4),
        "best_val_loss": round(best_val_loss, 4),
        "training_time_sec": round(time.time() - t_start, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
        "history": training_history
    }
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training log and metrics written to {log_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g. models/sarvam1_subtopic_script_constrained/checkpoint-step-400)"
    )
    args = parser.parse_args()
    train(resume_from_checkpoint=args.resume_from_checkpoint)
