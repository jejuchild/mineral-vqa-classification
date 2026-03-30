#!/usr/bin/env python3
"""Fine-tune Qwen2.5-1.5B-Instruct with LoRA for mineral spectral VQA.

Text-only approach: spectrum is encoded as structured text, no images needed.
Uses observation-wise split to prevent scene leakage.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    BASE_MODEL, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, MAX_LENGTH,
    GRAD_ACCUM_STEPS, MODEL_DIR, VQA_DATASET_PATH,
    SEED, TRAIN_RATIO, VAL_RATIO, NPZ_PATH,
)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Mars mineral spectroscopy expert. "
    "Given a CRISM spectrum (continuum-removed, 1.02-3.92um), "
    "answer questions about mineral identification, absorption features, "
    "and spectral characteristics."
)


def format_prompt(spectrum_text: str, question: str) -> str:
    """Format input as chat template."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{spectrum_text}\n\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_prompt_with_answer(spectrum_text: str, question: str, answer: str) -> str:
    """Format full training example."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{spectrum_text}\n\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MineralQADataset(Dataset):
    """Dataset for mineral spectral QA (text-only)."""

    def __init__(
        self,
        qa_pairs: list[dict],
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        full_text = format_prompt_with_answer(
            qa["spectrum"], qa["question"], qa["answer"],
        )

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # create labels: mask everything before "assistant\n"
        labels = input_ids.clone()

        # find the start of the answer
        prompt_text = format_prompt(qa["spectrum"], qa["question"])
        prompt_tokens = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length,
        )
        prompt_len = len(prompt_tokens["input_ids"])

        # mask prompt tokens in labels (only train on answer)
        labels[:prompt_len] = -100
        # mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Observation-wise split
# ---------------------------------------------------------------------------

def split_qa_obs_wise(
    qa_pairs: list[dict],
    seed: int = SEED,
) -> tuple[list, list, list]:
    """Split QA pairs by observation to prevent scene leakage."""
    obs_to_qa = {}
    for qa in qa_pairs:
        obs = qa.get("obs_idx", -1)
        obs_to_qa.setdefault(obs, []).append(qa)

    obs_list = sorted(obs_to_qa.keys())
    rng = random.Random(seed)
    rng.shuffle(obs_list)

    n_train = int(len(obs_list) * TRAIN_RATIO)
    n_val = int(len(obs_list) * VAL_RATIO)

    train_obs = set(obs_list[:n_train])
    val_obs = set(obs_list[n_train:n_train + n_val])
    test_obs = set(obs_list[n_train + n_val:])

    train_qa = [qa for obs in train_obs for qa in obs_to_qa.get(obs, [])]
    val_qa = [qa for obs in val_obs for qa in obs_to_qa.get(obs, [])]
    test_qa = [qa for obs in test_obs for qa in obs_to_qa.get(obs, [])]

    rng.shuffle(train_qa)
    rng.shuffle(val_qa)

    return train_qa, val_qa, test_qa


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    # load QA pairs
    print(f"Loading QA pairs from {args.qa_path} ...")
    with open(args.qa_path) as f:
        all_qa = json.load(f)
    print(f"  Total QA pairs: {len(all_qa)}")

    # split
    print("Splitting by observation ...")
    train_qa, val_qa, test_qa = split_qa_obs_wise(all_qa, args.seed)
    print(f"  Train: {len(train_qa)}, Val: {len(val_qa)}, Test: {len(test_qa)}")

    # save test set
    test_path = Path(args.qa_path).parent / "vqa_test_set.json"
    with open(test_path, "w") as f:
        json.dump(test_qa, f, indent=2, ensure_ascii=False)
    print(f"  Test set saved to {test_path}")

    # load model + tokenizer
    print(f"Loading {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )

    # apply LoRA
    print("Applying LoRA ...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if not use_cuda:
        model = model.to(device)

    # datasets
    train_ds = MineralQADataset(train_qa, tokenizer, args.max_length)
    val_ds = MineralQADataset(val_qa, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=use_cuda,
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, max(total_steps, 1),
    )

    # training loop
    best_val_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            target_device = next(model.parameters()).device
            batch = {k: v.to(target_device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += outputs.loss.item()
            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        # handle remaining gradient accumulation
        if (step + 1) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                target_device = next(model.parameters()).device
                batch = {k: v.to(target_device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(save_dir / "best")
            tokenizer.save_pretrained(save_dir / "best")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    # save final
    model.save_pretrained(save_dir / "final")
    tokenizer.save_pretrained(save_dir / "final")
    print(f"Training complete. Models saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train spectral QA model")
    parser.add_argument("--model-name", type=str, default=BASE_MODEL)
    parser.add_argument("--qa-path", type=str, default=str(VQA_DATASET_PATH))
    parser.add_argument("--save-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--lora-targets", nargs="+", default=LORA_TARGET_MODULES)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
