#!/usr/bin/env python3
"""Fine-tune BLIP-2 with LoRA for mineral spectral VQA.

Uses spectral plot images + QA pairs from the generation pipeline.
Trains only the Q-Former via LoRA for efficient fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    BASE_MODEL, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, MAX_LENGTH,
    GRAD_ACCUM_STEPS, MODEL_DIR, SPECTRAL_IMAGES_DIR, VQA_DATASET_PATH,
    SEED, TRAIN_RATIO, VAL_RATIO, NPZ_PATH,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MineralVQADataset(Dataset):
    """Dataset for mineral spectral VQA."""

    def __init__(
        self,
        qa_pairs: list[dict],
        image_dir: Path,
        processor: Blip2Processor,
        max_length: int = MAX_LENGTH,
    ):
        self.qa_pairs = qa_pairs
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        image_path = self.image_dir / qa["image"]
        image = Image.open(image_path).convert("RGB")

        question = qa["question"]
        answer = qa["answer"]

        # process image + question
        encoding = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # process answer as labels
        labels = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # squeeze batch dim
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = labels["input_ids"].squeeze(0)

        # mask padding in labels
        item["labels"][item["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return item


# ---------------------------------------------------------------------------
# Observation-wise split
# ---------------------------------------------------------------------------

def split_qa_obs_wise(
    qa_pairs: list[dict],
    manifest_path: Path,
    npz_path: Path,
    seed: int = SEED,
) -> tuple[list, list, list]:
    """Split QA pairs by observation to prevent scene leakage."""
    # load manifest to get obs_idx per image
    with open(manifest_path) as f:
        manifest = json.load(f)

    image_to_obs = {m["image"]: m["obs_idx"] for m in manifest}

    # load npz to get obs_ids
    data = np.load(npz_path, allow_pickle=True)
    obs_ids = data.get("obs_ids", np.arange(100))
    n_obs = len(np.unique(list(image_to_obs.values())))

    # group QA pairs by observation
    obs_to_qa = {}
    for qa in qa_pairs:
        obs = image_to_obs.get(qa["image"], -1)
        obs_to_qa.setdefault(obs, []).append(qa)

    # shuffle and split observations
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
    use_auto_device = torch.cuda.is_available()
    device = torch.device("cuda" if use_auto_device else "cpu")
    print(f"Device: {device}")

    # load QA pairs
    print(f"Loading QA pairs from {args.qa_path} ...")
    with open(args.qa_path) as f:
        all_qa = json.load(f)
    print(f"  Total QA pairs: {len(all_qa)}")

    # split
    print("Splitting by observation ...")
    manifest_path = Path(args.image_dir) / "manifest.json"
    train_qa, val_qa, test_qa = split_qa_obs_wise(
        all_qa, manifest_path, Path(args.npz), args.seed,
    )
    print(f"  Train: {len(train_qa)}, Val: {len(val_qa)}, Test: {len(test_qa)}")

    # save test set for later evaluation
    test_path = Path(args.qa_path).parent / "vqa_test_set.json"
    with open(test_path, "w") as f:
        json.dump(test_qa, f, indent=2, ensure_ascii=False)
    print(f"  Test set saved to {test_path}")

    # load model + processor
    print(f"Loading {args.model_name} ...")
    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto" if use_auto_device else None,
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

    # when using device_map="auto", tensors are already placed correctly
    # only manually move to device when not using auto mapping
    if not use_auto_device:
        model = model.to(device)

    # datasets
    image_dir = Path(args.image_dir)
    train_ds = MineralVQADataset(train_qa, image_dir, processor, args.max_length)
    val_ds = MineralVQADataset(val_qa, image_dir, processor, args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps,
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
            # with device_map="auto", move to the model's first device
            target_device = next(model.parameters()).device
            batch = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

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

        avg_train_loss = train_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                target_device = next(model.parameters()).device
                batch = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}"
        )

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(save_dir / "best")
            processor.save_pretrained(save_dir / "best")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    # save final
    model.save_pretrained(save_dir / "final")
    processor.save_pretrained(save_dir / "final")
    print(f"Training complete. Models saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train VQA model for mineral classification")
    parser.add_argument("--model-name", type=str, default=BASE_MODEL)
    parser.add_argument("--qa-path", type=str, default=str(VQA_DATASET_PATH))
    parser.add_argument("--image-dir", type=str, default=str(SPECTRAL_IMAGES_DIR))
    parser.add_argument("--npz", type=str, default=str(NPZ_PATH))
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
