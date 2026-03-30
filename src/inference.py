#!/usr/bin/env python3
"""CLI inference tool for mineral spectral QA.

Supports:
  1. Single spectrum from .npy file → encode → ask questions
  2. Raw text spectrum input
  3. Interactive Q&A mode
  4. Batch inference from npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import (
    BANDS, CLASS_NAME, MODEL_DIR, MINERAL_KB_PATH,
    ABSORPTION_BANDS_PATH, BASE_MODEL, MAX_LENGTH,
)
from spectrum_encoder import encode_spectrum_from_raw, encode_spectrum


# ---------------------------------------------------------------------------
# Prompt (matching train_vqa.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Mars mineral spectroscopy expert. "
    "Given a CRISM spectrum (continuum-removed, 1.02-3.92um), "
    "answer questions about mineral identification, absorption features, "
    "and spectral characteristics."
)


def format_prompt(spectrum_text: str, question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{spectrum_text}\n\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: str, base_model: str = BASE_MODEL):
    """Load fine-tuned model with LoRA adapter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    print(f"Loading tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"Loading base model {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {model_dir} ...")
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    if not use_cuda:
        model = model.to(device)

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def ask(
    model,
    tokenizer,
    spectrum_text: str,
    question: str,
    max_new_tokens: int = 200,
) -> str:
    """Ask a question about a spectrum."""
    prompt = format_prompt(spectrum_text, question)
    target_device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(target_device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # decode only the new tokens (after prompt)
    prompt_len = inputs["input_ids"].shape[1]
    answer_ids = generated[0][prompt_len:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer


def interactive_mode(model, tokenizer, spectrum_text: str):
    """Interactive Q&A loop."""
    print("\n--- Interactive Mineral QA Mode ---")
    print("Type questions about the spectrum. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = ask(model, tokenizer, spectrum_text, question)
        print(f"A: {answer}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mineral Spectral QA Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single spectrum (.npy)
  python inference.py --spectrum raw_spec.npy -q "What mineral is this?"

  # From npz with index
  python inference.py --npz data.npz --index 42 -q "What mineral is this?"

  # Interactive mode
  python inference.py --spectrum raw_spec.npy --interactive

  # Batch inference
  python inference.py --npz data.npz --batch-indices 0,1,2,3,4
        """,
    )
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR / "best"))
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--spectrum", type=str, help="Path to .npy spectrum (350 bands)")
    parser.add_argument("--npz", type=str, help="Path to .npz dataset")
    parser.add_argument("--index", type=int, help="Pixel index in npz")
    parser.add_argument("--batch-indices", type=str, help="Comma-separated indices")
    parser.add_argument("-q", "--question", type=str, help="Question to ask")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--show-encoding", action="store_true", help="Print spectrum encoding")
    args = parser.parse_args()

    if not args.spectrum and not args.npz:
        parser.error("Provide either --spectrum or --npz")

    # load model
    model, tokenizer, device = load_model(args.model_dir, args.base_model)

    # prepare spectrum
    if args.spectrum:
        print(f"Loading spectrum from {args.spectrum} ...")
        raw = np.load(args.spectrum).astype(np.float32)
        if raw.ndim > 1:
            raw = raw.flatten()[:350]
        spectrum_text = encode_spectrum_from_raw(raw)
    elif args.npz:
        data = np.load(args.npz, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"]

        if args.batch_indices:
            # batch mode
            indices = [int(i) for i in args.batch_indices.split(",")]
            question = args.question or "What mineral is this?"
            for idx in indices:
                spectrum_text = encode_spectrum_from_raw(X[idx])
                answer = ask(model, tokenizer, spectrum_text, question, args.max_tokens)
                label = CLASS_NAME.get(int(y[idx]), f"Class_{y[idx]}")
                print(f"[{idx}] True: {label} | Predicted: {answer}")
            return

        idx = args.index or 0
        print(f"Using pixel index {idx} (label: {CLASS_NAME.get(int(y[idx]), y[idx])})")
        spectrum_text = encode_spectrum_from_raw(X[idx])

    if args.show_encoding:
        print("\n" + spectrum_text + "\n")

    if args.interactive:
        interactive_mode(model, tokenizer, spectrum_text)
    elif args.question:
        answer = ask(model, tokenizer, spectrum_text, args.question, args.max_tokens)
        print(f"\nQ: {args.question}")
        print(f"A: {answer}")
    else:
        default_questions = [
            "What mineral is this?",
            "What absorption features are present in this spectrum?",
            "What mineral group does this belong to?",
        ]
        for q in default_questions:
            answer = ask(model, tokenizer, spectrum_text, q, args.max_tokens)
            print(f"\nQ: {q}")
            print(f"A: {answer}")


if __name__ == "__main__":
    main()
