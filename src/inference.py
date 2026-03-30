#!/usr/bin/env python3
"""CLI inference tool for mineral spectral VQA.

Supports:
  1. Single spectrum from .npy file → generate plot → ask questions
  2. Pre-generated spectral image → ask questions
  3. Interactive Q&A mode
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from peft import PeftModel

from config import (
    BANDS, CLASS_NAME, MODEL_DIR, MINERAL_KB_PATH,
    ABSORPTION_BANDS_PATH, BASE_MODEL, MAX_LENGTH,
)


def load_model(model_dir: str, base_model: str = BASE_MODEL):
    """Load fine-tuned BLIP-2 with LoRA adapter."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading processor from {model_dir} ...")
    processor = Blip2Processor.from_pretrained(model_dir)

    print(f"Loading base model {base_model} ...")
    model = Blip2ForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print(f"Loading LoRA adapter from {model_dir} ...")
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()

    return model, processor, device


def load_knowledge_context() -> str:
    """Load mineral KB as context string for enhanced inference."""
    try:
        with open(MINERAL_KB_PATH) as f:
            kb = json.load(f)
        with open(ABSORPTION_BANDS_PATH) as f:
            ab = json.load(f)

        rules = ab.get("decision_rules", {})
        context_parts = [
            "Mineral spectroscopy context:",
            "Key decision rules:",
        ]
        for name, rule in rules.items():
            context_parts.append(f"- {name}: {rule}")

        return "\n".join(context_parts)
    except FileNotFoundError:
        return ""


def spectrum_to_image(spectrum: np.ndarray, tmpdir: str | None = None) -> Path:
    """Convert a raw/CR spectrum to a spectral plot image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from generate_spectral_images import (
        preprocess, GROUP_COLORS, GROUPS_IDX,
    )

    # preprocess if raw (values > 1.5 likely raw reflectance)
    if spectrum.max() > 1.5:
        spectrum = preprocess(spectrum.reshape(1, -1))[0]

    fig, ax = plt.subplots(figsize=(3.84, 3.84), dpi=100)

    for gi, (s, e) in enumerate(GROUPS_IDX):
        ax.plot(BANDS[s:e], spectrum[s:e], color=GROUP_COLORS[gi],
                linewidth=0.8, alpha=0.9)
    ax.plot(BANDS, spectrum, color="gray", linewidth=0.3, alpha=0.4, zorder=0)

    ax.set_xlim(BANDS[0], BANDS[-1])
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel("Wavelength (μm)", fontsize=7)
    ax.set_ylabel("CR Reflectance", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.axhline(y=1.0, color="k", linewidth=0.3, linestyle="--", alpha=0.3)
    fig.tight_layout(pad=0.3)

    if tmpdir:
        out_path = Path(tmpdir) / "spectrum_query.png"
    else:
        out_path = Path(tempfile.mktemp(suffix=".png"))
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def ask(
    model,
    processor,
    device,
    image: Image.Image,
    question: str,
    context: str = "",
    max_new_tokens: int = 200,
) -> str:
    """Ask a question about a spectral image."""
    # prepend context to question if available
    full_question = f"{context}\n\nQuestion: {question}" if context else question

    inputs = processor(
        images=image,
        text=full_question,
        return_tensors="pt",
    ).to(device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=3,
            early_stopping=True,
        )

    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return answer


def interactive_mode(model, processor, device, image: Image.Image, context: str):
    """Interactive Q&A loop."""
    print("\n--- Interactive VQA Mode ---")
    print("Type your questions about the spectrum. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = ask(model, processor, device, image, question, context)
        print(f"A: {answer}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mineral Spectral VQA Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask about a spectral image
  python inference.py --image spectrum.png --question "What mineral is this?"

  # Process a raw spectrum file
  python inference.py --spectrum raw_spec.npy --question "What mineral is this?"

  # Interactive mode with an image
  python inference.py --image spectrum.png --interactive

  # Use specific model checkpoint
  python inference.py --model-dir models/vqa_lora/best --image spectrum.png -q "Describe the absorption features"
        """,
    )
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR / "best"))
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--image", type=str, help="Path to spectral plot image")
    parser.add_argument("--spectrum", type=str, help="Path to .npy spectrum file (350 bands)")
    parser.add_argument("-q", "--question", type=str, help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--no-context", action="store_true", help="Disable KB context injection")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    if not args.image and not args.spectrum:
        parser.error("Provide either --image or --spectrum")

    # load model
    model, processor, device = load_model(args.model_dir, args.base_model)

    # load context
    context = "" if args.no_context else load_knowledge_context()

    # prepare image
    if args.spectrum:
        print(f"Loading spectrum from {args.spectrum} ...")
        spectrum = np.load(args.spectrum).astype(np.float32)
        if spectrum.ndim > 1:
            spectrum = spectrum.flatten()[:350]
        image_path = spectrum_to_image(spectrum)
        print(f"  Generated plot: {image_path}")
    else:
        image_path = Path(args.image)

    image = Image.open(image_path).convert("RGB")

    if args.interactive:
        interactive_mode(model, processor, device, image, context)
    elif args.question:
        answer = ask(
            model, processor, device, image,
            args.question, context, args.max_tokens,
        )
        print(f"\nQ: {args.question}")
        print(f"A: {answer}")
    else:
        # default questions
        default_questions = [
            "What mineral is this?",
            "What absorption features are present in this spectrum?",
            "What mineral group does this belong to?",
        ]
        for q in default_questions:
            answer = ask(model, processor, device, image, q, context, args.max_tokens)
            print(f"\nQ: {q}")
            print(f"A: {answer}")


if __name__ == "__main__":
    main()
