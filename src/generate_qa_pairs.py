#!/usr/bin/env python3
"""Generate VQA question-answer pairs with spectrum text encoding.

Produces 4 types of QA pairs:
  Type A: Basic classification ("What mineral is this?")
  Type B: Absorption feature description ("Describe the absorption features")
  Type C: Differential diagnosis ("How do you distinguish this from X?")
  Type D: CRISM parameter-based ("What spectral parameters detect this?")

Each QA pair includes the full spectrum text encoding for direct LLM input.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import (
    BANDS, CLASS_NAME, MINERAL_KB_PATH, ABSORPTION_BANDS_PATH,
    VQA_DATASET_PATH, NPZ_PATH, SEED, MAX_SAMPLES,
)
from spectrum_encoder import (
    encode_spectrum, preprocess_batch, detect_absorptions, match_cause,
)


# ---------------------------------------------------------------------------
# Load knowledge bases
# ---------------------------------------------------------------------------

def load_kb() -> dict:
    with open(MINERAL_KB_PATH) as f:
        return json.load(f)

def load_absorption_catalog() -> dict:
    with open(ABSORPTION_BANDS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# QA pair generators
# ---------------------------------------------------------------------------

def generate_type_a(spectrum_text: str, mineral: str, kb_entry: dict) -> list[dict]:
    """Type A: Basic classification questions."""
    group = kb_entry.get("group", "Unknown")
    subgroup = kb_entry.get("subgroup", "Unknown")

    questions = [
        {
            "spectrum": spectrum_text,
            "question": "What mineral is this?",
            "answer": mineral,
            "type": "A",
        },
        {
            "spectrum": spectrum_text,
            "question": "What mineral group does this belong to?",
            "answer": f"{group}, {subgroup} subgroup",
            "type": "A",
        },
    ]

    formula = kb_entry.get("formula")
    if formula:
        questions.append({
            "spectrum": spectrum_text,
            "question": "What is the chemical formula of this mineral?",
            "answer": formula,
            "type": "A",
        })

    mars_ctx = kb_entry.get("mars_context")
    if mars_ctx:
        questions.append({
            "spectrum": spectrum_text,
            "question": "Where on Mars is this mineral typically found?",
            "answer": mars_ctx,
            "type": "A",
        })

    return questions


def generate_type_b(
    spectrum_text: str,
    mineral: str,
    kb_entry: dict,
    detected_absorptions: list[dict],
) -> list[dict]:
    """Type B: Absorption feature description."""
    questions = []

    if detected_absorptions:
        feature_strs = []
        for ab in detected_absorptions[:5]:
            wl = ab["wavelength"]
            depth = ab["depth"]
            cause = match_cause(wl)
            if cause:
                feature_strs.append(f"{wl:.2f}um (depth={depth:.3f}, {cause})")
            else:
                feature_strs.append(f"{wl:.2f}um (depth={depth:.3f})")

        answer = f"Absorption features detected: {'; '.join(feature_strs)}."
        questions.append({
            "spectrum": spectrum_text,
            "question": "What absorption features are present in this spectrum?",
            "answer": answer,
            "type": "B",
        })

    desc = kb_entry.get("spectral_description")
    if desc:
        questions.append({
            "spectrum": spectrum_text,
            "question": "Describe the spectral characteristics of this mineral.",
            "answer": desc,
            "type": "B",
        })

    band_assigns = kb_entry.get("band_assignments", {})
    if band_assigns:
        diag_bands = kb_entry.get("diagnostic_bands_um", [])
        if diag_bands:
            band_str = ", ".join(f"{b}um" for b in diag_bands)
            assign_strs = [f"{k}um: {v}" for k, v in band_assigns.items()]
            answer = (
                f"Diagnostic bands for {mineral}: {band_str}. "
                f"Assignments: {'; '.join(assign_strs)}."
            )
            questions.append({
                "spectrum": spectrum_text,
                "question": f"What are the diagnostic absorption bands for {mineral}?",
                "answer": answer,
                "type": "B",
            })

    return questions


def generate_type_c(spectrum_text: str, mineral: str, kb_entry: dict) -> list[dict]:
    """Type C: Differential diagnosis."""
    questions = []
    distinctions = kb_entry.get("distinguish_from", {})

    for other_mineral, explanation in distinctions.items():
        questions.append({
            "spectrum": spectrum_text,
            "question": f"How do you distinguish this from {other_mineral}?",
            "answer": explanation,
            "type": "C",
        })

    return questions


def generate_type_d(
    spectrum_text: str,
    mineral: str,
    kb_entry: dict,
    absorption_catalog: dict,
) -> list[dict]:
    """Type D: CRISM parameter-based."""
    questions = []
    params = kb_entry.get("crism_params", [])
    if not params:
        return questions

    param_details = absorption_catalog.get("spectral_parameters_viviano_beck", {})
    param_strs = []
    for p in params:
        info = param_details.get(p, {})
        if info:
            param_strs.append(f"{p} ({info.get('detects', 'mineral detection')})")
        else:
            param_strs.append(p)

    answer = f"CRISM spectral parameters for {mineral}: {'; '.join(param_strs)}."
    questions.append({
        "spectrum": spectrum_text,
        "question": "What CRISM spectral parameters would detect this mineral?",
        "answer": answer,
        "type": "D",
    })

    rules = absorption_catalog.get("decision_rules", {})
    group = kb_entry.get("group", "")
    for rule_name, rule_text in rules.items():
        if mineral.lower() in rule_text.lower() or group.lower() in rule_name.lower():
            questions.append({
                "spectrum": spectrum_text,
                "question": f"What spectral decision rule applies to classifying {mineral}?",
                "answer": rule_text,
                "type": "D",
            })
            break

    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate text-based VQA QA pairs")
    parser.add_argument("--npz", type=str, default=str(NPZ_PATH))
    parser.add_argument("--out", type=str, default=str(VQA_DATASET_PATH))
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # load resources
    print("Loading knowledge bases ...")
    kb = load_kb()
    absorption_catalog = load_absorption_catalog()

    print(f"Loading spectra from {args.npz} ...")
    data = np.load(args.npz, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)
    y = data["y"]
    obs_idx = data["obs_idx"]
    n_total = len(X_raw)

    print(f"  Total pixels: {n_total}")
    print(f"  Classes: {np.unique(y)}")

    # stratified subsample
    rng = np.random.RandomState(args.seed)
    if n_total > args.max_samples:
        indices = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n_cls = max(1, int(args.max_samples * len(cls_idx) / n_total))
            chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            indices.append(chosen)
        indices = np.concatenate(indices)
        rng.shuffle(indices)
        indices = indices[:args.max_samples]
    else:
        indices = np.arange(n_total)

    print(f"  Selected: {len(indices)} spectra")

    # preprocess
    print("Preprocessing (fill interp + CR) ...")
    X_sel = preprocess_batch(X_raw[indices])
    y_sel = y[indices]
    obs_sel = obs_idx[indices]

    # generate QA pairs
    print("Generating QA pairs ...")
    all_qa = []
    skipped = 0

    for i in tqdm(range(len(X_sel)), desc="Generating QA pairs"):
        label = int(y_sel[i])
        mineral = CLASS_NAME.get(label)
        if mineral is None:
            skipped += 1
            continue

        kb_entry = kb.get(mineral)
        if kb_entry is None:
            skipped += 1
            continue

        spectrum_cr = X_sel[i]
        spectrum_text = encode_spectrum(spectrum_cr)
        detected = detect_absorptions(spectrum_cr)

        qa_a = generate_type_a(spectrum_text, mineral, kb_entry)
        qa_b = generate_type_b(spectrum_text, mineral, kb_entry, detected)
        qa_c = generate_type_c(spectrum_text, mineral, kb_entry)
        qa_d = generate_type_d(spectrum_text, mineral, kb_entry, absorption_catalog)

        for qa in qa_a + qa_b + qa_c + qa_d:
            qa["obs_idx"] = int(obs_sel[i])
            qa["label"] = label
            all_qa.append(qa)

    print(f"Total QA pairs: {len(all_qa)}")
    print(f"Skipped (no KB): {skipped}")

    type_counts = {}
    for qa in all_qa:
        t = qa["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Type distribution: {type_counts}")

    # save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
