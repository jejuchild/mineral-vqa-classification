#!/usr/bin/env python3
"""Generate VQA question-answer pairs from spectral images + mineral KB.

Produces 4 types of QA pairs:
  Type A: Basic classification ("What mineral is this?")
  Type B: Absorption feature description ("Describe the absorption features")
  Type C: Differential diagnosis ("How do you distinguish this from X?")
  Type D: CRISM parameter-based ("What spectral parameters detect this?")
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import (
    BANDS, CLASS_NAME, GROUPS_IDX, GROUP_NAMES,
    MINERAL_KB_PATH, ABSORPTION_BANDS_PATH,
    SPECTRAL_IMAGES_DIR, VQA_DATASET_PATH, NPZ_PATH, SEED,
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
# Absorption band auto-detection from CR spectrum
# ---------------------------------------------------------------------------

def detect_absorptions(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    min_depth: float = 0.03,
    window: int = 7,
) -> list[dict]:
    """Detect absorption features as local minima in CR spectrum.

    Returns list of {wavelength, depth, band_index} sorted by depth.
    """
    from scipy.signal import argrelextrema

    # smooth spectrum slightly
    kernel = np.ones(window) / window
    smoothed = np.convolve(spectrum, kernel, mode="same")

    # find local minima
    minima_idx = argrelextrema(smoothed, np.less, order=window)[0]

    absorptions = []
    for idx in minima_idx:
        depth = 1.0 - smoothed[idx]  # depth below continuum (CR=1.0)
        if depth >= min_depth:
            absorptions.append({
                "wavelength_um": float(wavelengths[idx]),
                "depth": float(depth),
                "band_index": int(idx),
            })

    absorptions.sort(key=lambda x: -x["depth"])
    return absorptions


def match_absorption_to_cause(
    wl: float,
    absorption_catalog: dict,
    tolerance: float = 0.04,
) -> str | None:
    """Match a detected absorption wavelength to known molecular cause."""
    vibs = absorption_catalog.get("molecular_vibrations", {})
    best_match = None
    best_dist = tolerance

    for name, info in vibs.items():
        wl_range = info.get("wavelength_range", [])
        if len(wl_range) == 2:
            if wl_range[0] - tolerance <= wl <= wl_range[1] + tolerance:
                center = info.get("typical_center", sum(wl_range) / 2)
                dist = abs(wl - center)
                if dist < best_dist:
                    best_dist = dist
                    best_match = {
                        "feature_name": name,
                        "cause": info.get("cause", "Unknown"),
                        "typical_center": center,
                        "minerals": info.get("minerals", []),
                    }
    return best_match


# ---------------------------------------------------------------------------
# QA pair generators
# ---------------------------------------------------------------------------

def generate_type_a(image_name: str, mineral: str, kb_entry: dict) -> list[dict]:
    """Type A: Basic classification questions."""
    group = kb_entry.get("group", "Unknown")
    subgroup = kb_entry.get("subgroup", "Unknown")

    questions = [
        {
            "image": image_name,
            "question": "What mineral is this?",
            "answer": mineral,
            "type": "A",
        },
        {
            "image": image_name,
            "question": "What mineral group does this belong to?",
            "answer": f"{group}, {subgroup} subgroup",
            "type": "A",
        },
    ]

    formula = kb_entry.get("formula")
    if formula:
        questions.append({
            "image": image_name,
            "question": "What is the chemical formula of this mineral?",
            "answer": formula,
            "type": "A",
        })

    mars_ctx = kb_entry.get("mars_context")
    if mars_ctx:
        questions.append({
            "image": image_name,
            "question": "Where on Mars is this mineral typically found?",
            "answer": mars_ctx,
            "type": "A",
        })

    return questions


def generate_type_b(
    image_name: str,
    mineral: str,
    kb_entry: dict,
    detected_absorptions: list[dict],
    absorption_catalog: dict,
) -> list[dict]:
    """Type B: Absorption feature description."""
    questions = []

    # describe detected features
    if detected_absorptions:
        feature_strs = []
        for ab in detected_absorptions[:5]:  # top 5 deepest
            wl = ab["wavelength_um"]
            depth = ab["depth"]
            match = match_absorption_to_cause(wl, absorption_catalog)
            if match:
                feature_strs.append(
                    f"{wl:.2f}um (depth={depth:.2f}, {match['cause']})"
                )
            else:
                feature_strs.append(f"{wl:.2f}um (depth={depth:.2f})")

        answer = f"Absorption features detected: {'; '.join(feature_strs)}."
        questions.append({
            "image": image_name,
            "question": "What absorption features are present in this spectrum?",
            "answer": answer,
            "type": "B",
        })

    # spectral description from KB
    desc = kb_entry.get("spectral_description")
    if desc:
        questions.append({
            "image": image_name,
            "question": "Describe the spectral characteristics of this mineral.",
            "answer": desc,
            "type": "B",
        })

    # band assignments
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
                "image": image_name,
                "question": f"What are the diagnostic absorption bands for {mineral}?",
                "answer": answer,
                "type": "B",
            })

    return questions


def generate_type_c(
    image_name: str,
    mineral: str,
    kb_entry: dict,
) -> list[dict]:
    """Type C: Differential diagnosis."""
    questions = []
    distinctions = kb_entry.get("distinguish_from", {})

    for other_mineral, explanation in distinctions.items():
        questions.append({
            "image": image_name,
            "question": f"How do you distinguish this from {other_mineral}?",
            "answer": explanation,
            "type": "C",
        })

    return questions


def generate_type_d(
    image_name: str,
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
            param_strs.append(
                f"{p} ({info.get('detects', 'mineral detection')})"
            )
        else:
            param_strs.append(p)

    answer = f"CRISM spectral parameters for {mineral}: {'; '.join(param_strs)}."
    questions.append({
        "image": image_name,
        "question": "What CRISM spectral parameters would detect this mineral?",
        "answer": answer,
        "type": "D",
    })

    # add decision rule question if applicable
    rules = absorption_catalog.get("decision_rules", {})
    group = kb_entry.get("group", "")
    relevant_rules = []
    for rule_name, rule_text in rules.items():
        if mineral.lower() in rule_text.lower() or group.lower() in rule_name.lower():
            relevant_rules.append(rule_text)

    if relevant_rules:
        questions.append({
            "image": image_name,
            "question": f"What spectral decision rule applies to classifying {mineral}?",
            "answer": relevant_rules[0],
            "type": "D",
        })

    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate VQA QA pairs")
    parser.add_argument("--manifest", type=str,
                        default=str(SPECTRAL_IMAGES_DIR / "manifest.json"))
    parser.add_argument("--npz", type=str, default=str(NPZ_PATH))
    parser.add_argument("--out", type=str, default=str(VQA_DATASET_PATH))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # load resources
    print("Loading knowledge bases ...")
    kb = load_kb()
    absorption_catalog = load_absorption_catalog()

    print("Loading manifest ...")
    with open(args.manifest) as f:
        manifest = json.load(f)

    print(f"Loading spectra from {args.npz} for absorption detection ...")
    data = np.load(args.npz, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)

    # preprocess for absorption detection (reuse from generate_spectral_images)
    from generate_spectral_images import preprocess

    all_qa = []
    skipped = 0

    for entry in tqdm(manifest, desc="Generating QA pairs"):
        image_name = entry["image"]
        label = entry["label"]
        mineral = entry["mineral"]

        # find KB entry
        kb_entry = kb.get(mineral)
        if kb_entry is None:
            skipped += 1
            continue

        # detect absorptions from the actual spectrum using original npz index
        npz_idx = entry.get("npz_index", entry.get("index", 0))
        if npz_idx < len(X_raw):
            spec_raw = X_raw[npz_idx:npz_idx+1]
            spec_cr = preprocess(spec_raw)[0]
            detected = detect_absorptions(spec_cr, BANDS)
        else:
            detected = []

        # generate all QA types
        qa_a = generate_type_a(image_name, mineral, kb_entry)
        qa_b = generate_type_b(image_name, mineral, kb_entry, detected, absorption_catalog)
        qa_c = generate_type_c(image_name, mineral, kb_entry)
        qa_d = generate_type_d(image_name, mineral, kb_entry, absorption_catalog)

        all_qa.extend(qa_a + qa_b + qa_c + qa_d)

    print(f"Total QA pairs: {len(all_qa)}")
    print(f"Skipped (no KB entry): {skipped}")

    # type distribution
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
