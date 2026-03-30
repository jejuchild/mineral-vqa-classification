#!/usr/bin/env python3
"""Encode CR-preprocessed CRISM spectra as structured text for LLM input.

Encoding includes:
  1. Per-group statistics (mean, min value, min wavelength position)
  2. Auto-detected absorption features (position, depth, molecular cause)
  3. Subsampled spectrum values (~70 evenly spaced points)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema

from config import BANDS, GROUPS_IDX, GROUP_NAMES, N_BANDS


# ---------------------------------------------------------------------------
# Preprocessing (matching crism_v2_pipeline.py)
# ---------------------------------------------------------------------------

def interpolate_fill(spectra: np.ndarray) -> np.ndarray:
    """Replace fill bands (0.0) with linear interpolation."""
    out = spectra.copy()
    for i in range(len(out)):
        spec = out[i]
        zero_mask = spec == 0.0
        if not zero_mask.any():
            continue
        valid = ~zero_mask
        if valid.sum() < 2:
            continue
        xp = np.where(valid)[0]
        fp = spec[valid]
        out[i] = np.interp(np.arange(len(spec)), xp, fp).astype(np.float32)
    return out


def upper_hull_cr(wl: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Continuum removal via upper convex hull for one spectrum."""
    from scipy.spatial import ConvexHull

    n = len(spec)
    if n < 3:
        return spec.copy()
    pts = np.column_stack([wl, spec])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return spec.copy()

    verts = hull.vertices
    upper = verts[pts[verts, 1] >= np.median(spec)]
    if len(upper) < 2:
        return spec.copy()
    upper = upper[np.argsort(pts[upper, 0])]

    if 0 not in upper:
        upper = np.concatenate([[0], upper])
    if n - 1 not in upper:
        upper = np.concatenate([upper, [n - 1]])

    continuum = np.interp(wl, pts[upper, 0], pts[upper, 1])
    continuum = np.maximum(continuum, 1e-10)
    cr = spec / continuum
    return np.clip(cr, 0.0, 2.0).astype(np.float32)


def preprocess_single(spectrum: np.ndarray) -> np.ndarray:
    """Preprocess a single spectrum: fill interp → clip → CR."""
    spec = spectrum.copy()
    # fill interpolation
    zero_mask = spec == 0.0
    if zero_mask.any():
        valid = ~zero_mask
        if valid.sum() >= 2:
            xp = np.where(valid)[0]
            fp = spec[valid]
            spec = np.interp(np.arange(len(spec)), xp, fp).astype(np.float32)
    spec = np.clip(spec, 0.0, 1.0)
    spec = upper_hull_cr(BANDS, spec)
    return spec


def preprocess_batch(spectra: np.ndarray) -> np.ndarray:
    """Preprocess a batch of spectra."""
    out = interpolate_fill(spectra)
    out = np.clip(out, 0.0, 1.0)
    result = np.empty_like(out)
    for i in range(len(out)):
        result[i] = upper_hull_cr(BANDS, out[i])
    return result


# ---------------------------------------------------------------------------
# Absorption detection
# ---------------------------------------------------------------------------

def detect_absorptions(
    spectrum: np.ndarray,
    min_depth: float = 0.03,
    window: int = 7,
    max_features: int = 8,
) -> list[dict]:
    """Detect absorption features as local minima in CR spectrum."""
    kernel = np.ones(window) / window
    smoothed = np.convolve(spectrum, kernel, mode="same")

    minima_idx = argrelextrema(smoothed, np.less, order=window)[0]

    absorptions = []
    for idx in minima_idx:
        depth = 1.0 - smoothed[idx]
        if depth >= min_depth:
            absorptions.append({
                "wavelength": float(BANDS[idx]),
                "depth": round(float(depth), 3),
                "band_idx": int(idx),
            })

    absorptions.sort(key=lambda x: -x["depth"])
    return absorptions[:max_features]


# ---------------------------------------------------------------------------
# Absorption cause matching
# ---------------------------------------------------------------------------

_ABSORPTION_CATALOG = None

def _load_catalog() -> dict:
    global _ABSORPTION_CATALOG
    if _ABSORPTION_CATALOG is None:
        catalog_path = Path(__file__).parent.parent / "knowledge" / "absorption_bands.json"
        with open(catalog_path) as f:
            _ABSORPTION_CATALOG = json.load(f)
    return _ABSORPTION_CATALOG


def match_cause(wl: float, tolerance: float = 0.04) -> str:
    """Match absorption wavelength to molecular cause."""
    catalog = _load_catalog()
    vibs = catalog.get("molecular_vibrations", {})
    best = None
    best_dist = tolerance

    for name, info in vibs.items():
        wl_range = info.get("wavelength_range", [])
        if len(wl_range) == 2 and wl_range[0] - tolerance <= wl <= wl_range[1] + tolerance:
            center = info.get("typical_center", sum(wl_range) / 2)
            dist = abs(wl - center)
            if dist < best_dist:
                best_dist = dist
                best = info.get("cause", name)
    return best or ""


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

SUBSAMPLE_N = 70  # number of evenly spaced points to include


def encode_spectrum(spectrum_cr: np.ndarray) -> str:
    """Encode a CR-preprocessed spectrum as structured text.

    Args:
        spectrum_cr: Continuum-removed spectrum (350 bands), values ~0-2.

    Returns:
        Structured text representation for LLM input.
    """
    parts = ["[SPECTRUM]"]
    parts.append("Wavelength: 1.02-3.92um, 350 bands, continuum-removed")

    # 1. Group statistics
    parts.append("Group stats:")
    for gi, ((s, e), gname) in enumerate(zip(GROUPS_IDX, GROUP_NAMES)):
        seg = spectrum_cr[s:e]
        seg_wl = BANDS[s:e]
        mean_val = float(np.mean(seg))
        min_idx = int(np.argmin(seg))
        min_val = float(seg[min_idx])
        min_wl = float(seg_wl[min_idx])
        parts.append(
            f"  B{gi} ({seg_wl[0]:.2f}-{seg_wl[-1]:.2f}um, {gname}): "
            f"mean={mean_val:.3f}, min={min_val:.3f} at {min_wl:.2f}um"
        )

    # 2. Detected absorptions
    absorptions = detect_absorptions(spectrum_cr)
    if absorptions:
        parts.append("Absorptions:")
        for ab in absorptions:
            wl = ab["wavelength"]
            depth = ab["depth"]
            cause = match_cause(wl)
            cause_str = f", {cause}" if cause else ""
            parts.append(f"  {wl:.2f}um (depth={depth:.3f}{cause_str})")
    else:
        parts.append("Absorptions: none detected")

    # 3. Subsampled values
    indices = np.linspace(0, N_BANDS - 1, SUBSAMPLE_N, dtype=int)
    vals = [f"{BANDS[i]:.2f}:{spectrum_cr[i]:.3f}" for i in indices]
    parts.append(f"Sampled ({SUBSAMPLE_N} pts): " + ", ".join(vals))

    parts.append("[/SPECTRUM]")
    return "\n".join(parts)


def encode_spectrum_from_raw(spectrum_raw: np.ndarray) -> str:
    """Preprocess raw spectrum and encode as text."""
    cr = preprocess_single(spectrum_raw)
    return encode_spectrum(cr)
