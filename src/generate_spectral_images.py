#!/usr/bin/env python3
"""Generate spectral plot images from CR-preprocessed CRISM spectra.

Reads crism_training_data_f16.npz, applies Continuum Removal,
and saves each spectrum as a PNG plot for VQA training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

from config import (
    BANDS, CLASS_NAME, GROUPS_IDX, GROUP_NAMES,
    MAX_SAMPLES, NPZ_PATH, SPECTRAL_IMAGES_DIR, IMAGE_DPI, SEED,
)


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
    n = len(spec)
    if n < 3:
        return spec.copy()
    pts = np.column_stack([wl, spec])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return spec.copy()

    # upper hull vertices (sorted by wavelength)
    verts = hull.vertices
    upper = verts[pts[verts, 1] >= np.median(spec)]
    if len(upper) < 2:
        return spec.copy()
    upper = upper[np.argsort(pts[upper, 0])]

    # ensure endpoints
    if 0 not in upper:
        upper = np.concatenate([[0], upper])
    if n - 1 not in upper:
        upper = np.concatenate([upper, [n - 1]])

    continuum = np.interp(wl, pts[upper, 0], pts[upper, 1])
    continuum = np.maximum(continuum, 1e-10)
    cr = spec / continuum
    return np.clip(cr, 0.0, 2.0).astype(np.float32)


def continuum_removal(spectra: np.ndarray) -> np.ndarray:
    """Batch continuum removal."""
    out = np.empty_like(spectra)
    for i in range(len(spectra)):
        out[i] = upper_hull_cr(BANDS, spectra[i])
    return out


def preprocess(spectra: np.ndarray) -> np.ndarray:
    """Full preprocessing: interpolate fills -> clip -> CR."""
    result = interpolate_fill(spectra)
    result = np.clip(result, 0.0, 1.0)
    result = continuum_removal(result)
    return result


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

GROUP_COLORS = [
    "#e74c3c", "#e67e22", "#2ecc71", "#3498db",
    "#9b59b6", "#1abc9c", "#95a5a6",
]


def plot_spectrum(
    spectrum: np.ndarray,
    label: int,
    idx: int,
    out_dir: Path,
) -> Path:
    """Create and save a spectral plot image.

    The plot shows the CR spectrum with color-coded spectral regions
    and annotated absorption features.
    """
    fig, ax = plt.subplots(figsize=(3.84, 3.84), dpi=IMAGE_DPI)

    # plot color-coded spectral groups
    for gi, (s, e) in enumerate(GROUPS_IDX):
        ax.plot(
            BANDS[s:e], spectrum[s:e],
            color=GROUP_COLORS[gi], linewidth=0.8, alpha=0.9,
        )

    # overall thin gray line for continuity
    ax.plot(BANDS, spectrum, color="gray", linewidth=0.3, alpha=0.4, zorder=0)

    ax.set_xlim(BANDS[0], BANDS[-1])
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel("Wavelength (μm)", fontsize=7)
    ax.set_ylabel("CR Reflectance", fontsize=7)
    ax.tick_params(labelsize=6)

    mineral = CLASS_NAME.get(label, f"Class_{label}")
    ax.set_title(f"{mineral}", fontsize=8, fontweight="bold")

    # light grid for readability
    ax.grid(True, alpha=0.2, linewidth=0.3)
    ax.axhline(y=1.0, color="k", linewidth=0.3, linestyle="--", alpha=0.3)

    fig.tight_layout(pad=0.3)

    fname = f"spec_{idx:06d}_c{label}.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=IMAGE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate spectral images for VQA")
    parser.add_argument("--npz", type=str, default=str(NPZ_PATH))
    parser.add_argument("--out-dir", type=str, default=str(SPECTRAL_IMAGES_DIR))
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)
    X_raw = data["X"].astype(np.float32)
    y = data["y"]
    obs_idx = data["obs_idx"]

    n_total = len(X_raw)
    print(f"  Total pixels: {n_total}")
    print(f"  Classes: {np.unique(y)}")

    # subsample if needed
    rng = np.random.RandomState(args.seed)
    if n_total > args.max_samples:
        # stratified sampling to keep class distribution
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
    print("Preprocessing (fill interp + CR) ...")
    X_sel = preprocess(X_raw[indices])
    y_sel = y[indices]
    obs_sel = obs_idx[indices]

    # generate images
    print(f"Generating spectral images to {out_dir} ...")
    manifest = []
    for i, (orig_idx, spec, label, oidx) in enumerate(
        tqdm(zip(indices, X_sel, y_sel, obs_sel), total=len(X_sel), desc="Plotting")
    ):
        out_path = plot_spectrum(spec, int(label), i, out_dir)
        manifest.append({
            "index": i,
            "npz_index": int(orig_idx),
            "image": out_path.name,
            "label": int(label),
            "mineral": CLASS_NAME.get(int(label), f"Class_{label}"),
            "obs_idx": int(oidx),
        })

    # save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. {len(manifest)} images saved.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
