#!/usr/bin/env python3
"""
SOM-based Quality Control Analysis for CRISM Mineral Classification Training Data.
Variant: loads pre-extracted data from CRISM_labeled_pixels_ratioed.mat
(used by earlier CNN models v3-v5).

Usage:
    python som_qc_analysis_mat.py
"""

import os
import sys
import time
import csv
import subprocess
import numpy as np
from collections import defaultdict, Counter

try:
    from minisom import MiniSom
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "minisom"])
    from minisom import MiniSom

from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from scipy.spatial.distance import cdist

# ============================================================
# CONFIGURATION
# ============================================================
MAT_PATH = "/home/cspark/data/crism_ml/CRISM_labeled_pixels_ratioed.mat"
OUT_DIR = "/home/cspark/mineral_classification/som_results_mat"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
MIN_SAMPLES = 100

SOM_TRAIN_SUBSAMPLE = 100_000
SOM_MAX_SIDE = 30
SOM_MIN_SIDE = 10
SOM_SIGMA = 1.5
SOM_LEARNING_RATE = 0.5
SOM_ITER_MULTIPLIER = 5

WATER_UNRELATED = {5, 13, 30, 33, 34, 35, 36, 37}
KEEP_CLASSES = {
    1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12,
    14, 15, 16, 17, 18, 19, 23, 25, 26,
    27, 29, 31, 38, 39
}
NEW_WATER_UNRELATED_ID = 100

CLASS_NAME = {
    1: "CO2 Ice",
    2: "H2O Ice",
    3: "Gypsum",
    4: "Ferric Hydroxysulfate",
    6: "Fe smectite",
    7: "Mg smectite",
    8: "Prehnite",
    9: "Jarosite",
    10: "Serpentine",
    11: "Alunite",
    12: "Akaganeite",
    14: "Al smectite 1",
    15: "Kaolinite",
    16: "Bassanite",
    17: "Epidote",
    18: "Al smectite 2",
    19: "Polyhydrated sulfate",
    23: "Illite",
    25: "Analcime",
    26: "Monohydrated sulfate",
    27: "Hydrated silica",
    29: "Ferricopiapite",
    31: "Chlorite",
    38: "Artifact",
    39: "Neutral",
    100: "Water-unrelated",
}


# ============================================================
# DATA LOADING (from .mat file)
# ============================================================
def load_mat_data(mat_path):
    """
    Load CRISM_labeled_pixels_ratioed.mat.
    Returns X (N, 350), y (N,), scene_ids (N,) as string scene names.

    Mat file keys:
      pixspec  (592413, 350) float64  - ratioed spectra
      pixlabs  (592413, 1)   uint8    - mineral labels (1-39)
      pixims   (592413, 1)   int16    - scene index (1-indexed)
      im_names (77,)         str      - scene hex IDs
    """
    print(f"  Loading {mat_path}...")
    mat = loadmat(mat_path)

    X = mat["pixspec"].astype(np.float32)
    y = mat["pixlabs"].squeeze().astype(np.int64)
    pixims = mat["pixims"].squeeze()  # 1-indexed scene indices
    im_names = mat["im_names"]  # (77,) array of scene name strings

    # Map scene indices to scene name strings
    scene_ids = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        idx = int(pixims[i]) - 1  # convert 1-indexed to 0-indexed
        scene_ids[i] = str(im_names[idx])

    return X, y, scene_ids


# ============================================================
# LABEL REMAPPING & FILTERING
# ============================================================
def remap_labels(X, y, scene_ids):
    """Apply WATER_UNRELATED merge and KEEP_CLASSES filter (vectorized)."""
    y_new = y.copy()
    water_mask = np.isin(y, list(WATER_UNRELATED))
    y_new[water_mask] = NEW_WATER_UNRELATED_ID
    keep_mask = water_mask | np.isin(y, list(KEEP_CLASSES))
    return X[keep_mask], y_new[keep_mask], scene_ids[keep_mask]


def filter_small_classes(X, y, scene_ids, min_samples=MIN_SAMPLES):
    """Remove classes with fewer than min_samples."""
    unique, counts = np.unique(y, return_counts=True)
    valid = set(unique[counts >= min_samples])
    removed = set(unique) - valid
    if removed:
        for r in sorted(removed):
            cnt = counts[unique == r][0]
            print(f"    Removed class {r} ({CLASS_NAME.get(int(r), '?')}): {cnt} samples")
    mask = np.isin(y, list(valid))
    return X[mask], y[mask], scene_ids[mask], valid


# ============================================================
# PREPROCESSING
# ============================================================
def mean_normalize(X):
    """Mean-normalize each spectrum: spectrum / spectrum.mean()."""
    means = X.mean(axis=1, keepdims=True)
    means = np.where(np.abs(means) < 1e-10, 1.0, means)
    return X / means


# ============================================================
# SOM TRAINING
# ============================================================
def stratified_subsample(y, n_total, rng):
    """Stratified subsample maintaining class proportions."""
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    n_per_class = np.maximum((proportions * n_total).astype(int), 1)

    diff = n_total - n_per_class.sum()
    if diff > 0:
        top_k = np.argsort(-counts)[:diff]
        n_per_class[top_k] += 1
    elif diff < 0:
        top_k = np.argsort(counts)[:abs(diff)]
        n_per_class[top_k] = np.maximum(n_per_class[top_k] - 1, 1)

    indices = []
    for cls, n_cls in zip(classes, n_per_class):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) <= n_cls:
            indices.extend(cls_idx)
        else:
            indices.extend(rng.choice(cls_idx, n_cls, replace=False))
    return np.array(indices)


def train_som(X_norm, y):
    """Train MiniSom on (possibly subsampled) data."""
    som_side = min(int(np.sqrt(len(X_norm) / 10)), SOM_MAX_SIDE)
    som_side = max(som_side, SOM_MIN_SIDE)
    print(f"  SOM grid: {som_side}x{som_side}")

    som = MiniSom(
        x=som_side, y=som_side,
        input_len=X_norm.shape[1],
        sigma=SOM_SIGMA,
        learning_rate=SOM_LEARNING_RATE,
        neighborhood_function='gaussian',
        topology='rectangular',
        random_seed=SEED,
    )

    if len(X_norm) > SOM_TRAIN_SUBSAMPLE:
        rng = np.random.RandomState(SEED)
        idx = stratified_subsample(y, SOM_TRAIN_SUBSAMPLE, rng)
        X_train = X_norm[idx]
        print(f"  Subsampled {len(X_train):,} for training (from {len(X_norm):,})")
    else:
        X_train = X_norm

    som.random_weights_init(X_train)
    n_iter = SOM_ITER_MULTIPLIER * len(X_train)
    print(f"  Training for {n_iter:,} iterations...")
    t0 = time.time()
    som.train_random(X_train, n_iter, verbose=True)
    print(f"  Training complete in {time.time() - t0:.1f}s")

    return som, som_side


def map_all_to_som(som, X_norm, batch_size=10000):
    """Vectorized BMU mapping using cdist for speed."""
    weights = som.get_weights()
    som_x, som_y, _ = weights.shape
    W = weights.reshape(-1, X_norm.shape[1])

    bmus = np.empty((len(X_norm), 2), dtype=int)
    for i in range(0, len(X_norm), batch_size):
        batch = X_norm[i:i + batch_size]
        dists = cdist(batch, W, metric='euclidean')
        winners = np.argmin(dists, axis=1)
        bmus[i:i + batch_size, 0] = winners // som_y
        bmus[i:i + batch_size, 1] = winners % som_y
    return bmus


# ============================================================
# COLOR UTILITIES
# ============================================================
def build_class_colors(all_classes):
    """Build a distinct color for each class."""
    tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    all_colors = np.vstack([tab20, tab20b])
    return {cls: all_colors[i % len(all_colors)] for i, cls in enumerate(sorted(all_classes))}


# ============================================================
# VIZ 1: U-MATRIX
# ============================================================
def plot_umatrix(som, som_side):
    umat = som.distance_map()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(umat, cmap='bone_r', interpolation='nearest')
    ax.set_title("U-Matrix (Average Neighbor Distance) [.mat data]", fontsize=14)
    ax.set_xlabel("SOM Column")
    ax.set_ylabel("SOM Row")
    plt.colorbar(im, ax=ax, label="Normalized Distance")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_umatrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {path}")


# ============================================================
# VIZ 2: MINERAL CONFUSION MAP
# ============================================================
def draw_pie_at(ax, cx, cy, sizes, colors, radius=0.4):
    theta1 = 0
    for size, color in zip(sizes, colors):
        if size <= 0:
            continue
        theta2 = theta1 + size * 360
        wedge = Wedge((cx, cy), radius, theta1, theta2,
                      facecolor=color, edgecolor='white', linewidth=0.2)
        ax.add_patch(wedge)
        theta1 = theta2


def compute_confusion_pairs(neuron_classes, threshold=0.10):
    """Rank mineral pairs by co-occurrence in neurons (both > threshold)."""
    pair_scores = Counter()
    pair_neuron_counts = Counter()

    for (bx, by), counter in neuron_classes.items():
        total = sum(counter.values())
        if total < 5:
            continue
        classes = sorted(counter.keys())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                ca, cb = classes[i], classes[j]
                frac_a = counter[ca] / total
                frac_b = counter[cb] / total
                if frac_a > threshold and frac_b > threshold:
                    pair_scores[(ca, cb)] += min(frac_a, frac_b)
                    pair_neuron_counts[(ca, cb)] += 1

    result = [
        (ca, cb, score, pair_neuron_counts[(ca, cb)])
        for (ca, cb), score in pair_scores.items()
    ]
    result.sort(key=lambda x: -x[2])
    return result


def plot_mineral_confusion_map(som_side, bmus, y):
    neuron_classes = defaultdict(Counter)
    for (bx, by), label in zip(bmus, y):
        neuron_classes[(bx, by)][label] += 1

    all_classes = sorted(set(y))
    class_colors = build_class_colors(all_classes)

    fig, ax = plt.subplots(figsize=(som_side * 1.2 + 4, som_side * 1.2))
    ax.set_xlim(-0.5, som_side - 0.5)
    ax.set_ylim(-0.5, som_side - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(range(som_side))
    ax.set_yticks(range(som_side))
    ax.grid(True, alpha=0.2)

    for (bx, by), counter in neuron_classes.items():
        total = sum(counter.values())
        if total < 2:
            continue
        sizes = []
        pie_colors = []
        for cls in sorted(counter.keys()):
            sizes.append(counter[cls] / total)
            pie_colors.append(class_colors[cls])
        draw_pie_at(ax, by, bx, sizes, pie_colors, radius=0.45)

    ax.set_title("Mineral Confusion Map [.mat data]", fontsize=14)
    ax.set_xlabel("SOM Column")
    ax.set_ylabel("SOM Row")

    patches = [
        mpatches.Patch(color=class_colors[c], label=f"{c}: {CLASS_NAME.get(c, '?')}")
        for c in all_classes
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=7, ncol=1)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_mineral_confusion_map.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {path}")

    confusion_pairs = compute_confusion_pairs(neuron_classes)
    return confusion_pairs


# ============================================================
# VIZ 3: SCENE DEPENDENCY MAP
# ============================================================
def plot_scene_dependency_map(som_side, bmus, scene_ids):
    neuron_scenes = defaultdict(Counter)
    for (bx, by), sid in zip(bmus, scene_ids):
        neuron_scenes[(bx, by)][sid] += 1

    diversity_map = np.full((som_side, som_side), np.nan)
    dominated_neurons = []

    for bx in range(som_side):
        for by in range(som_side):
            counter = neuron_scenes.get((bx, by))
            if counter is None or sum(counter.values()) == 0:
                continue
            n_scenes = len(counter)
            total = sum(counter.values())
            diversity_map[bx, by] = n_scenes

            top_scene, top_count = counter.most_common(1)[0]
            if top_count / total > 0.80:
                dominated_neurons.append({
                    'bmu_x': bx, 'bmu_y': by,
                    'dominant_scene': top_scene,
                    'dominant_pct': top_count / total,
                    'total_samples': total,
                    'n_scenes': n_scenes,
                })

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(diversity_map, cmap='RdYlGn', interpolation='nearest')
    ax.set_title("Scene Dependency Map [.mat data]", fontsize=14)
    ax.set_xlabel("SOM Column")
    ax.set_ylabel("SOM Row")
    plt.colorbar(im, ax=ax, label="Number of Distinct Scenes")

    for d in dominated_neurons:
        ax.plot(d['bmu_y'], d['bmu_x'], 'kx', markersize=6, markeredgewidth=1.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_scene_dependency_map.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {path}")
    print(f"    {len(dominated_neurons)} neurons dominated by single scene (>80%)")

    return dominated_neurons


# ============================================================
# VIZ 4: CLASS-BY-SCENE HEATMAP
# ============================================================
def plot_class_scene_heatmap(y, scene_ids):
    classes = sorted(set(y))
    scenes = sorted(set(scene_ids))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    scene_to_idx = {s: i for i, s in enumerate(scenes)}

    matrix = np.zeros((len(classes), len(scenes)), dtype=int)
    for yi, si in zip(y, scene_ids):
        matrix[class_to_idx[yi], scene_to_idx[si]] += 1

    per_class_diversity = {}
    for c in classes:
        ci = class_to_idx[c]
        n_scenes = int(np.sum(matrix[ci, :] > 0))
        total = int(matrix[ci, :].sum())
        risk = "HIGH" if n_scenes < 3 else ("MEDIUM" if n_scenes < 5 else "LOW")
        per_class_diversity[c] = {
            'n_scenes': n_scenes,
            'total_samples': total,
            'risk': risk,
        }

    fig_width = max(14, len(scenes) * 0.4)
    fig_height = max(8, len(classes) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    log_matrix = np.log1p(matrix).astype(np.float64)
    im = ax.imshow(log_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(
        [f"{c}: {CLASS_NAME.get(c, '?')} [{per_class_diversity[c]['n_scenes']}sc]"
         for c in classes],
        fontsize=7,
    )
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, fontsize=6, rotation=90)
    ax.set_title("Class-by-Scene Heatmap (log scale) [.mat data]", fontsize=14)
    ax.set_xlabel("Scene ID")
    ax.set_ylabel("Mineral Class")
    plt.colorbar(im, ax=ax, label="log(1 + count)")

    for c, info in per_class_diversity.items():
        if info['n_scenes'] < 3:
            ci = class_to_idx[c]
            ax.axhline(ci, color='red', linewidth=2, linestyle='--', alpha=0.8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_class_scene_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {path}")

    high_risk = [c for c, info in per_class_diversity.items() if info['risk'] == "HIGH"]
    if high_risk:
        print(f"    HIGH risk classes (<3 scenes): {[CLASS_NAME.get(c, c) for c in high_risk]}")

    return per_class_diversity


# ============================================================
# VIZ 5: PER-CLASS SOM FOOTPRINT
# ============================================================
def plot_per_class_footprint(som_side, bmus, y):
    classes = sorted(set(y))
    n_classes = len(classes)
    ncols = min(6, n_classes)
    nrows = (n_classes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, cls in enumerate(classes):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        density = np.zeros((som_side, som_side))
        mask = y == cls
        cls_bmus = bmus[mask]
        np.add.at(density, (cls_bmus[:, 0], cls_bmus[:, 1]), 1)

        ax.imshow(density, cmap='hot', interpolation='nearest')
        n_total = mask.sum()
        ax.set_title(f"{cls}: {CLASS_NAME.get(cls, '?')}\n(n={n_total:,})", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_classes, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    fig.suptitle("Per-Class SOM Footprint [.mat data]", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_per_class_footprint.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {path}")


# ============================================================
# VIZ 6: NOISE / MISLABEL DETECTION
# ============================================================
def compute_noise_scores(som, som_side, bmus, y):
    """Compute prior/posterior per sample and flag clean/analyze/remove."""
    neuron_dist = defaultdict(Counter)
    neuron_totals = defaultdict(int)
    for (bx, by), label in zip(bmus, y):
        neuron_dist[(bx, by)][label] += 1
        neuron_totals[(bx, by)] += 1

    neighbor_dist = defaultdict(Counter)
    neighbor_totals = defaultdict(int)
    for bx in range(som_side):
        for by in range(som_side):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = bx + dx, by + dy
                    if 0 <= nx < som_side and 0 <= ny < som_side:
                        for cls, cnt in neuron_dist[(nx, ny)].items():
                            neighbor_dist[(bx, by)][cls] += cnt
                        neighbor_totals[(bx, by)] += neuron_totals[(nx, ny)]

    priors = np.zeros(len(y), dtype=np.float32)
    posteriors = np.zeros(len(y), dtype=np.float32)
    flags = np.empty(len(y), dtype=object)

    for i in range(len(y)):
        bx, by = bmus[i]
        label = y[i]

        total = neuron_totals[(bx, by)]
        prior = neuron_dist[(bx, by)][label] / total if total > 0 else 0
        priors[i] = prior

        ntotal = neighbor_totals[(bx, by)]
        posterior = neighbor_dist[(bx, by)][label] / ntotal if ntotal > 0 else 0
        posteriors[i] = posterior

        if prior >= 0.6 and posterior >= 0.6:
            flags[i] = "clean"
        elif prior >= 0.6 and posterior < 0.6:
            flags[i] = "analyze"
        else:
            flags[i] = "remove"

    return priors, posteriors, flags


def plot_noise_detection(priors, posteriors, flags):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for flag, color in [("clean", "green"), ("analyze", "orange"), ("remove", "red")]:
        mask = flags == flag
        if mask.sum() > 0:
            ax.hist(priors[mask], bins=50, alpha=0.6,
                    label=f"{flag} ({mask.sum():,})", color=color)
    ax.set_xlabel("Prior P(class | neuron)")
    ax.set_ylabel("Count")
    ax.set_title("Prior Probability Distribution by Flag")
    ax.legend()

    ax = axes[1]
    rng = np.random.RandomState(SEED)
    n_show = min(50000, len(priors))
    idx = rng.choice(len(priors), n_show, replace=False)
    color_map = {"clean": "green", "analyze": "orange", "remove": "red"}
    for flag in ["clean", "analyze", "remove"]:
        mask = flags[idx] == flag
        if mask.sum() > 0:
            ax.scatter(priors[idx][mask], posteriors[idx][mask],
                       c=color_map[flag], alpha=0.15, s=3, label=flag)
    ax.set_xlabel("Prior P(class | BMU neuron)")
    ax.set_ylabel("Posterior P(class | BMU + neighbors)")
    ax.set_title("Noise / Mislabel Detection [.mat data]")
    ax.legend(markerscale=5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_noise_mislabel_detection.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {path}")


# ============================================================
# VIZ 7: CLASS IMBALANCE
# ============================================================
def plot_class_imbalance(y):
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]
    counts = counts[order]

    class_colors = build_class_colors(set(y))
    ratio = counts[0] / counts[-1] if counts[-1] > 0 else float('inf')

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(
        range(len(classes)), counts,
        color=[class_colors[c] for c in classes],
    )

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(
        [f"{c}\n{CLASS_NAME.get(c, '?')}" for c in classes],
        fontsize=7, rotation=45, ha='right',
    )
    ax.set_ylabel("Sample Count")
    ax.set_title(
        f"Class Imbalance [.mat data] (max/min ratio: {ratio:.1f}x)",
        fontsize=13,
    )

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "07_class_imbalance.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved {path}")


# ============================================================
# CSV REPORT GENERATION
# ============================================================
def save_flagged_samples_csv(bmus, priors, posteriors, flags, scene_ids, y):
    path = os.path.join(OUT_DIR, "flagged_samples.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_idx', 'bmu_x', 'bmu_y', 'prior_prob',
            'posterior_prob', 'flag', 'scene_id', 'label', 'mineral_name',
        ])
        for i in range(len(y)):
            writer.writerow([
                i, bmus[i][0], bmus[i][1],
                f"{priors[i]:.4f}", f"{posteriors[i]:.4f}",
                flags[i], scene_ids[i], int(y[i]),
                CLASS_NAME.get(int(y[i]), "Unknown"),
            ])
    print(f"    Saved {path} ({len(y):,} rows)")


def save_confusion_ranking_csv(confusion_pairs):
    path = os.path.join(OUT_DIR, "mineral_confusion_ranking.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'class_a', 'mineral_a', 'class_b', 'mineral_b',
            'confusion_score', 'n_shared_neurons',
        ])
        for rank, (ca, cb, score, n_neurons) in enumerate(confusion_pairs, 1):
            writer.writerow([
                rank, int(ca), CLASS_NAME.get(int(ca), "Unknown"),
                int(cb), CLASS_NAME.get(int(cb), "Unknown"),
                f"{score:.4f}", n_neurons,
            ])
    print(f"    Saved {path} ({len(confusion_pairs)} pairs)")


def save_scene_dependency_csv(dominated_neurons):
    path = os.path.join(OUT_DIR, "scene_dependency_report.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'bmu_x', 'bmu_y', 'dominant_scene',
            'dominant_pct', 'total_samples', 'n_scenes',
        ])
        for d in dominated_neurons:
            writer.writerow([
                d['bmu_x'], d['bmu_y'], d['dominant_scene'],
                f"{d['dominant_pct']:.4f}", d['total_samples'], d['n_scenes'],
            ])
    print(f"    Saved {path} ({len(dominated_neurons)} dominated neurons)")


def save_class_scene_diversity_csv(per_class_diversity):
    path = os.path.join(OUT_DIR, "per_class_scene_diversity.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'class_id', 'mineral_name', 'n_scenes', 'total_samples', 'risk_flag',
        ])
        for cls in sorted(per_class_diversity.keys()):
            info = per_class_diversity[cls]
            writer.writerow([
                int(cls), CLASS_NAME.get(int(cls), "Unknown"),
                info['n_scenes'], info['total_samples'], info['risk'],
            ])
    print(f"    Saved {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("SOM-based Quality Control Analysis (.mat variant)")
    print("CRISM_labeled_pixels_ratioed.mat")
    print("=" * 70)

    np.random.seed(SEED)
    t_start = time.time()

    # Step 1: Load from .mat
    print("\n[STEP 1] Loading .mat data...")
    X, y, scene_ids = load_mat_data(MAT_PATH)
    print(f"  Total: {len(X):,} pixels, {len(np.unique(y))} classes, {len(np.unique(scene_ids))} scenes")

    # Step 2: Remap labels
    print("\n[STEP 2] Remapping labels...")
    X, y, scene_ids = remap_labels(X, y, scene_ids)
    print(f"  After remapping: {len(X):,} pixels, {len(np.unique(y))} classes")

    # Step 3: Filter small classes
    print(f"\n[STEP 3] Filtering small classes (< {MIN_SAMPLES})...")
    X, y, scene_ids, valid_classes = filter_small_classes(X, y, scene_ids)
    print(f"  After filtering: {len(X):,} pixels, {len(valid_classes)} classes")

    # Step 4: Preprocessing
    print("\n[STEP 4] Mean-normalizing spectra...")
    X_norm = mean_normalize(X)
    del X
    print(f"  Normalized shape: {X_norm.shape}")

    # Step 5: Train SOM
    print("\n[STEP 5] Training SOM...")
    som, som_side = train_som(X_norm, y)

    # Step 6: Map all samples
    print("\n[STEP 6] Mapping all samples to BMUs (vectorized)...")
    t0 = time.time()
    bmus = map_all_to_som(som, X_norm)
    print(f"  Mapped {len(bmus):,} samples in {time.time() - t0:.1f}s")

    # Step 7: Visualizations
    print("\n[STEP 7] Generating visualizations...")

    print("  1/7 U-Matrix...")
    plot_umatrix(som, som_side)

    print("  2/7 Mineral Confusion Map...")
    confusion_pairs = plot_mineral_confusion_map(som_side, bmus, y)

    print("  3/7 Scene Dependency Map...")
    dominated_neurons = plot_scene_dependency_map(som_side, bmus, scene_ids)

    print("  4/7 Class-by-Scene Heatmap...")
    per_class_diversity = plot_class_scene_heatmap(y, scene_ids)

    print("  5/7 Per-Class SOM Footprint...")
    plot_per_class_footprint(som_side, bmus, y)

    print("  6/7 Noise/Mislabel Detection...")
    priors, posteriors, flags = compute_noise_scores(som, som_side, bmus, y)
    plot_noise_detection(priors, posteriors, flags)

    print("  7/7 Class Imbalance...")
    plot_class_imbalance(y)

    # Step 8: CSV reports
    print("\n[STEP 8] Saving CSV reports...")
    save_flagged_samples_csv(bmus, priors, posteriors, flags, scene_ids, y)
    save_confusion_ranking_csv(confusion_pairs)
    save_scene_dependency_csv(dominated_neurons)
    save_class_scene_diversity_csv(per_class_diversity)

    # Summary
    elapsed = time.time() - t_start
    n_clean = np.sum(flags == "clean")
    n_analyze = np.sum(flags == "analyze")
    n_remove = np.sum(flags == "remove")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE (.mat variant)")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Total samples analyzed: {len(y):,}")
    print(f"  SOM grid: {som_side}x{som_side}")
    print(f"  Sample flags:")
    print(f"    Clean:   {n_clean:,} ({100*n_clean/len(y):.1f}%)")
    print(f"    Analyze: {n_analyze:,} ({100*n_analyze/len(y):.1f}%)")
    print(f"    Remove:  {n_remove:,} ({100*n_remove/len(y):.1f}%)")
    print(f"  Scene-dominated neurons: {len(dominated_neurons)}")
    if confusion_pairs:
        print(f"  Top 5 confused mineral pairs:")
        for ca, cb, score, n_neurons in confusion_pairs[:5]:
            na = CLASS_NAME.get(int(ca), '?')
            nb = CLASS_NAME.get(int(cb), '?')
            print(f"    {na} vs {nb}: score={score:.3f} ({n_neurons} neurons)")

    high_risk = [
        c for c, info in per_class_diversity.items() if info['risk'] == "HIGH"
    ]
    if high_risk:
        print(f"  HIGH scene-risk classes (<3 scenes):")
        for c in high_risk:
            info = per_class_diversity[c]
            print(f"    {CLASS_NAME.get(c, c)}: {info['n_scenes']} scenes, {info['total_samples']:,} samples")

    print(f"\n  All outputs saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
