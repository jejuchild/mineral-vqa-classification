#!/usr/bin/env python3
"""
SOM-based Quality Control Analysis for CRISM Mineral Classification Training Data.

Generates 7 visualizations and 4 CSV reports to identify:
- Mineral confusion zones (spectrally indistinguishable pairs)
- Scene dependency / bias (clusters driven by observation rather than mineralogy)
- Potential mislabeled or noisy samples
- Class imbalance issues

Usage:
    python som_qc_analysis.py
"""

import os
import sys
import re
import glob
import time
import csv
import subprocess
import numpy as np
from collections import defaultdict, Counter

# Ensure minisom is installed
try:
    from minisom import MiniSom
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "minisom"])
    from minisom import MiniSom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from scipy.spatial.distance import cdist

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = "/home/cspark/data/crism_ml"
OUT_DIR = "/home/cspark/mineral_classification/som_results"
os.makedirs(OUT_DIR, exist_ok=True)

IGNORE_LABEL = -1
# NOTE: Training code uses 65534.0 (not 65535.0 which is the actual ATP fill).
# This means the training fill-filter was effectively a no-op, and the CNN was
# trained on all labeled pixels including those with some 65535-valued bands.
# We match this behavior so the SOM analyzes the exact same data the CNN saw.
FILL_VALUE = 65534.0
IR_COUNT = 350
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

VALID_PREFIXES = ("frt", "frs", "hrl", "hrs")

# CRISM IR wavelength table (350 bands, 1.021-3.477 um)
BANDS = np.array([
    1.021, 1.02755, 1.0341, 1.04065, 1.0472, 1.05375, 1.0603, 1.06685,
    1.07341, 1.07996, 1.08651, 1.09307, 1.09962, 1.10617, 1.11273, 1.11928,
    1.12584, 1.13239, 1.13895, 1.14551, 1.15206, 1.15862, 1.16518, 1.17173,
    1.17829, 1.18485, 1.19141, 1.19797, 1.20453, 1.21109, 1.21765, 1.22421,
    1.23077, 1.23733, 1.24389, 1.25045, 1.25701, 1.26357, 1.27014, 1.2767,
    1.28326, 1.28983, 1.29639, 1.30295, 1.30952, 1.31608, 1.32265, 1.32921,
    1.33578, 1.34234, 1.34891, 1.35548, 1.36205, 1.36861, 1.37518, 1.38175,
    1.38832, 1.39489, 1.40145, 1.40802, 1.41459, 1.42116, 1.42773, 1.43431,
    1.44088, 1.44745, 1.45402, 1.46059, 1.46716, 1.47374, 1.48031, 1.48688,
    1.49346, 1.50003, 1.50661, 1.51318, 1.51976, 1.52633, 1.53291, 1.53948,
    1.54606, 1.55264, 1.55921, 1.56579, 1.57237, 1.57895, 1.58552, 1.5921,
    1.59868, 1.60526, 1.61184, 1.61842, 1.625, 1.63158, 1.63816, 1.64474,
    1.65133, 1.65791, 1.66449, 1.67107, 1.67766, 1.68424, 1.69082, 1.69741,
    1.70399, 1.71058, 1.71716, 1.72375, 1.73033, 1.73692, 1.74351, 1.75009,
    1.75668, 1.76327, 1.76985, 1.77644, 1.78303, 1.78962, 1.79621, 1.8028,
    1.80939, 1.81598, 1.82257, 1.82916, 1.83575, 1.84234, 1.84893, 1.85552,
    1.86212, 1.86871, 1.8753, 1.8819, 1.88849, 1.89508, 1.90168, 1.90827,
    1.91487, 1.92146, 1.92806, 1.93465, 1.94125, 1.94785, 1.95444, 1.96104,
    1.96764, 1.97424, 1.98084, 1.98743, 1.99403, 2.00063, 2.00723, 2.01383,
    2.02043, 2.02703, 2.03363, 2.04024, 2.04684, 2.05344, 2.06004, 2.06664,
    2.07325, 2.07985, 2.08645, 2.09306, 2.09966, 2.10627, 2.11287, 2.11948,
    2.12608, 2.13269, 2.1393, 2.1459, 2.15251, 2.15912, 2.16572, 2.17233,
    2.17894, 2.18555, 2.19216, 2.19877, 2.20538, 2.21199, 2.2186, 2.22521,
    2.23182, 2.23843, 2.24504, 2.25165, 2.25827, 2.26488, 2.27149, 2.2781,
    2.28472, 2.29133, 2.29795, 2.30456, 2.31118, 2.31779, 2.32441, 2.33102,
    2.33764, 2.34426, 2.35087, 2.35749, 2.36411, 2.37072, 2.37734, 2.38396,
    2.39058, 2.3972, 2.40382, 2.41044, 2.41706, 2.42368, 2.4303, 2.43692,
    2.44354, 2.45017, 2.45679, 2.46341, 2.47003, 2.47666, 2.48328, 2.4899,
    2.49653, 2.50312, 2.50972, 2.51632, 2.52292, 2.52951, 2.53611, 2.54271,
    2.54931, 2.55591, 2.56251, 2.56911, 2.57571, 2.58231, 2.58891, 2.59551,
    2.60212, 2.60872, 2.61532, 2.62192, 2.62853, 2.63513, 2.64174, 2.64834,
    2.80697, 2.81358, 2.8202, 2.82681, 2.83343, 2.84004, 2.84666, 2.85328,
    2.85989, 2.86651, 2.87313, 2.87975, 2.88636, 2.89298, 2.8996, 2.90622,
    2.91284, 2.91946, 2.92608, 2.9327, 2.93932, 2.94595, 2.95257, 2.95919,
    2.96581, 2.97244, 2.97906, 2.98568, 2.99231, 2.99893, 3.00556, 3.01218,
    3.01881, 3.02544, 3.03206, 3.03869, 3.04532, 3.05195, 3.05857, 3.0652,
    3.07183, 3.07846, 3.08509, 3.09172, 3.09835, 3.10498, 3.11161, 3.11825,
    3.12488, 3.13151, 3.13814, 3.14478, 3.15141, 3.15804, 3.16468, 3.17131,
    3.17795, 3.18458, 3.19122, 3.19785, 3.20449, 3.21113, 3.21776, 3.2244,
    3.23104, 3.23768, 3.24432, 3.25096, 3.2576, 3.26424, 3.27088, 3.27752,
    3.28416, 3.2908, 3.29744, 3.30408, 3.31073, 3.31737, 3.32401, 3.33066,
    3.3373, 3.34395, 3.35059, 3.35724, 3.36388, 3.37053, 3.37717, 3.38382,
    3.39047, 3.39712, 3.40376, 3.41041, 3.41706, 3.42371, 3.43036, 3.43701,
    3.44366, 3.45031, 3.45696, 3.46361, 3.47026, 3.47692
])


# ============================================================
# DATA LOADING FUNCTIONS (from training code)
# ============================================================
def parse_pds3_label(path):
    meta = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("/*") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta


def find_crism_obs_list(data_root, suffix="_ATP_IRONLY.img", sort=True):
    obs_list = [f for f in os.listdir(data_root) if f.endswith(suffix)]
    return sorted(obs_list) if sort else obs_list


def load_atp_ironly_cube(atp_path, trr_lbl, ir_count=IR_COUNT):
    meta = parse_pds3_label(trr_lbl)
    cols = int(meta["LINE_SAMPLES"])
    raw = np.fromfile(atp_path, dtype=np.float32)
    if raw.size % (cols * ir_count) != 0:
        raise ValueError(f"ATP raw size not divisible: {raw.size} vs cols*bands")
    rows = raw.size // (cols * ir_count)
    cube_lbc = raw.reshape(rows, ir_count, cols)
    cube_rcb = np.transpose(cube_lbc, (0, 2, 1))
    return cube_rcb, rows


def load_label(path):
    return np.load(path)


def align_label_to_cube(label, cube_rows, cube_cols):
    label_rows, label_cols = label.shape
    if label_cols != cube_cols:
        if label_cols == cube_cols * 2:
            label = label[:, ::2]
            label_cols = label.shape[1]
        elif cube_cols == label_cols * 2:
            label = np.repeat(label, 2, axis=1)
            label_cols = label.shape[1]
        else:
            return None
    if label_rows == cube_rows:
        return label
    elif label_rows > cube_rows:
        start = (label_rows - cube_rows) // 2
        return label[start:start + cube_rows, :]
    else:
        return label


def extract_labeled_pixels(cube, label_map, fill_value=FILL_VALUE):
    valid_mask = (
        (label_map != IGNORE_LABEL) &
        np.isfinite(cube[..., 0]) &
        (cube[..., 0] != fill_value)
    )
    for b in range(cube.shape[-1]):
        valid_mask &= np.isfinite(cube[..., b]) & (cube[..., b] != fill_value)
    return cube[valid_mask], label_map[valid_mask]


def find_label_for_img(img_name, data_root):
    stem = img_name.split("_")[0]
    m = re.match(r"(frt|frs|hrl|hrs)([0-9a-fA-F]+)", stem)
    if m is None:
        return None, None
    obs_hex = m.group(2).upper().lstrip("0")
    candidates = glob.glob(os.path.join(data_root, f"im_*{obs_hex}_LABEL.npy"))
    if len(candidates) == 0:
        return obs_hex, None
    return obs_hex, candidates[0]


# ============================================================
# DATA LOADING PIPELINE
# ============================================================
def load_all_observations(data_root):
    """Load all observations, extract labeled pixels with scene tracking."""
    obs_list = find_crism_obs_list(data_root)
    obs_list = [f for f in obs_list if f.startswith(VALID_PREFIXES)]
    print(f"  Found {len(obs_list)} valid observations")

    X_all, y_all, scene_all = [], [], []

    for i, img_name in enumerate(obs_list):
        obs_id, label_path = find_label_for_img(img_name, data_root)
        if label_path is None:
            print(f"  [{i+1}/{len(obs_list)}] {img_name}: SKIP (no label)")
            continue

        trr_lbl = os.path.join(data_root, img_name.replace("_ATP_IRONLY.img", ".lbl"))
        if not os.path.exists(trr_lbl):
            print(f"  [{i+1}/{len(obs_list)}] {img_name}: SKIP (no .lbl)")
            continue

        img_path = os.path.join(data_root, img_name)
        labels_full = load_label(label_path)
        cube, atp_rows = load_atp_ironly_cube(img_path, trr_lbl)
        cube_rows, cube_cols = cube.shape[:2]

        labels = align_label_to_cube(labels_full, cube_rows, cube_cols)
        if labels is None:
            print(f"  [{i+1}/{len(obs_list)}] {img_name}: SKIP (col mismatch)")
            continue

        if labels.shape[0] < cube_rows:
            start = (cube_rows - labels.shape[0]) // 2
            cube = cube[start:start + labels.shape[0], :, :]

        if cube.shape[:2] != labels.shape:
            print(f"  [{i+1}/{len(obs_list)}] {img_name}: SKIP (shape mismatch)")
            continue

        X, y_obs = extract_labeled_pixels(cube, labels)
        del cube

        if len(X) > 0:
            X_all.append(X)
            y_all.append(y_obs)
            scene_all.append(np.full(len(X), obs_id, dtype=object))
            print(f"  [{i+1}/{len(obs_list)}] {img_name}: {len(X):,} pixels (obs={obs_id})")

    if len(X_all) == 0:
        raise RuntimeError("No labeled pixels loaded. Check data directory and label files.")

    X = np.concatenate(X_all, axis=0).astype(np.float32)
    y = np.concatenate(y_all, axis=0).astype(np.int64)
    scene_ids = np.concatenate(scene_all, axis=0)
    return X, y, scene_ids


# ============================================================
# LABEL REMAPPING & FILTERING
# ============================================================
def remap_labels(X, y, scene_ids):
    """Apply WATER_UNRELATED merge and KEEP_CLASSES filter (vectorized)."""
    y_new = y.copy()

    # Merge water-unrelated classes
    water_mask = np.isin(y, list(WATER_UNRELATED))
    y_new[water_mask] = NEW_WATER_UNRELATED_ID

    # Keep only valid classes
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


def compute_first_derivative(X):
    """Compute first derivative along spectral axis."""
    return np.diff(X, axis=1)


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
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s")

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
    ax.set_title("U-Matrix (Average Neighbor Distance)", fontsize=14)
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

    ax.set_title("Mineral Confusion Map (Pie Charts per Neuron)", fontsize=14)
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
    ax.set_title("Scene Dependency Map (Red=1 scene, Green=many)", fontsize=14)
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
    ax.set_title("Class-by-Scene Heatmap (log scale)", fontsize=14)
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

    fig.suptitle("Per-Class SOM Footprint", fontsize=14)
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

    # Also build neighbor distributions for posterior
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
    ax.set_title("Noise / Mislabel Detection")
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
        f"Class Imbalance: Sample Counts per Mineral Class (max/min ratio: {ratio:.1f}x)",
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
    print("SOM-based Quality Control Analysis")
    print("CRISM Mineral Classification Training Data")
    print("=" * 70)

    np.random.seed(SEED)
    t_start = time.time()

    # Step 1: Load
    print("\n[STEP 1] Loading all observations...")
    X, y, scene_ids = load_all_observations(DATA_ROOT)
    print(f"  Total: {len(X):,} pixels, {len(set(y))} classes, {len(set(scene_ids))} scenes")

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
    del X  # Free raw spectra (~670 MB)
    print(f"  Normalized shape: {X_norm.shape}")
    print("  (First-derivative version available as fallback)")

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
    print("ANALYSIS COMPLETE")
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
