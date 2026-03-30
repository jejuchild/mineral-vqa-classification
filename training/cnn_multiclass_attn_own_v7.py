# ==========================================
# Multi-Branch + Branch-Attention CNN (Multiclass)
# v7: Fixes for robust evaluation
#   1) Split by observation (no spatial leakage)
#   2) Input normalization (per-branch standardization)
#   3) Filter FILL_VALUE pixels
#   4) Numerical stability (gradient clipping + logit clamping)
#   5) Class weights computed after observation-wise split
#   6) Learning rate scheduler (ReduceLROnPlateau)
# ==========================================

import os
import json
import re
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# CONFIG
# -------------------------------
DATA_ROOT = "../data/crism_ml"
OUT_DIR = "./epoch_attn_v8"
os.makedirs(OUT_DIR, exist_ok=True)

IGNORE_LABEL = -1
FILL_VALUE = 65534.0

BATCH_TRAIN = 64
BATCH_VAL = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Attention hyperparameters
# -------------------------------
ATT_TEMPERATURE = 0.5    # sharpen attention distribution (lower = sharper)
ATT_ENT_LAMBDA = 0.01    # entropy regularization strength

# -------------------------------
# Gradient clipping for numerical stability
# -------------------------------
GRAD_CLIP_NORM = 1.0

# -------------------------------
# ATP cube fixed geometry
# -------------------------------
ROWS = 450
COLS = 640
IR_COUNT = 350
DTYPE = np.float32

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

# -------------------------------
# Auto-discover observations
# -------------------------------
def find_crism_obs_list(data_root, suffix="_ATP_IRONLY.img", sort=True):
    obs_list = [f for f in os.listdir(data_root) if f.endswith(suffix)]
    return sorted(obs_list) if sort else obs_list

OBS_LIST = find_crism_obs_list(DATA_ROOT)

print(f"[INFO] Found {len(OBS_LIST)} ATP observations")
for f in OBS_LIST[:5]:
    print(" ", f)
# Use all observation types: frt, frs, hrl, hrs
VALID_PREFIXES = ("frt", "frs", "hrl", "hrs")
OBS_LIST = [f for f in OBS_LIST if f.startswith(VALID_PREFIXES)]
print(f"[INFO] Using all products (FRT+FRS+HRL+HRS): {len(OBS_LIST)} observations")

# -------------------------------
# CRISM IR wavelength table
# -------------------------------
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

def idx(w):
    return int(np.searchsorted(BANDS, w, side="left"))

# -------------------------------
# Spectral groups (7)
# -------------------------------
GROUPS_UM = [
    (1.02, 1.35),
    (1.35, 1.75),
    (1.75, 2.05),
    (2.05, 2.35),
    (2.35, 2.65),
    (2.81, 3.10),
    (3.10, 3.48),
]

GROUPS_IDX = [(idx(a), idx(b)) for a, b in GROUPS_UM]

# ============================================================
# Custom ATP_IRONLY binary loader
# ============================================================
def load_atp_ironly_cube(atp_path, trr_lbl, ir_count=350):
    meta = parse_pds3_label(trr_lbl)
    cols = int(meta["LINE_SAMPLES"])

    raw = np.fromfile(atp_path, dtype=np.float32)

    if raw.size % (cols * ir_count) != 0:
        raise ValueError(
            f"ATP raw size not divisible: {raw.size} vs cols*bands"
        )

    rows = raw.size // (cols * ir_count)

    cube_lbc = raw.reshape(rows, ir_count, cols)
    cube_rcb = np.transpose(cube_lbc, (0, 2, 1))
    return cube_rcb, rows

def load_label(path):
    return np.load(path)

# ============================================================
# FIX #3: Filter FILL_VALUE pixels
# ============================================================
def extract_labeled_pixels(cube, label_map, fill_value=FILL_VALUE):
    """Extract valid labeled pixels, filtering out FILL_VALUE and non-finite values."""
    # Check first band for fill values and non-finite
    valid_mask = (
        (label_map != IGNORE_LABEL) &
        np.isfinite(cube[..., 0]) &
        (cube[..., 0] != fill_value)
    )
    # Also check that no band has fill value or non-finite
    for b in range(cube.shape[-1]):
        valid_mask &= np.isfinite(cube[..., b]) & (cube[..., b] != fill_value)

    return cube[valid_mask], label_map[valid_mask]

def align_label_to_cube(label, cube_rows, cube_cols):
    """
    Align label to cube dimensions, handling:
    - Row mismatch (crop center or skip if label too small)
    - Column mismatch (subsample for HRL/HRS half-resolution products)
    """
    label_rows, label_cols = label.shape

    # Handle column mismatch (HRL/HRS: 320 cols vs label 640 cols)
    if label_cols != cube_cols:
        if label_cols == cube_cols * 2:
            # Subsample: take every other column
            label = label[:, ::2]
            label_cols = label.shape[1]
        elif cube_cols == label_cols * 2:
            # Shouldn't happen, but handle it
            label = np.repeat(label, 2, axis=1)
            label_cols = label.shape[1]
        else:
            return None  # Can't fix column mismatch

    # Handle row mismatch
    if label_rows == cube_rows:
        return label
    elif label_rows > cube_rows:
        # Label has more rows - crop center
        start = (label_rows - cube_rows) // 2
        return label[start:start + cube_rows, :]
    else:
        # Label has fewer rows than cube - crop cube region that matches label
        # Return label as-is, caller will crop cube
        return label

# ============================================================
# Robust label matching
# ============================================================
def find_label_for_img(img_name, data_root):
    stem = img_name.split("_")[0]
    m = re.match(r"(frt|frs|hrl|hrs)([0-9a-fA-F]+)", stem)
    if m is None:
        return None, None

    obs_hex = m.group(2).upper().lstrip("0")

    candidates = glob.glob(
        os.path.join(data_root, f"im_*{obs_hex}_LABEL.npy")
    )

    if len(candidates) == 0:
        return obs_hex, None
    return obs_hex, candidates[0]

# ============================================================
# FIX #1: Load observations with observation ID tracking
# ============================================================
print("[INFO] Loading observations...")

obs_data = []  # List of (obs_id, X, y) tuples

for img_name in OBS_LIST:
    obs_id, label_path = find_label_for_img(img_name, DATA_ROOT)
    img_path = os.path.join(DATA_ROOT, img_name)

    if label_path is None:
        print(f"[WARN] Missing label -> skip {img_name}")
        continue

    trr_lbl = os.path.join(
        DATA_ROOT,
        img_name.replace("_ATP_IRONLY.img", ".lbl")
    )

    if not os.path.exists(trr_lbl):
        print(f"[WARN] Missing TRR lbl -> skip {img_name}")
        continue

    print(f"[INFO] Loading {img_name} (obs_id={obs_id})")

    labels_full = load_label(label_path)

    cube, atp_rows = load_atp_ironly_cube(img_path, trr_lbl)
    cube_rows, cube_cols = cube.shape[:2]

    # Align label to cube dimensions (handles HRL/HRS column mismatch)
    labels = align_label_to_cube(labels_full, cube_rows, cube_cols)

    if labels is None:
        print(f"[SKIP] Can't align label columns: {img_name}")
        continue

    # If label has fewer rows, crop cube to match
    if labels.shape[0] < cube_rows:
        label_rows = labels.shape[0]
        # Crop cube center to match label
        start = (cube_rows - label_rows) // 2
        cube = cube[start:start + label_rows, :, :]
        print(f"  [INFO] Cropped cube to {label_rows} rows to match label")

    if cube.shape[:2] != labels.shape:
        print(
            f"[SKIP] shape mismatch after alignment: "
            f"{img_name} | cube {cube.shape[:2]} vs label {labels.shape}"
        )
        continue

    X, y = extract_labeled_pixels(cube, labels)

    if len(X) > 0:
        obs_data.append((obs_id, X, y))
        print(f"  [INFO] Extracted {len(X)} valid pixels")

if len(obs_data) == 0:
    raise RuntimeError("No labeled samples loaded. Check label files.")

print(f"[INFO] Total observations loaded: {len(obs_data)}")

# ============================================================
# 2) Label remapping (per observation)
# ============================================================
WATER_UNRELATED = {5, 13, 30, 33, 34, 35, 36, 37}
KEEP_CLASSES = {
    1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12,
    14, 15, 16, 17, 18, 19, 23, 25, 26,
    27, 29, 31, 38, 39
}
NEW_WATER_UNRELATED_ID = 100

def remap_labels(X, y):
    """Remap labels: merge water-unrelated, keep specific classes."""
    X_new, y_new = [], []
    for xi, yi in zip(X, y):
        yi = int(yi)
        if yi in WATER_UNRELATED:
            X_new.append(xi)
            y_new.append(NEW_WATER_UNRELATED_ID)
        elif yi in KEEP_CLASSES:
            X_new.append(xi)
            y_new.append(yi)
    return np.asarray(X_new, dtype=np.float32), np.asarray(y_new, dtype=np.int64)

# Apply remapping to each observation
obs_data_remapped = []
for obs_id, X, y in obs_data:
    X_remap, y_remap = remap_labels(X, y)
    if len(X_remap) > 0:
        obs_data_remapped.append((obs_id, X_remap, y_remap))

obs_data = obs_data_remapped
print(f"[INFO] Observations after remapping: {len(obs_data)}")

# ============================================================
# FIX #1: Split by observation (no spatial leakage)
# ============================================================
print("[INFO] Splitting by observation (no spatial leakage)...")

np.random.seed(SEED)
obs_ids = [obs_id for obs_id, _, _ in obs_data]
n_obs = len(obs_ids)

# Shuffle observation indices
obs_indices = np.random.permutation(n_obs)

# Split: 80% train, 10% val, 10% test
n_train = int(0.8 * n_obs)
n_val = int(0.1 * n_obs)

train_obs_idx = obs_indices[:n_train]
val_obs_idx = obs_indices[n_train:n_train + n_val]
test_obs_idx = obs_indices[n_train + n_val:]

print(f"[INFO] Observation split: {len(train_obs_idx)} train, {len(val_obs_idx)} val, {len(test_obs_idx)} test")

# Collect pixels for each split
def collect_pixels(obs_indices):
    X_list, y_list = [], []
    for i in obs_indices:
        _, X, y = obs_data[i]
        X_list.append(X)
        y_list.append(y)
    if len(X_list) == 0:
        return np.array([]), np.array([])
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

X_tr, y_tr = collect_pixels(train_obs_idx)
X_va, y_va = collect_pixels(val_obs_idx)
X_te, y_te = collect_pixels(test_obs_idx)

print(f"[INFO] Pixel counts: {len(X_tr)} train, {len(X_va)} val, {len(X_te)} test")

# Save observation split info
split_info = {
    "train_obs": [obs_data[i][0] for i in train_obs_idx],
    "val_obs": [obs_data[i][0] for i in val_obs_idx],
    "test_obs": [obs_data[i][0] for i in test_obs_idx],
}
with open(os.path.join(OUT_DIR, "obs_split.json"), "w") as f:
    json.dump(split_info, f, indent=2)

# ============================================================
# 2.5) Remove small classes (based on training set)
# ============================================================
MIN_SAMPLES = 100

unique, counts = np.unique(y_tr, return_counts=True)
class_counts = dict(zip(unique, counts))

print("[INFO] Class sample counts in training set:")
for k, v in sorted(class_counts.items()):
    print(f"  class {k}: {v}")

valid_classes = {k for k, v in class_counts.items() if v >= MIN_SAMPLES}

def filter_classes(X, y, valid_classes):
    mask = np.isin(y, list(valid_classes))
    return X[mask], y[mask]

X_tr, y_tr = filter_classes(X_tr, y_tr, valid_classes)
X_va, y_va = filter_classes(X_va, y_va, valid_classes)
X_te, y_te = filter_classes(X_te, y_te, valid_classes)

print(f"[INFO] Kept {len(valid_classes)} classes after filtering")
print(f"[INFO] Remaining pixels: {len(X_tr)} train, {len(X_va)} val, {len(X_te)} test")

# Create class mapping
labels_unique = np.unique(y_tr)
class_map = {int(lbl): i for i, lbl in enumerate(labels_unique)}
inv_class_map = {i: int(lbl) for lbl, i in class_map.items()}

y_tr = np.array([class_map[int(v)] for v in y_tr])
y_va = np.array([class_map.get(int(v), -1) for v in y_va])
y_te = np.array([class_map.get(int(v), -1) for v in y_te])

# Remove samples with unmapped classes in val/test
valid_va = y_va >= 0
valid_te = y_te >= 0
X_va, y_va = X_va[valid_va], y_va[valid_va]
X_te, y_te = X_te[valid_te], y_te[valid_te]

n_classes = len(labels_unique)
print(f"[INFO] Final classes: {class_map}")
print(f"[INFO] Final pixels: {len(X_tr)} train, {len(X_va)} val, {len(X_te)} test")

with open(os.path.join(OUT_DIR, "class_map.json"), "w") as f:
    json.dump(class_map, f, indent=2)

with open(os.path.join(OUT_DIR, "inv_class_map.json"), "w") as f:
    json.dump(inv_class_map, f, indent=2)

# ============================================================
# 3) Spectral grouping
# ============================================================
def split_to_groups(X):
    return [X[:, s:e] for (s, e) in GROUPS_IDX]

Xg_tr = split_to_groups(X_tr)
Xg_va = split_to_groups(X_va)
Xg_te = split_to_groups(X_te)

# ============================================================
# FIX #2: Input normalization (per-branch, computed on training set)
# ============================================================
print("[INFO] Computing normalization statistics from training set...")

norm_stats = []
for i, g in enumerate(Xg_tr):
    mean = g.mean(axis=0, keepdims=True)
    std = g.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
    norm_stats.append((mean.flatten(), std.flatten()))
    print(f"  Branch {i}: mean range [{mean.min():.4f}, {mean.max():.4f}], std range [{std.min():.4f}, {std.max():.4f}]")

# Save normalization stats
norm_stats_save = {
    f"branch_{i}": {"mean": m.tolist(), "std": s.tolist()}
    for i, (m, s) in enumerate(norm_stats)
}
with open(os.path.join(OUT_DIR, "norm_stats.json"), "w") as f:
    json.dump(norm_stats_save, f, indent=2)

def normalize_groups(Xg, norm_stats):
    """Apply per-branch normalization."""
    return [(g - m) / s for g, (m, s) in zip(Xg, norm_stats)]

Xg_tr = normalize_groups(Xg_tr, norm_stats)
Xg_va = normalize_groups(Xg_va, norm_stats)
Xg_te = normalize_groups(Xg_te, norm_stats)

print("[INFO] Normalization applied to all splits")

# ============================================================
# 5) Dataset / Loader
# ============================================================
train_ds = TensorDataset(*[torch.from_numpy(g.astype(np.float32)) for g in Xg_tr], torch.from_numpy(y_tr))
val_ds = TensorDataset(*[torch.from_numpy(g.astype(np.float32)) for g in Xg_va], torch.from_numpy(y_va))
test_ds = TensorDataset(*[torch.from_numpy(g.astype(np.float32)) for g in Xg_te], torch.from_numpy(y_te))

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_VAL)
test_loader = DataLoader(test_ds, batch_size=BATCH_VAL)

# ============================================================
# Attention entropy helper
# ============================================================
def attention_entropy(w: torch.Tensor) -> torch.Tensor:
    # w: (B, n_branches), already softmaxed
    return -(w * torch.log(w + 1e-8)).sum(dim=1).mean()

# ============================================================
# 6) Model with numerical stability improvements
# ============================================================
class SpectralBranch(nn.Module):
    def __init__(self, out_ch=64, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool)
        )

    def forward(self, x):
        x = self.net(x.unsqueeze(1))   # (B, out_ch, pool)
        return x.flatten(1)            # (B, out_ch * pool)


class MultiBranchAttnCNN(nn.Module):
    def __init__(self, n_branches, n_classes, feat_ch=64, pool=2, att_temperature=1.0):
        super().__init__()

        feat_dim = feat_ch * pool

        self.branches = nn.ModuleList(
            [SpectralBranch(out_ch=feat_ch, pool=pool) for _ in range(n_branches)]
        )

        self.att_fc = nn.Sequential(
            nn.Linear(feat_dim * n_branches, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, n_branches)
        )

        self.cls = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

        self.att_temperature = float(att_temperature)

    def forward(self, *xg, return_attn=False):
        feats = [b(x) for b, x in zip(self.branches, xg)]
        cat = torch.cat(feats, dim=1)

        # FIX #4: Clamp logits before temperature scaling for numerical stability
        att_logits = self.att_fc(cat)
        att_logits = torch.clamp(att_logits, min=-20.0, max=20.0)
        w = torch.softmax(att_logits / self.att_temperature, dim=1)

        f = sum(w[:, i:i+1] * feats[i] for i in range(len(feats)))
        logits = self.cls(f)
        return (logits, w) if return_attn else logits


model = MultiBranchAttnCNN(len(GROUPS_UM), n_classes, att_temperature=ATT_TEMPERATURE).to(device)

# ============================================================
# FIX #5: Class weights computed from training set after observation split
# ============================================================
counts = np.bincount(y_tr, minlength=n_classes)
freq = counts / counts.sum()
weights = 1.0 / np.sqrt(np.clip(freq, 1e-12, None))
weights /= weights.mean()

print("[INFO] Class weights (inverse sqrt frequency):")
for i, w in enumerate(weights):
    print(f"  class {inv_class_map[i]}: weight={w:.3f}, count={counts[i]}")

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32, device=device)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ============================================================
# FIX #6: Learning rate scheduler
# ============================================================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# ============================================================
# 8) Training with gradient clipping
# ============================================================
best_val = 0.0
ent_hist = []
lr_hist = []

print("\n[INFO] Starting training...")

for epoch in range(EPOCHS):
    model.train()
    ent_accum = []

    for batch in train_loader:
        *xg, yb = [t.to(device) for t in batch]

        logits, w = model(*xg, return_attn=True)

        cls_loss = criterion(logits, yb)
        ent_loss = attention_entropy(w)

        # Maximize entropy -> subtract entropy term
        loss = cls_loss - ATT_ENT_LAMBDA * ent_loss

        optimizer.zero_grad()
        loss.backward()

        # FIX #4: Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()

        ent_accum.append(ent_loss.detach().item())

    ent_mean = float(np.mean(ent_accum)) if len(ent_accum) else float("nan")
    ent_hist.append(ent_mean)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            *xg, yb = [t.to(device) for t in batch]
            pred = model(*xg).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()

    acc = correct / total if total > 0 else 0.0
    current_lr = optimizer.param_groups[0]['lr']
    lr_hist.append(current_lr)

    print(f"Epoch {epoch+1:02d} | Val Acc: {acc:.4f} | Attn Ent: {ent_mean:.4f} | LR: {current_lr:.6f}")

    # FIX #6: Step scheduler
    scheduler.step(acc)

    if acc > best_val:
        best_val = acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))

print(f"\n[DONE] Training finished. Best Val Acc: {best_val:.4f}")

# Save training history
with open(os.path.join(OUT_DIR, "train_history.json"), "w") as f:
    json.dump({
        "temperature": ATT_TEMPERATURE,
        "entropy_lambda": ATT_ENT_LAMBDA,
        "epoch_mean_entropy": ent_hist,
        "epoch_lr": lr_hist,
        "best_val_acc": best_val,
        "theoretical_max_lnK": float(np.log(len(GROUPS_UM))),
    }, f, indent=2)

# ============================================================
# 9) POST ANALYSIS on TEST SET
# ============================================================
print("\n[INFO] Running post-analysis on TEST set")

model.load_state_dict(
    torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=device)
)
model.eval()

all_attn = []
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        *xg, yb = [t.to(device) for t in batch]

        logits, attn = model(*xg, return_attn=True)
        preds = logits.argmax(dim=1)

        all_attn.append(attn.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(yb.cpu().numpy())

attn_all = np.concatenate(all_attn, axis=0)
preds_all = np.concatenate(all_preds, axis=0)
labels_all = np.concatenate(all_labels, axis=0)

np.save(os.path.join(OUT_DIR, "test_attention.npy"), attn_all)
np.save(os.path.join(OUT_DIR, "test_preds.npy"), preds_all)
np.save(os.path.join(OUT_DIR, "test_labels.npy"), labels_all)

print("[INFO] Raw test attention / preds / labels saved")

# ============================================================
# 10) Compute test metrics
# ============================================================
test_acc = (preds_all == labels_all).mean()
print(f"\n[TEST] Accuracy: {test_acc:.4f}")

# Cohen's Kappa
n_total = len(labels_all)
p_o = test_acc

p_e = 0.0
for cls_idx in range(n_classes):
    prop_pred = (preds_all == cls_idx).sum() / n_total
    prop_true = (labels_all == cls_idx).sum() / n_total
    p_e += prop_pred * prop_true

kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0
print(f"[TEST] Cohen's Kappa: {kappa:.4f}")

# Save test metrics
test_metrics = {
    "accuracy": float(test_acc),
    "cohens_kappa": float(kappa),
    "n_test_samples": int(n_total),
    "n_classes": int(n_classes),
}
with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)

# ============================================================
# 11) Class-wise attention statistics
# ============================================================
print("\n[INFO] Computing class-wise attention usage")

class_attn_stats = {}

for class_idx in range(n_classes):
    mask = labels_all == class_idx
    n = mask.sum()

    if n == 0:
        continue

    mean_attn = attn_all[mask].mean(axis=0)
    mean_attn = mean_attn / mean_attn.sum()  # safety normalization

    class_name = str(inv_class_map[class_idx])

    class_attn_stats[class_name] = {
        "num_samples": int(n),
        "branches": {
            f"branch_{i}": float(mean_attn[i])
            for i in range(len(mean_attn))
        }
    }

    print(f"\n[CLASS {class_name}] (n={n})")
    for i, (a, b) in enumerate(GROUPS_UM):
        print(f"  Branch {i} ({a:.2f}-{b:.2f} um): {mean_attn[i]:.3f}")

with open(os.path.join(OUT_DIR, "classwise_attention.json"), "w") as f:
    json.dump(class_attn_stats, f, indent=2)

print("\n[INFO] Class-wise attention statistics saved")
print(f"\n[DONE] All outputs saved to: {OUT_DIR}")
