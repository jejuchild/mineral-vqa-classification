#!/usr/bin/env python3
"""
CRISM Mineral Classification v2 — 7-Branch Training, Inference & Analysis
=========================================================================

Features:
  - 7-branch spectral grouping (full 350-band IR coverage)
  - Continuum Removal + SNV preprocessing
  - Pre-Activation ResBlocks with dilated convolutions
  - Hierarchical taxonomy loss (group → subgroup → class)
  - Observation-wise train/val/test split (no scene leakage)
  - Per-scene inference with Bayesian spatial smoothing (MRF)
  - Branch attention analysis & pruning recommendations
  - NO SSL pretraining (supervised only)

Colab (auto-download, recommended):
  1. Copy this script to a Colab cell
  2. Runtime → Change runtime type → GPU (T4)
  3. Run: !python crism_v2_pipeline.py --phase all
  Data (247MB) is auto-downloaded from GitHub on first run.

Colab (manual Drive):
  1. Upload raw ATP cubes to Google Drive
  2. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
  3. Run: !python crism_v2_pipeline.py --phase all --data-root /content/drive/MyDrive/crism_ml

CLI (local, with raw cubes):
  python crism_v2_pipeline.py --phase all --data-root /path/to/crism_ml

CLI (local, with pre-extracted npz):
  python crism_v2_pipeline.py --phase all --npz-path crism_training_data_f16.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# CONFIGURATION — EDIT THESE
# ============================================================================

# --- Paths ---
DATA_ROOT = "/home/cspark/data/crism_ml"  # directory with ATP_IRONLY.img + LABEL.npy
NPZ_URL = "https://github.com/jejuchild/crism-mineral-data/releases/download/v1.0/crism_training_data_f16.npz"
NPZ_LOCAL = "crism_training_data_f16.npz"  # auto-downloaded in Colab
OUT_DIR = "./results_v2"

# --- Spectral Constants ---
N_BANDS = 350
BANDS = np.linspace(1.0213, 3.9206, N_BANDS).astype(np.float32)
FILL_VALUE = 65535.0
IGNORE_LABEL = -1

# --- 7-Branch Spectral Groups ---
GROUPS_IDX: List[Tuple[int, int]] = [
    (0, 83),    # B0  1.02–1.35 μm  Fe²⁺ crystal field
    (83, 116),  # B1  1.35–1.75 μm  H₂O 1.4 μm
    (116, 175), # B2  1.75–2.15 μm  Al-OH, Mg-OH
    (175, 248), # B3  2.15–2.65 μm  Metal-OH, CO₃
    (248, 296), # B4  2.65–3.10 μm  H₂O 3 μm broad
    (296, 320), # B5  3.10–3.48 μm  CO₃ ν₃
    (320, 350), # B6  3.48–3.92 μm  Thermal tail
]
N_BRANCHES = len(GROUPS_IDX)
BRANCH_LABELS = [
    f"B{i}: {BANDS[s]:.2f}–{BANDS[min(e-1,N_BANDS-1)]:.2f}μm"
    for i, (s, e) in enumerate(GROUPS_IDX)
]

# --- Observation Prefixes ---
VALID_PREFIXES = ("frt", "frs", "hrl", "hrs")

# --- Class Names (covers known CRISM mineral label IDs) ---
CLASS_NAME: Dict[int, str] = {
    1: "Nontronite", 2: "Fe smectite", 3: "Saponite",
    4: "Montmorillonite", 5: "Mg smectite", 6: "Vermiculite",
    7: "Fe/Mg smectite", 8: "Al smectite", 9: "Kaolinite",
    10: "Chlorite", 11: "Prehnite", 12: "Kieserite",
    13: "Polyhydr. sulfate", 14: "Gypsum", 15: "Bassanite",
    16: "Magnesite", 17: "Calcite", 18: "Hematite",
    19: "Goethite", 20: "LCP", 21: "HCP",
    22: "Olivine", 23: "Mg carbonate",
    24: "CO₂ ice", 25: "Fe/Mg clay", 26: "Pyroxene",
    27: "HCP (high-Ca)", 28: "Zeolite", 29: "Sulfate mix",
    30: "Serpentine", 31: "Pyroxene (low-Ca)",
    38: "Olivine (Fo-rich)", 100: "Water-unrelated",
}

# --- Mineral Taxonomy (for Hierarchical Loss) ---
# Maps original class ID → (group_name, subgroup_name)
TAXONOMY: Dict[int, Tuple[str, str]] = {
    1: ("Phyllosilicate", "Fe_Mg_phyllo"), 2: ("Phyllosilicate", "Fe_Mg_phyllo"),
    3: ("Phyllosilicate", "Fe_Mg_phyllo"), 5: ("Phyllosilicate", "Fe_Mg_phyllo"),
    6: ("Phyllosilicate", "Fe_Mg_phyllo"), 7: ("Phyllosilicate", "Fe_Mg_phyllo"),
    25: ("Phyllosilicate", "Fe_Mg_phyllo"),
    4: ("Phyllosilicate", "Al_phyllo"), 8: ("Phyllosilicate", "Al_phyllo"),
    9: ("Phyllosilicate", "Al_phyllo"),
    10: ("Phyllosilicate", "Chlorite"), 11: ("Phyllosilicate", "Mica_Prehnite"),
    30: ("Phyllosilicate", "Serpentine"),
    12: ("Sulfate", "Monohydrated"), 13: ("Sulfate", "Polyhydrated"),
    14: ("Sulfate", "Polyhydrated"), 15: ("Sulfate", "Polyhydrated"),
    29: ("Sulfate", "Polyhydrated"),
    16: ("Carbonate", "Mg_Ca_carbonate"), 17: ("Carbonate", "Mg_Ca_carbonate"),
    23: ("Carbonate", "Mg_Ca_carbonate"),
    18: ("Oxide", "Fe_oxide"), 19: ("Oxide", "Fe_oxide"),
    20: ("Pyroxene_Olivine", "LCP_HCP"), 21: ("Pyroxene_Olivine", "LCP_HCP"),
    26: ("Pyroxene_Olivine", "LCP_HCP"), 27: ("Pyroxene_Olivine", "LCP_HCP"),
    31: ("Pyroxene_Olivine", "LCP_HCP"),
    22: ("Pyroxene_Olivine", "Olivine"), 38: ("Pyroxene_Olivine", "Olivine"),
    24: ("Other", "Ice"), 28: ("Other", "Zeolite"), 100: ("Other", "Unclassified"),
}

# --- Training Hyperparameters ---
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_train: int = 512
    batch_val: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    att_temperature: float = 0.5
    att_ent_lambda: float = 0.01
    min_samples: int = 30
    seed: int = 42
    device: str = "auto"
    # Spatial smoothing
    spatial_beta: float = 0.5
    spatial_kernel: int = 5
    spatial_iters: int = 3
    # Architecture
    use_resblock: bool = True
    dropout_conv: float = 0.3
    dropout_fc: float = 0.5
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 1])
    # Loss
    hierarchical: bool = True
    hierarchy_weights: Tuple[float, ...] = (0.25, 0.35, 0.40)


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# PREPROCESSING: Fill Interpolation + Continuum Removal + SNV
# ============================================================================

def _interpolate_fill(spectra: np.ndarray) -> np.ndarray:
    """Replace fill bands (exactly 0.0) with linearly interpolated values.

    CRISM ATP cubes have FILL_VALUE=65535 in certain bands:
      - Bands 85-90 (~1.73-1.77μm): detector join region (62-98% fill)
      - Bands 348-349 (~3.91-3.92μm): detector edge (100% fill)
    During extraction these were zeroed to 0.0. We must interpolate them
    from neighboring valid bands BEFORE CR and SNV, otherwise:
      - CR: zeros survive as ~0 after hull normalization
      - SNV: mean is pulled down, std inflated → all z-scores corrupted
    """
    result = spectra.astype(np.float32, copy=True)
    fill = (result == 0.0)
    has_fill = fill.any(axis=1)

    if not has_fill.any():
        return result

    bands = np.arange(result.shape[1], dtype=np.float32)
    indices = np.where(has_fill)[0]
    step = max(len(indices) // 5, 1)

    for count, i in enumerate(indices):
        valid = ~fill[i]
        n_valid = valid.sum()
        if n_valid >= 2:
            result[i, fill[i]] = np.interp(
                bands[fill[i]], bands[valid], result[i, valid]
            )
        elif n_valid == 1:
            result[i, fill[i]] = result[i, valid][0]
        # else: all fill → leave as 0, will be filtered by min_samples later
        if (count + 1) % step == 0:
            _log(f"  Fill interpolation: {count+1:,}/{len(indices):,}")

    _log(f"  Interpolated fill bands in {len(indices):,}/{spectra.shape[0]:,} pixels")
    return result


def _upper_hull_cr(wl: np.ndarray, spec: np.ndarray) -> np.ndarray:
    """Continuum removal via upper convex hull for one spectrum."""
    n = len(wl)
    if n < 3:
        return spec.copy()
    floor = spec.min()
    shifted = (spec - floor + 1e-6) if floor <= 0 else spec.copy()
    hull = [0]
    for i in range(1, n):
        while len(hull) >= 2:
            j, k = hull[-1], hull[-2]
            dw = wl[i] - wl[k]
            if dw < 1e-10:
                hull.pop()
                continue
            slope = (shifted[i] - shifted[k]) / dw
            if shifted[j] <= shifted[k] + slope * (wl[j] - wl[k]) + 1e-10:
                hull.pop()
            else:
                break
        hull.append(i)
    hull_arr = np.array(hull)
    continuum = np.interp(wl, wl[hull_arr], shifted[hull_arr])
    return shifted / np.maximum(continuum, 1e-10)


def continuum_removal(spectra: np.ndarray, wl: np.ndarray = BANDS) -> np.ndarray:
    n = spectra.shape[0]
    cr = np.empty_like(spectra, dtype=np.float32)
    step = max(n // 10, 1)
    for i in range(n):
        cr[i] = _upper_hull_cr(wl, spectra[i])
        if (i + 1) % step == 0:
            _log(f"  CR progress: {i+1}/{n}")
    return np.clip(cr, 0.0, 2.0).astype(np.float32)


def snv(spectra: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: per-spectrum z-score."""
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    return ((spectra - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def preprocess(spectra: np.ndarray) -> np.ndarray:
    """Interpolate fills → clip(0,1) → CR → SNV"""
    result = _interpolate_fill(spectra)
    result = np.clip(result, 0.0, 1.0)
    result = continuum_removal(result)
    return result


# ============================================================================
# DATA LOADING
# ============================================================================

def download_npz(url: str = NPZ_URL, local: str = NPZ_LOCAL) -> str:
    """Download pre-extracted training data from GitHub release."""
    if os.path.exists(local):
        _log(f"Data already exists: {local}")
        return local
    import urllib.request
    _log(f"Downloading {url}...")
    _log("  (This is ~247 MB, may take a few minutes)")
    urllib.request.urlretrieve(url, local)
    _log(f"  Saved to {local}")
    return local


def load_from_npz(path: str) -> List[Dict[str, Any]]:
    """Load pre-extracted data from .npz and create per-observation dicts."""
    _log(f"Loading from {path}...")
    d = np.load(path, allow_pickle=False)
    X = d["X"].astype(np.float32)  # upcast float16 → float32
    y = d["y"].astype(np.int64)
    obs_idx = d["obs_idx"].astype(np.int32)
    obs_ids = d["obs_ids"]
    _log(f"  {X.shape[0]:,} pixels, {len(obs_ids)} observations")

    observations: List[Dict[str, Any]] = []
    for i, oid in enumerate(obs_ids):
        mask = obs_idx == i
        if not mask.any():
            continue
        observations.append({
            "obs_id": str(oid),
            "X": X[mask],
            "y": y[mask],
            "rows": np.zeros(mask.sum(), dtype=np.int32),  # placeholder
            "cols": np.zeros(mask.sum(), dtype=np.int32),  # placeholder
            "cube_shape": (480, 640),  # standard CRISM cube shape
            "cube_path": "",  # not available from npz
            "lbl_path": "",
        })
    _log(f"  Created {len(observations)} observation dicts")
    return observations

def _parse_pds3(path: str) -> Dict[str, str]:
    meta = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("/*") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta


def _load_cube(atp: str, lbl: str) -> Tuple[np.ndarray, int, int]:
    meta = _parse_pds3(lbl)
    cols = int(meta["LINE_SAMPLES"])
    raw = np.fromfile(atp, dtype=np.float32)
    block = cols * N_BANDS
    if raw.size % block != 0:
        raise ValueError(f"ATP size {raw.size} not divisible by {block}")
    rows = raw.size // block
    cube = np.transpose(raw.reshape(rows, N_BANDS, cols), (0, 2, 1))  # (R, C, B)
    return cube, rows, cols


def _find_label(img_name: str, data_root: str) -> Tuple[Optional[str], Optional[str]]:
    stem = img_name.split("_")[0]
    m = re.match(r"(frt|frs|hrl|hrs)([0-9a-fA-F]+)", stem)
    if m is None:
        return None, None
    hex_id = m.group(2).upper().lstrip("0")
    cands = sorted(glob.glob(os.path.join(data_root, f"im_*{hex_id}_LABEL.npy")))
    return hex_id, (cands[0] if cands else None)


def _align(label: np.ndarray, cr: int, cc: int) -> Optional[np.ndarray]:
    lr, lc = label.shape[:2]
    if lc != cc:
        if lc == cc * 2:
            label = label[:, ::2]
        elif cc == lc * 2:
            label = np.repeat(label, 2, axis=1)
        else:
            return None
    lr = label.shape[0]
    if lr == cr:
        return label
    if lr > cr:
        s = (lr - cr) // 2
        return label[s:s + cr]
    return label


def load_observations(data_root: str) -> List[Dict[str, Any]]:
    """Load all ATP_IRONLY observations with matched labels."""
    files = sorted(glob.glob(os.path.join(data_root, "*_ATP_IRONLY.img")))
    files = [f for f in files if os.path.basename(f).startswith(VALID_PREFIXES)]
    observations: List[Dict[str, Any]] = []

    for img_path in files:
        name = os.path.basename(img_path)
        obs_id, label_path = _find_label(name, data_root)
        if obs_id is None or label_path is None:
            continue
        lbl_path = os.path.join(data_root, name.replace("_ATP_IRONLY.img", ".lbl"))
        if not os.path.exists(lbl_path):
            continue
        try:
            cube, cr, cc = _load_cube(img_path, lbl_path)
        except Exception as e:
            _log(f"  Skip {name}: {e}")
            continue
        label_full = np.load(label_path)
        label = _align(label_full, cr, cc)
        if label is None:
            continue

        # Crop cube if label is shorter
        if label.shape[0] < cr:
            off = (cr - label.shape[0]) // 2
            cube = cube[off:off + label.shape[0]]
        if cube.shape[:2] != label.shape[:2]:
            continue

        # Zero-out fill bands instead of rejecting entire pixels
        fill_mask = (cube == FILL_VALUE)
        cube[fill_mask] = 0.0
        n_valid_bands = (~fill_mask).sum(axis=-1)

        # Valid mask: labeled + finite + enough valid bands
        valid = (
            (label != IGNORE_LABEL)
            & np.all(np.isfinite(cube), axis=-1)
            & (n_valid_bands >= 250)
        )
        if not valid.any():
            continue

        X = cube[valid].astype(np.float32)
        y = label[valid].astype(np.int64)
        rows_arr, cols_arr = np.where(valid)

        observations.append({
            "obs_id": obs_id, "X": X, "y": y,
            "rows": rows_arr.astype(np.int32),
            "cols": cols_arr.astype(np.int32),
            "cube_shape": cube.shape[:2],
            "cube_path": img_path, "lbl_path": lbl_path,
        })
        _log(f"  Loaded {obs_id}: {X.shape[0]} pixels")

    return observations


def obs_wise_split(
    obs: List[Dict], train_f: float = 0.8, val_f: float = 0.1, seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    n = len(obs)
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=int)
    rng.shuffle(idx)
    nt = max(int(train_f * n), 1)
    nv = max(int(val_f * n), 1)
    return idx[:nt].tolist(), idx[nt:nt + nv].tolist(), idx[nt + nv:].tolist()


def collect_split(obs: List[Dict], indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    Xs = [obs[i]["X"] for i in indices if obs[i]["X"].shape[0] > 0]
    ys = [obs[i]["y"] for i in indices if obs[i]["y"].shape[0] > 0]
    if not Xs:
        return np.empty((0, N_BANDS), np.float32), np.empty((0,), np.int64)
    return np.concatenate(Xs), np.concatenate(ys)


def build_class_map(y: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    labels = sorted(set(int(v) for v in np.unique(y)))
    fwd = {l: i for i, l in enumerate(labels)}
    inv = {i: l for l, i in fwd.items()}
    return fwd, inv


def split_groups(X: np.ndarray) -> List[np.ndarray]:
    return [X[:, s:e].astype(np.float32) for s, e in GROUPS_IDX]


def normalize_groups(
    train_g: List[np.ndarray],
    val_g: Optional[List[np.ndarray]] = None,
    test_g: Optional[List[np.ndarray]] = None,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], Dict]:
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    t_out, v_out, te_out = [], [] if val_g else None, [] if test_g else None
    for i, g in enumerate(train_g):
        m = g.mean(axis=0).astype(np.float32)
        s = np.maximum(g.std(axis=0).astype(np.float32), 1e-8)
        stats[f"branch_{i}"] = {"mean": m, "std": s}
        t_out.append(((g - m) / s).astype(np.float32))
        if val_g is not None and v_out is not None:
            v_out.append(((val_g[i] - m) / s).astype(np.float32))
        if test_g is not None and te_out is not None:
            te_out.append(((test_g[i] - m) / s).astype(np.float32))
    return t_out, v_out, te_out, stats


# ============================================================================
# MODEL
# ============================================================================

class PreActResBlock(nn.Module):
    """Pre-activation residual block: BN→ReLU→Conv→BN→ReLU→Conv + skip."""
    def __init__(self, ci: int, co: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(ci)
        self.conv1 = nn.Conv1d(ci, co, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(co)
        self.conv2 = nn.Conv1d(co, co, 3, padding=1)
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.drop(out)
        out = self.conv2(out)
        return out + identity


class SpectralBranch(nn.Module):
    def __init__(self, use_res: bool, dilations: List[int], drop_conv: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
        )
        if use_res:
            cfgs = [(32, 64), (64, 128), (128, 128), (128, 128)]
            blocks = []
            for j, (ci, co) in enumerate(cfgs):
                d = dilations[j] if j < len(dilations) else 1
                blocks.append(PreActResBlock(ci, co, dilation=d, dropout=drop_conv))
            self.body = nn.Sequential(*blocks)
        else:
            self.body = nn.Sequential(
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(self.stem(x.unsqueeze(1)))).squeeze(-1)


class MultiBranchAttnCNN(nn.Module):
    def __init__(self, n_branches: int, n_classes: int, cfg: TrainConfig):
        super().__init__()
        self.n_branches = n_branches
        self.branches = nn.ModuleList([
            SpectralBranch(cfg.use_resblock, cfg.dilations, cfg.dropout_conv)
            for _ in range(n_branches)
        ])
        F_DIM = 128
        self.attn_fc1 = nn.Linear(n_branches * F_DIM, F_DIM)
        self.attn_fc2 = nn.Linear(F_DIM, n_branches)
        self.temperature = cfg.att_temperature
        self.dropout = nn.Dropout(cfg.dropout_fc)
        self.classifier = nn.Linear(F_DIM, n_classes)

    def forward(self, *branch_inputs: torch.Tensor, return_attn: bool = False):
        feats = [b(x) for b, x in zip(self.branches, branch_inputs)]
        stacked = torch.stack(feats, dim=1)                     # (B, nb, 128)
        cat = stacked.reshape(stacked.size(0), -1)              # (B, nb*128)
        a = torch.tanh(self.attn_fc1(cat))
        a = self.attn_fc2(a) / max(self.temperature, 1e-6)
        w = F.softmax(a, dim=1)                                 # (B, nb)
        fused = (stacked * w.unsqueeze(-1)).sum(dim=1)           # (B, 128)
        logits = self.classifier(self.dropout(fused))
        return (logits, w) if return_attn else logits


def attention_entropy(w: torch.Tensor) -> torch.Tensor:
    return -(w * torch.log(w + 1e-8)).sum(dim=1).mean()


# ============================================================================
# HIERARCHICAL LOSS
# ============================================================================

class HierarchicalLoss(nn.Module):
    def __init__(
        self, tax_map: Dict[int, Tuple[int, int]],
        weights: Tuple[float, ...], class_w: Optional[torch.Tensor], device: str,
    ):
        super().__init__()
        self.w = weights
        mc = max(tax_map.keys()) + 1
        gm = torch.zeros(mc, dtype=torch.long)
        sm = torch.zeros(mc, dtype=torch.long)
        for c, (g, s) in tax_map.items():
            gm[c] = g; sm[c] = s
        self.register_buffer("gm", gm)
        self.register_buffer("sm", sm)
        ng = int(gm.max().item()) + 1
        ns = int(sm.max().item()) + 1
        self.ce_cls = nn.CrossEntropyLoss(weight=class_w)
        self.ce_g = nn.CrossEntropyLoss()
        self.ce_s = nn.CrossEntropyLoss()
        self.head_g = nn.Linear(mc, ng).to(device)
        self.head_s = nn.Linear(mc, ns).to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_cls = self.ce_cls(logits, targets)
        l_g = self.ce_g(self.head_g(logits), self.gm[targets])
        l_s = self.ce_s(self.head_s(logits), self.sm[targets])
        return self.w[0] * l_g + self.w[1] * l_s + self.w[2] * l_cls


def _build_taxonomy_mapping(class_map: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    """Map mapped class index → (group_idx, subgroup_idx)."""
    all_groups: List[str] = sorted(set(g for g, _ in TAXONOMY.values()))
    all_subs: List[str] = sorted(set(s for _, s in TAXONOMY.values()))
    g2i = {n: i for i, n in enumerate(all_groups)}
    s2i = {n: i for i, n in enumerate(all_subs)}
    # Add fallback
    fb_g = len(all_groups)
    fb_s = len(all_subs)
    mapping: Dict[int, Tuple[int, int]] = {}
    for orig_id, mapped_idx in class_map.items():
        if orig_id in TAXONOMY:
            gn, sn = TAXONOMY[orig_id]
            mapping[mapped_idx] = (g2i[gn], s2i[sn])
        else:
            mapping[mapped_idx] = (fb_g, fb_s)
    return mapping


# ============================================================================
# TRAINING
# ============================================================================

def _class_weights(y: np.ndarray, nc: int, dev: torch.device) -> torch.Tensor:
    c = np.bincount(y, minlength=nc).astype(np.float64)
    f = c / max(c.sum(), 1.0)
    w = 1.0 / np.sqrt(np.clip(f, 1e-12, None))
    w /= w.mean()
    return torch.tensor(w, dtype=torch.float32, device=dev)


def train_model(
    model: MultiBranchAttnCNN, train_ld: DataLoader, val_ld: DataLoader,
    n_classes: int, cfg: TrainConfig, class_map: Dict[int, int],
    out_dir: str, device: torch.device,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)

    # Collect train labels
    all_y = np.concatenate([b[-1].numpy() for b in train_ld])
    cw = _class_weights(all_y, n_classes, device)

    # Criterion
    if cfg.hierarchical:
        tax = _build_taxonomy_mapping(class_map)
        criterion = HierarchicalLoss(tax, cfg.hierarchy_weights, cw, str(device)).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=5)

    hist: Dict[str, List[float]] = {"val_acc": [], "loss": [], "entropy": [], "lr": []}
    best = 0.0

    for ep in range(cfg.epochs):
        model.train()
        losses, ents = [], []
        for batch in train_ld:
            *xg, yb = [t.to(device) for t in batch]
            logits, w = model(*xg, return_attn=True)
            loss = criterion(logits, yb) - cfg.att_ent_lambda * attention_entropy(w)
            optim.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()
            losses.append(loss.item())
            ents.append(attention_entropy(w).item())

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_ld:
                *xg, yb = [t.to(device) for t in batch]
                p = model(*xg).argmax(1)
                correct += (p == yb).sum().item()
                total += yb.numel()
        va = correct / max(total, 1)
        sched.step(va)
        lr = max(pg["lr"] for pg in optim.param_groups)

        hist["val_acc"].append(va)
        hist["loss"].append(float(np.mean(losses)))
        hist["entropy"].append(float(np.mean(ents)))
        hist["lr"].append(lr)

        if va > best:
            best = va
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

        _log(f"  Ep {ep+1}/{cfg.epochs}: loss={np.mean(losses):.4f} "
             f"val={va:.4f} ent={np.mean(ents):.4f} lr={lr:.2e}")

    hist["best_val_acc"] = [best]
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    return hist


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_pixels(
    model: nn.Module, loader: DataLoader, inv_map: Dict[int, int], device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    P, L, A, PR = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            *xg, yb = [t.to(device) for t in batch]
            logits, w = model(*xg, return_attn=True)
            P.append(logits.argmax(1).cpu().numpy())
            L.append(yb.cpu().numpy())
            A.append(w.cpu().numpy())
            PR.append(F.softmax(logits, dim=1).cpu().numpy())
    preds, labels = np.concatenate(P), np.concatenate(L)
    attn, probs = np.concatenate(A), np.concatenate(PR)
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cids = sorted(inv_map.keys())
    names = [CLASS_NAME.get(inv_map[c], str(inv_map[c])) for c in cids]
    report = classification_report(
        labels, preds, labels=cids, target_names=names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=cids)
    return dict(
        accuracy=acc, kappa=kappa, predictions=preds, labels=labels,
        attention=attn, probabilities=probs, report=report,
        confusion_matrix=cm, class_ids=cids, class_names=names,
    )


# ============================================================================
# SPATIAL SMOOTHING + PER-SCENE INFERENCE
# ============================================================================

def spatial_smooth(
    prob: np.ndarray, beta: float = 0.5, kernel: int = 5, n_iter: int = 3,
) -> np.ndarray:
    """Bayesian MRF smoothing on (H, W, C) probability map."""
    sm = prob.copy()
    for _ in range(n_iter):
        lbl = sm.argmax(axis=-1)
        for c in range(sm.shape[2]):
            agree = uniform_filter((lbl == c).astype(np.float64), size=kernel, mode="reflect")
            sm[:, :, c] *= np.exp(beta * agree)
        sm /= np.maximum(sm.sum(axis=-1, keepdims=True), 1e-10)
    return sm


def evaluate_per_scene(
    model: nn.Module, observations: List[Dict], test_idx: List[int],
    class_map: Dict[int, int], norm_stats: Dict, cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Per-scene evaluation: raw pixels + spatial smoothing."""
    model.eval()
    results: Dict[str, Any] = {}

    for idx in test_idx:
        obs = observations[idx]
        oid = obs["obs_id"]
        try:
            cube, _, _ = _load_cube(obs["cube_path"], obs["lbl_path"])
        except Exception:
            continue
        H, W = obs["cube_shape"]
        if cube.shape[0] > H:
            cube = cube[:H]

        fill_mask_scene = (cube == FILL_VALUE)
        cube[fill_mask_scene] = 0.0
        n_valid_bands_scene = (~fill_mask_scene).sum(axis=-1)
        valid = np.all(np.isfinite(cube), axis=-1) & (n_valid_bands_scene >= 250)
        vr, vc = np.where(valid)
        if len(vr) == 0:
            continue

        # Preprocess (same pipeline as training)
        X = _interpolate_fill(cube[valid].astype(np.float32))
        X = np.clip(X, 0.0, 1.0)
        X = continuum_removal(X)
        X = snv(X)
        groups = split_groups(X)
        norm_g = []
        for i, g in enumerate(groups):
            k = f"branch_{i}"
            m, s = norm_stats[k]["mean"], np.maximum(norm_stats[k]["std"], 1e-8)
            norm_g.append(((g - m) / s).astype(np.float32))

        # Predict
        n = X.shape[0]
        all_probs, all_attn = [], []
        with torch.no_grad():
            for st in range(0, n, cfg.batch_val):
                en = min(st + cfg.batch_val, n)
                xg = [torch.from_numpy(g[st:en]).to(device) for g in norm_g]
                logits, w = model(*xg, return_attn=True)
                all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
                all_attn.append(w.cpu().numpy())
        probs = np.concatenate(all_probs)
        attn = np.concatenate(all_attn)
        nc = probs.shape[1]

        # Reconstruct & smooth
        pmap = np.zeros((H, W, nc), dtype=np.float64)
        for i, (r, c) in enumerate(zip(vr, vc)):
            if r < H and c < W:
                pmap[r, c] = probs[i]
        pmap_sm = spatial_smooth(pmap, cfg.spatial_beta, cfg.spatial_kernel, cfg.spatial_iters)
        lbl_raw = pmap.argmax(axis=-1)
        lbl_sm = pmap_sm.argmax(axis=-1)

        # Evaluate on labeled pixels
        y_true = np.array([class_map.get(int(v), -1) for v in obs["y"]])
        keep = y_true >= 0
        if keep.sum() == 0:
            continue
        or_, oc_ = obs["rows"][keep], obs["cols"][keep]
        yt = y_true[keep]
        yp_raw = lbl_raw[or_, oc_]
        yp_sm = lbl_sm[or_, oc_]
        ar = float(accuracy_score(yt, yp_raw))
        a_s = float(accuracy_score(yt, yp_sm))

        results[oid] = {
            "acc_raw": ar, "acc_smooth": a_s, "delta": a_s - ar,
            "n_labeled": int(keep.sum()), "n_valid": int(valid.sum()),
            "attn_mean": attn.mean(axis=0).tolist(),
        }
        _log(f"  Scene {oid}: raw={ar:.4f} smooth={a_s:.4f} Δ={a_s-ar:+.4f} n={keep.sum()}")

    # Summary
    if results:
        raws = [v["acc_raw"] for v in results.values()]
        sms = [v["acc_smooth"] for v in results.values()]
        summary = {
            "mean_raw": float(np.mean(raws)), "mean_smooth": float(np.mean(sms)),
            "std_raw": float(np.std(raws)), "std_smooth": float(np.std(sms)),
            "delta": float(np.mean(sms) - np.mean(raws)),
            "n_scenes": len(results),
        }
    else:
        summary = {}
    return {"per_scene": results, "summary": summary}


# ============================================================================
# ATTENTION ANALYSIS & PRUNING
# ============================================================================

def analyze_attention(
    ev: Dict[str, Any], inv_map: Dict[int, int], out_dir: str,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    attn = ev["attention"]   # (N, nb)
    labels = ev["labels"]
    nb = attn.shape[1]

    # --- Global stats ---
    gstats = {}
    for i in range(nb):
        a = attn[:, i]
        gstats[BRANCH_LABELS[i]] = {
            "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)),
        }

    # --- Per-class × per-branch heatmap ---
    cids = sorted(set(labels.tolist()))
    class_attn = np.zeros((len(cids), nb))
    cnames = []
    for ci, c in enumerate(cids):
        mask = labels == c
        class_attn[ci] = attn[mask].mean(axis=0)
        orig = inv_map.get(c, c)
        cnames.append(CLASS_NAME.get(orig, str(orig)))

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(cids) * 0.45)))
    sns.heatmap(
        class_attn, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=BRANCH_LABELS, yticklabels=cnames, ax=ax,
    )
    ax.set_title("Per-Class × Per-Branch Attention Weights")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attention_heatmap.png"), dpi=150)
    plt.close()

    # Distribution
    fig, axes = plt.subplots(1, nb, figsize=(3 * nb, 4), sharey=True)
    if nb == 1:
        axes = [axes]
    for i in range(nb):
        axes[i].hist(attn[:, i], bins=50, alpha=0.7, color=f"C{i}")
        axes[i].axvline(attn[:, i].mean(), color="red", ls="--", lw=1)
        axes[i].set_title(BRANCH_LABELS[i], fontsize=8)
        axes[i].set_xlabel("Weight")
    axes[0].set_ylabel("Count")
    plt.suptitle("Attention Weight Distribution per Branch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attention_dist.png"), dpi=150)
    plt.close()

    # Bar chart
    means = [gstats[BRANCH_LABELS[i]]["mean"] for i in range(nb)]
    stds = [gstats[BRANCH_LABELS[i]]["std"] for i in range(nb)]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d32f2f" if m < 0.05 else "#fb8c00" if m < 1.0 / nb * 0.7 else "#1976d2"
              for m in means]
    ax.bar(BRANCH_LABELS, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.axhline(1.0 / nb, color="gray", ls="--", label=f"Uniform ({1/nb:.3f})")
    ax.axhline(0.05, color="red", ls=":", label="Prune threshold (0.05)")
    ax.set_ylabel("Mean Attention")
    ax.set_title("Branch Importance")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "attention_bar.png"), dpi=150)
    plt.close()

    # Pruning recommendations
    pruning = []
    unif = 1.0 / nb
    for i in range(nb):
        m = means[i]
        if m < 0.05:
            verdict = "PRUNE"
        elif m < unif * 0.7:
            verdict = "LOW"
        else:
            verdict = "KEEP"
        pruning.append({"branch": i, "label": BRANCH_LABELS[i],
                        "mean": m, "verdict": verdict})

    analysis = {"global_stats": gstats, "pruning": pruning}
    with open(os.path.join(out_dir, "attention_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # Print
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS — PRUNING RECOMMENDATIONS")
    print("=" * 60)
    icons = {"PRUNE": "🔴", "LOW": "🟡", "KEEP": "🟢"}
    for p in pruning:
        print(f"  {icons[p['verdict']]} {p['label']}: mean={p['mean']:.4f} → {p['verdict']}")
    n_prune = sum(1 for p in pruning if p["verdict"] == "PRUNE")
    if n_prune:
        kept = [p["label"] for p in pruning if p["verdict"] != "PRUNE"]
        print(f"\n  → Prune {n_prune} branch(es). Next phase: {len(kept)} branches")
        print(f"  → Keep: {', '.join(kept)}")
    else:
        print("\n  → All branches contribute. No pruning recommended.")
    print("=" * 60)
    return analysis


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(cfg: TrainConfig, data_root: str, out_dir: str, phase: str = "all", npz_path: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else (cfg.device if cfg.device != "auto" else "cpu")
    )
    _log(f"Device: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    do_train = phase in ("all", "train")
    do_eval = phase in ("all", "evaluate", "analyze")
    do_analyze = phase in ("all", "analyze")

    # ---- DATA ----
    if npz_path or (IN_COLAB and not os.path.isdir(data_root)):
        # Use pre-extracted npz (Colab mode or explicit --npz-path)
        if not npz_path:
            npz_path = download_npz()
        obs = load_from_npz(npz_path)
    else:
        _log("Loading observations from raw ATP cubes...")
        obs = load_observations(data_root)
    _log(f"Total: {len(obs)} observations, {sum(o['X'].shape[0] for o in obs):,} pixels")

    _log("Obs-wise split (80/10/10)...")
    tr_i, va_i, te_i = obs_wise_split(obs, seed=cfg.seed)
    _log(f"Split: train={len(tr_i)} val={len(va_i)} test={len(te_i)} observations")

    X_tr, y_tr = collect_split(obs, tr_i)
    X_va, y_va = collect_split(obs, va_i)
    X_te, y_te = collect_split(obs, te_i)
    _log(f"Pixels: train={len(y_tr):,} val={len(y_va):,} test={len(y_te):,}")

    # ---- FILTER KNOWN CLASSES ----
    valid_cls = set(TAXONOMY.keys())
    _log(f"Filtering to {len(valid_cls)} known mineral classes (removing undefined 32-39 etc.)...")
    for name, arr_x, arr_y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
        n_before = len(arr_y)
        n_valid = np.isin(arr_y, list(valid_cls)).sum()
        _log(f"  {name}: {n_before:,} → {n_valid:,} ({n_before - n_valid:,} removed)")
    m_tr = np.isin(y_tr, list(valid_cls))
    m_va = np.isin(y_va, list(valid_cls))
    m_te = np.isin(y_te, list(valid_cls))
    X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
    X_va, y_va = X_va[m_va], y_va[m_va]
    X_te, y_te = X_te[m_te], y_te[m_te]
    _log(f"After filter: train={len(y_tr):,} val={len(y_va):,} test={len(y_te):,}")

    # ---- PREPROCESS ----
    _log("Preprocessing (interpolate fills → clip → CR → SNV)...")
    X_tr = preprocess(X_tr)
    X_va = preprocess(X_va)
    X_te = preprocess(X_te)

    # ---- CLASS MAP ----
    _log("Building class map (filtering rare classes)...")
    ulbl, ucnt = np.unique(y_tr, return_counts=True)
    keep_set = set(ulbl[ucnt >= cfg.min_samples].tolist())
    mtr = np.isin(y_tr, list(keep_set))
    mva = np.isin(y_va, list(keep_set))
    mte = np.isin(y_te, list(keep_set))
    X_tr, y_tr = X_tr[mtr], y_tr[mtr]
    X_va, y_va = X_va[mva], y_va[mva]
    X_te, y_te = X_te[mte], y_te[mte]

    cmap, inv_map = build_class_map(y_tr)
    nc = len(cmap)
    _log(f"{nc} classes: {[CLASS_NAME.get(inv_map[i], inv_map[i]) for i in range(nc)]}")

    y_tr = np.array([cmap[int(v)] for v in y_tr], dtype=np.int64)
    y_va = np.array([cmap[int(v)] for v in y_va], dtype=np.int64)
    y_te = np.array([cmap[int(v)] for v in y_te], dtype=np.int64)

    # ---- GROUPS & NORMALIZE ----
    _log(f"Splitting to {N_BRANCHES} spectral groups & normalizing...")
    Xg_tr = split_groups(X_tr)
    Xg_va = split_groups(X_va)
    Xg_te = split_groups(X_te)
    Xg_tr, Xg_va, Xg_te, nstats = normalize_groups(Xg_tr, Xg_va, Xg_te)

    # Save metadata
    np.savez(os.path.join(out_dir, "norm_stats.npz"),
             **{f"{k}_{sk}": sv for k, sd in nstats.items() for sk, sv in sd.items()})
    with open(os.path.join(out_dir, "class_map.json"), "w") as f:
        json.dump({str(k): v for k, v in cmap.items()}, f, indent=2)
    with open(os.path.join(out_dir, "inv_class_map.json"), "w") as f:
        json.dump({str(k): v for k, v in inv_map.items()}, f, indent=2)

    # ---- TRAIN ----
    if do_train:
        _log("=" * 60)
        _log("TRAINING — 7-branch ResBlock + HierarchicalLoss, no SSL")
        _log("=" * 60)
        model = MultiBranchAttnCNN(N_BRANCHES, nc, cfg)
        _log(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        nw = 2 if not IN_COLAB else 0
        tr_ld = DataLoader(
            TensorDataset(*[torch.from_numpy(g) for g in Xg_tr], torch.from_numpy(y_tr)),
            batch_size=cfg.batch_train, shuffle=True, num_workers=nw, pin_memory=True,
        )
        va_ld = DataLoader(
            TensorDataset(*[torch.from_numpy(g) for g in Xg_va], torch.from_numpy(y_va)),
            batch_size=cfg.batch_val, num_workers=nw, pin_memory=True,
        )
        hist = train_model(model, tr_ld, va_ld, nc, cfg, cmap, out_dir, device)
        _log(f"Best val accuracy: {hist['best_val_acc'][0]:.4f}")

    # ---- EVALUATE ----
    if do_eval:
        _log("=" * 60)
        _log("EVALUATION")
        _log("=" * 60)
        model = MultiBranchAttnCNN(N_BRANCHES, nc, cfg).to(device)
        model.load_state_dict(torch.load(
            os.path.join(out_dir, "best.pt"), map_location=device, weights_only=True))

        te_ld = DataLoader(
            TensorDataset(*[torch.from_numpy(g) for g in Xg_te], torch.from_numpy(y_te)),
            batch_size=cfg.batch_val, pin_memory=True,
        )

        _log("Per-pixel evaluation...")
        ev = evaluate_pixels(model, te_ld, inv_map, device)
        _log(f"  Accuracy: {ev['accuracy']:.4f}  Kappa: {ev['kappa']:.4f}")
        _log(f"  Macro F1: {ev['report']['macro avg']['f1-score']:.4f}  "
             f"Weighted F1: {ev['report']['weighted avg']['f1-score']:.4f}")

        # Save
        with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
            json.dump({"accuracy": ev["accuracy"], "kappa": ev["kappa"],
                       "macro_f1": ev["report"]["macro avg"]["f1-score"],
                       "weighted_f1": ev["report"]["weighted avg"]["f1-score"],
                       "n_test": len(ev["labels"]), "n_branches": N_BRANCHES}, f, indent=2)

        # Per-class report
        _log("\n  Per-class F1:")
        for cid in ev["class_ids"]:
            name = CLASS_NAME.get(inv_map[cid], str(inv_map[cid]))
            r = ev["report"].get(name, {})
            f1 = r.get("f1-score", 0)
            sup = r.get("support", 0)
            icon = "✅" if f1 >= 0.85 else ("⚠️" if f1 >= 0.7 else "❌")
            _log(f"    {icon} {name:<25} F1={f1:.3f}  n={int(sup)}")

        with open(os.path.join(out_dir, "classification_report.json"), "w") as f:
            json.dump(ev["report"], f, indent=2)

        # Per-scene (only when raw cubes are available)
        has_cubes = any(obs[i].get("cube_path", "") for i in te_i)
        if has_cubes:
            _log("\nPer-scene evaluation with spatial smoothing...")
            scene_res = evaluate_per_scene(model, obs, te_i, cmap, nstats, cfg, device)
            if scene_res["summary"]:
                s = scene_res["summary"]
                _log(f"  Scenes: raw={s['mean_raw']:.4f} smooth={s['mean_smooth']:.4f} "
                     f"\u0394={s['delta']:+.4f} (n={s['n_scenes']})")
            with open(os.path.join(out_dir, "scene_results.json"), "w") as f:
                json.dump(scene_res, f, indent=2, default=str)
        else:
            _log("\nSkipping per-scene spatial smoothing (no raw cubes in npz mode)")

    # ---- ATTENTION ANALYSIS ----
    if do_analyze:
        _log("=" * 60)
        _log("ATTENTION ANALYSIS")
        _log("=" * 60)
        analysis = analyze_attention(ev, inv_map, os.path.join(out_dir, "analysis"))
        _log(f"Saved to {out_dir}/analysis/")

    _log("=" * 60)
    _log("PIPELINE COMPLETE")
    _log("=" * 60)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRISM v2 Pipeline")
    parser.add_argument("--phase", default="all",
                        choices=["train", "evaluate", "analyze", "all"])
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-resblock", action="store_true")
    parser.add_argument("--no-hierarchical", action="store_true")
    parser.add_argument("--npz-path", default=None,
                        help="Path to pre-extracted .npz (auto-downloaded in Colab)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs, batch_train=args.batch, lr=args.lr,
        seed=args.seed, use_resblock=not args.no_resblock,
        hierarchical=not args.no_hierarchical,
    )
    run_pipeline(config, args.data_root, args.out_dir, args.phase, npz_path=args.npz_path)
