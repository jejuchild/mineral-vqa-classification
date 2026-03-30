#!/usr/bin/env python3
# ============================================================
# CRISM TRR False Color + CNN Mineral Overlay (OBS-wise)
# - Inference logic: from post-analysis code
# - Visualization: from crism_ml false color plotting
# ============================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import loadmat

from crism_ml.io import load_image, image_shape
from crism_ml.preprocessing import filter_bad_pixels
from crism_ml.plot import get_false_colors

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "trr")
MAT_PATH = os.path.join(
    BASE_DIR, "data", "labeled_pixels",
    "CRISM_labeled_pixels_ratioed.mat"
)
CKPT_PATH = os.path.join(
    BASE_DIR, "model", "multi_attn_best.pt"
)

OUT_DIR = os.path.join(BASE_DIR, "output", "overlay_results_obswise")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256

# ============================================================
# Mineral names (label id)
# ============================================================
CLASS_NAME = {
    1: "CO Ice",
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
    100: "Water-unrelated",
}

# ============================================================
# Helper
# ============================================================
def extract_obsid(path):
    return os.path.basename(path).split("_")[0][-5:].lower()

# ============================================================
# Load labeled pixel dataset
# ============================================================
print("[INFO] Loading labeled pixels...")
mat = loadmat(MAT_PATH)

pixspec = mat["pixspec"].astype(np.float32)
pixlabs = mat["pixlabs"].squeeze().astype(int)
pixims  = mat["pixims"].squeeze().astype(int) - 1
pixcrds = mat["pixcrds"].astype(int)
im_names = mat["im_names"].squeeze()

im_names = np.array([
    str(s[0]) if isinstance(s, np.ndarray) else str(s)
    for s in im_names
])

rows = pixcrds[:, 0]
cols = pixcrds[:, 1]

# ============================================================
# Label remap (same as training)
# ============================================================
WATER_UNRELATED = {5, 13, 30, 33, 34, 35, 36, 37}
KEEP_CLASSES = {
    1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12,
    14, 15, 16, 17, 18, 19, 23, 25, 26,
    27, 29, 31, 38, 39
}
NEW_WATER_UNRELATED_ID = 100

def remap_label(v):
    if v in WATER_UNRELATED:
        return NEW_WATER_UNRELATED_ID
    if v in KEEP_CLASSES:
        return v
    return None

# ============================================================
# Load CNN (same as post-analysis)
# ============================================================
print("[INFO] Loading CNN model...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

GROUPS_IDX = ckpt["band_groups_idx"]
GROUP_LENS = ckpt["group_lens"]

class_map = {int(k): int(v) for k, v in ckpt["class_map"].items()}
inv_class_map = {int(k): int(v) for k, v in ckpt["inv_class_map"].items()}
N_CLASSES = len(class_map)

class SpectralBranch(nn.Module):
    def __init__(self, L, C=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, C, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)

class MultiBranchAttnCNN(nn.Module):
    def __init__(self, group_lens, n_classes):
        super().__init__()
        self.branches = nn.ModuleList([SpectralBranch(L) for L in group_lens])
        self.att_fc = nn.Sequential(
            nn.Linear(64 * len(group_lens), 64),
            nn.ReLU(),
            nn.Linear(64, len(group_lens))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, *xg):
        feats = [b(x) for b, x in zip(self.branches, xg)]
        cat = torch.cat(feats, dim=1)
        w = torch.softmax(self.att_fc(cat), dim=1)
        f = sum(w[:, i:i+1] * feats[i] for i in range(len(feats)))
        return self.classifier(f)

model = MultiBranchAttnCNN(GROUP_LENS, N_CLASSES).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

def split_groups_np(X):
    return [
        torch.from_numpy(X[:, s:e]).float().to(DEVICE)
        for s, e in GROUPS_IDX
    ]

@torch.no_grad()
def infer(X):
    preds = []
    for i in range(0, len(X), BATCH):
        xb = X[i:i+BATCH]
        logits = model(*split_groups_np(xb))
        preds.append(torch.argmax(logits, 1).cpu().numpy())
    return np.concatenate(preds)

# ============================================================
# MAIN LOOP
# ============================================================
trr_imgs = sorted(glob.glob(os.path.join(DATA_DIR, "*trr3.img")))
print(f"[INFO] Found {len(trr_imgs)} TRR images")

for img_path in trr_imgs:
    obsid = extract_obsid(img_path)
    print(f"\n[PROCESS] {obsid}")

    try:
        # ------------------------
        # False color background
        # ------------------------
        trr = load_image(img_path)
        IF_clean, bad_mask = filter_bad_pixels(trr["IF"])
        H, W = image_shape(trr)

        IF_img = IF_clean.reshape(H, W, -1)
        mask_img = bad_mask.reshape(H, W)
        fc = get_false_colors(IF_img, mask_img).astype(np.float32)
        if fc.max() > 1.5:
            fc /= 255.0

        gray = 0.299*fc[...,0] + 0.587*fc[...,1] + 0.114*fc[...,2]
        bg = 0.8 * np.stack([gray]*3, axis=-1)

        # ------------------------
        # Select pixels for obsid
        # ------------------------
        Xc, rc, cc = [], [], []
        for x, y, im, r, c in zip(pixspec, pixlabs, pixims, rows, cols):
            if remap_label(int(y)) is None:
                continue
            if obsid not in im_names[im].lower():
                continue
            Xc.append(x); rc.append(r); cc.append(c)

        if len(Xc) == 0:
            print("  [SKIP] no pixels")
            continue

        Xc = np.asarray(Xc, np.float32)
        rc = np.asarray(rc)
        cc = np.asarray(cc)

        # ------------------------
        # Inference
        # ------------------------
        y_pred = infer(Xc)

        label_map = np.full((H, W), np.nan)
        for r, c, p in zip(rc, cc, y_pred):
            if 0 <= r < H and 0 <= c < W:
                label_map[r, c] = inv_class_map[int(p)]

        present = [int(v) for v in np.unique(label_map) if not np.isnan(v)]

        # ------------------------
        # Plot
        # ------------------------
        plt.figure(figsize=(8,6))
        plt.imshow(bg)
        plt.imshow(label_map, cmap="tab20", alpha=0.85)
        plt.axis("off")
        plt.title(f"{obsid} – CNN Mineral Map")

        cmap = plt.cm.get_cmap("tab20", 20)
        patches = [
            mpatches.Patch(color=cmap(v % 20),
                           label=f"{v}: {CLASS_NAME[v]}")
            for v in present if v in CLASS_NAME
        ]

        plt.legend(handles=patches,
                   bbox_to_anchor=(1.02,1),
                   loc="upper left",
                   fontsize=8)

        out = os.path.join(OUT_DIR, f"{obsid}_overlay.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  [OK] saved {out}")
        print(f"  [PRESENT] {present}")

    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n[FINISHED] All scenes processed.")
