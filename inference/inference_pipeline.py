#!/usr/bin/env python3
"""
Complete inference pipeline for CRISM TRR3 images using trained CNN model.
1. JCAT atmospheric correction
2. CNN inference with attention model
3. Save results (prediction map + visualization)
"""

import os
import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn

# Add JCAT_Automation to path for autojcat
sys.path.insert(0, "../JCAT_Automation")
import autojcat

# ============================================================
# CONFIG
# ============================================================
TRR_IMG = "frt00008a1e_07_if168l_trr3.img"
TRR_LBL = "frt00008a1e_07_if168l_trr3.lbl"
DDR_IMG = "frt00008a1e_07_de168l_ddr1.img"
DDR_LBL = "frt00008a1e_07_de168l_ddr1.lbl"

MODEL_PATH = "epoch_attn_v7/best.pt"
CLASS_MAP_PATH = "epoch_attn_v7/class_map.json"
INV_CLASS_MAP_PATH = "epoch_attn_v7/inv_class_map.json"

ADR_IMG = "../JCAT/src/main/resources/resources/adr/vs/ADR90947778566_11D87_VS00L_8.IMG"

OUT_DIR = "./inference_output"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024
FILL_VALUE = 65535.0

# IR band configuration
# Training used bands 88-437 from 438-band L-sensor TRR files
# This matches the ADR reference which starts at band 88
IR_START = 88
IR_COUNT = 350

# Wavelength array (must match training)
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

# Spectral groups (same as training)
GROUPS_UM = [
    (1.02, 1.35),
    (1.35, 1.75),
    (1.75, 2.05),
    (2.05, 2.35),
    (2.35, 2.65),
    (2.81, 3.10),
    (3.10, 3.48),
]

def idx(w):
    return int(np.searchsorted(BANDS, w, side="left"))

GROUPS_IDX = [(idx(a), idx(b)) for a, b in GROUPS_UM]

# Mineral class names
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
    38: "Chlorite-smectite",
    100: "Water-unrelated",
}

# ============================================================
# PDS3 Label Parser
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

# ============================================================
# Data Loading
# ============================================================
def load_trr_cube(trr_img, trr_lbl):
    """Load TRR image cube (rows, cols, bands)."""
    meta = parse_pds3_label(trr_lbl)
    rows = int(meta["LINES"])
    cols = int(meta["LINE_SAMPLES"])
    bands = int(meta["BANDS"])

    raw = np.fromfile(trr_img, dtype=np.float32)
    expected = rows * cols * bands

    if raw.size < expected:
        available_rows = raw.size // (cols * bands)
        print(f"  [WARN] Truncated: using {available_rows}/{rows} rows")
        rows = available_rows
        expected = rows * cols * bands

    raw = raw[:expected]
    cube_lbc = raw.reshape(rows, bands, cols)
    cube_rcb = np.transpose(cube_lbc, (0, 2, 1))

    return cube_rcb, rows, cols

def load_ddr_ina(ddr_img, ddr_lbl):
    """Load incidence angle from DDR."""
    meta = parse_pds3_label(ddr_lbl)
    rows = int(meta["LINES"])
    cols = int(meta["LINE_SAMPLES"])
    bands = int(meta["BANDS"])

    raw = np.fromfile(ddr_img, dtype=np.float32)
    raw = raw[:bands * rows * cols]
    cube = raw.reshape(bands, rows, cols)

    # Band 0 is INA
    ina_deg = float(cube[0, rows // 2, cols // 2])

    if ina_deg >= 65534 or not np.isfinite(ina_deg):
        center = cube[0, rows//2-5:rows//2+5, cols//2-5:cols//2+5]
        valid = center[(center < 65534) & np.isfinite(center)]
        ina_deg = float(np.median(valid)) if len(valid) > 0 else 30.0

    return math.radians(ina_deg)

def load_vs_adr(adr_img, ir_count):
    """Load VS ADR for atmospheric correction."""
    adr_bands = 438
    raw = np.fromfile(adr_img, dtype=np.float32)
    total_records = raw.size // 640

    if total_records == adr_bands + 1:
        lines = 1
    elif total_records == adr_bands * 3 + 1:
        lines = 3
    else:
        raise ValueError(f"Unexpected ADR layout: {total_records} records")

    raw = raw[:adr_bands * lines * 640]
    cube = raw.reshape(lines, adr_bands, 640)

    mid = 640 // 2
    vstrans = cube[0, :ir_count, mid]
    vsart = cube[1, :ir_count, mid] if lines > 1 else np.zeros(ir_count)

    return vstrans.astype(np.float32), vsart.astype(np.float32)

# ============================================================
# Model Definition (must match training)
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
        x = self.net(x.unsqueeze(1))
        return x.flatten(1)

class MultiBranchAttnCNN(nn.Module):
    def __init__(self, n_branches, n_classes, feat_ch=64, pool=2, att_temperature=0.5):
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

        att_logits = self.att_fc(cat)
        w = torch.softmax(att_logits / self.att_temperature, dim=1)

        f = sum(w[:, i:i+1] * feats[i] for i in range(len(feats)))
        logits = self.cls(f)
        return (logits, w) if return_attn else logits

# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("=" * 60)
    print("CRISM CNN Inference Pipeline")
    print("=" * 60)

    # Load class maps
    with open(CLASS_MAP_PATH) as f:
        class_map = {int(k): int(v) for k, v in json.load(f).items()}
    with open(INV_CLASS_MAP_PATH) as f:
        inv_class_map = {int(k): int(v) for k, v in json.load(f).items()}

    n_classes = len(class_map)
    print(f"[INFO] Classes: {n_classes}")

    # Step 1: Load TRR cube
    print("\n[STEP 1] Loading TRR cube...")
    cube, rows, cols = load_trr_cube(TRR_IMG, TRR_LBL)
    print(f"  Shape: {cube.shape} (rows={rows}, cols={cols}, bands={cube.shape[2]})")

    # Step 2: Load DDR for incidence angle
    print("\n[STEP 2] Loading DDR...")
    ina_rad = load_ddr_ina(DDR_IMG, DDR_LBL)
    print(f"  Incidence angle: {math.degrees(ina_rad):.2f} deg")

    # Step 3: Load ADR
    print("\n[STEP 3] Loading ADR...")
    vstrans, vsart = load_vs_adr(ADR_IMG, IR_COUNT)
    print(f"  VS trans/art shape: {vstrans.shape}")

    # Step 4: JCAT correction
    print("\n[STEP 4] Applying JCAT correction...")
    wl = BANDS.tolist()
    vstrans_list = vstrans.tolist()
    vsart_list = vsart.tolist()

    # Extract bands 88-437 (same as training) and apply correction
    cube_ir = cube[:, :, IR_START:IR_START+IR_COUNT].copy()
    cube_corrected = np.empty_like(cube_ir)

    total_pixels = rows * cols
    for r in range(rows):
        if r % 50 == 0:
            print(f"  Processing row {r}/{rows} ({100*r/rows:.1f}%)", end='\r')
        for c in range(cols):
            spec = cube_ir[r, c, :].tolist()
            if autojcat.get_median(spec) >= 65534.0:
                cube_corrected[r, c, :] = cube_ir[r, c, :]
            else:
                cube_corrected[r, c, :] = autojcat.jcat_correction_pipeline(
                    intensity=spec,
                    wavelength=wl,
                    vstrans=vstrans_list,
                    vsart=vsart_list,
                    ina_rad=ina_rad
                )
    print(f"\n  Done. Corrected cube shape: {cube_corrected.shape}")

    # Save corrected cube
    atp_path = os.path.join(OUT_DIR, TRR_IMG.replace(".img", "_ATP_IRONLY.img"))
    cube_bil = np.transpose(cube_corrected, (0, 2, 1))
    cube_bil.astype(np.float32).tofile(atp_path)
    print(f"  Saved: {atp_path}")

    # Step 5: Load model
    print("\n[STEP 5] Loading CNN model...")
    model = MultiBranchAttnCNN(len(GROUPS_UM), n_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"  Model loaded from {MODEL_PATH}")
    print(f"  Device: {DEVICE}")

    # Step 6: Run inference
    print("\n[STEP 6] Running inference...")

    # Reshape to (N, bands) for batch processing
    X_flat = cube_corrected.reshape(-1, IR_COUNT)

    # Create validity mask - require at least 250 valid bands
    valid_band_count = np.sum((X_flat < 65534) & np.isfinite(X_flat), axis=1)
    valid_mask = valid_band_count >= 250
    print(f"  Valid pixels (>=250 bands): {valid_mask.sum()} / {len(valid_mask)}")

    # Replace fill values with 0 for valid pixels (model can handle some zeros)
    X_flat = np.where(X_flat >= 65534, 0, X_flat)
    X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

    # Initialize predictions
    pred_flat = np.full(len(X_flat), -1, dtype=np.int32)
    attn_flat = np.zeros((len(X_flat), len(GROUPS_UM)), dtype=np.float32)

    # Get valid indices
    valid_idx = np.where(valid_mask)[0]
    X_valid = X_flat[valid_idx]

    # Batch inference
    with torch.no_grad():
        for i in range(0, len(X_valid), BATCH_SIZE):
            if i % 10000 == 0:
                print(f"  Batch {i}/{len(X_valid)}", end='\r')

            batch = X_valid[i:i+BATCH_SIZE]

            # Split into spectral groups
            xg = [
                torch.from_numpy(batch[:, s:e]).float().to(DEVICE)
                for s, e in GROUPS_IDX
            ]

            logits, attn = model(*xg, return_attn=True)
            preds = logits.argmax(dim=1).cpu().numpy()

            pred_flat[valid_idx[i:i+BATCH_SIZE]] = preds
            attn_flat[valid_idx[i:i+BATCH_SIZE]] = attn.cpu().numpy()

    print(f"\n  Inference complete.")

    # Reshape predictions
    pred_map = pred_flat.reshape(rows, cols)
    attn_map = attn_flat.reshape(rows, cols, -1)

    # Convert to original class IDs
    pred_orig = np.full_like(pred_map, -1)
    for i in range(rows):
        for j in range(cols):
            if pred_map[i, j] >= 0:
                pred_orig[i, j] = inv_class_map[pred_map[i, j]]

    # Step 7: Save results
    print("\n[STEP 7] Saving results...")

    np.save(os.path.join(OUT_DIR, "prediction_map.npy"), pred_orig)
    np.save(os.path.join(OUT_DIR, "attention_map.npy"), attn_map)
    print(f"  Saved: prediction_map.npy, attention_map.npy")

    # Print class statistics
    print("\n[RESULTS] Class distribution:")
    unique, counts = np.unique(pred_orig[pred_orig >= 0], return_counts=True)
    for cls_id, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        name = CLASS_NAME.get(cls_id, f"Class {cls_id}")
        print(f"  {cls_id:3d}: {name:25s} - {cnt:7d} pixels ({100*cnt/valid_mask.sum():.2f}%)")

    # Create visualization
    print("\n[STEP 8] Creating visualization...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Prediction map
        cmap = plt.cm.get_cmap("tab20", 20)
        im = axes[0].imshow(pred_orig, cmap=cmap, vmin=0, vmax=100)
        axes[0].set_title("CNN Mineral Prediction")
        axes[0].axis("off")

        # Legend
        patches = [
            mpatches.Patch(color=cmap(int(cls_id) % 20),
                          label=f"{cls_id}: {CLASS_NAME.get(cls_id, '?')}")
            for cls_id in unique if cls_id in CLASS_NAME
        ]
        axes[0].legend(handles=patches, bbox_to_anchor=(1.02, 1),
                      loc="upper left", fontsize=7)

        # Attention entropy map
        entropy = -np.sum(attn_map * np.log(attn_map + 1e-8), axis=2)
        entropy[~valid_mask.reshape(rows, cols)] = np.nan
        im2 = axes[1].imshow(entropy, cmap="viridis")
        axes[1].set_title("Attention Entropy")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, "prediction_visualization.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig_path}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping visualization")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
