#!/usr/bin/env python3
"""Extract labeled pixels from ATP cubes → compact .npz for Colab upload."""
import glob, os, re, time, sys
import numpy as np

DATA_ROOT = "/home/cspark/data/crism_ml"
OUT_PATH = "/home/cspark/mineral_classification/crism_training_data.npz"
N_BANDS = 350
FILL_VALUE = 65535.0
IGNORE_LABEL = -1
VALID_PREFIXES = ("frt", "frs", "hrl", "hrs")

def parse_lbl(path):
    meta = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("/*"):
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip().strip('"')
    return meta

def find_label(name, root):
    stem = name.split("_")[0]
    m = re.match(r"(frt|frs|hrl|hrs)([0-9a-fA-F]+)", stem)
    if not m: return None, None
    hid = m.group(2).upper().lstrip("0")
    cands = sorted(glob.glob(os.path.join(root, f"im_*{hid}_LABEL.npy")))
    return hid, (cands[0] if cands else None)

def align(label, cr, cc):
    lr, lc = label.shape[:2]
    if lc != cc:
        if lc == cc * 2: label = label[:, ::2]
        elif cc == lc * 2: label = np.repeat(label, 2, axis=1)
        else: return None
    lr = label.shape[0]
    if lr == cr: return label
    if lr > cr:
        s = (lr - cr) // 2
        return label[s:s+cr]
    return label

t0 = time.time()
files = sorted(glob.glob(os.path.join(DATA_ROOT, "*_ATP_IRONLY.img")))
files = [f for f in files if os.path.basename(f).startswith(VALID_PREFIXES)]
print(f"Found {len(files)} ATP_IRONLY files")

all_obs_ids = []
all_X = []
all_y = []
all_obs_idx = []  # which observation each pixel belongs to
obs_metadata = {} # obs_id → {cube_shape, cube_path, lbl_path}

for fi, img_path in enumerate(files):
    name = os.path.basename(img_path)
    oid, lp = find_label(name, DATA_ROOT)
    if oid is None or lp is None: continue
    lbl_path = os.path.join(DATA_ROOT, name.replace("_ATP_IRONLY.img", ".lbl"))
    if not os.path.exists(lbl_path): continue

    meta = parse_lbl(lbl_path)
    cols = int(meta["LINE_SAMPLES"])
    raw = np.fromfile(img_path, dtype=np.float32)
    block = cols * N_BANDS
    if raw.size % block != 0: continue
    rows = raw.size // block
    cube = np.transpose(raw.reshape(rows, N_BANDS, cols), (0, 2, 1))

    label_full = np.load(lp)
    label = align(label_full, rows, cols)
    if label is None: continue
    if label.shape[0] < rows:
        off = (rows - label.shape[0]) // 2
        cube = cube[off:off+label.shape[0]]
    if cube.shape[:2] != label.shape[:2]: continue

    # Zero-out fill bands instead of rejecting entire pixels
    fill_mask = (cube == FILL_VALUE)
    cube[fill_mask] = 0.0
    n_valid_bands = (~fill_mask).sum(axis=-1)

    valid = (
        (label != IGNORE_LABEL)
        & np.all(np.isfinite(cube), axis=-1)
        & (n_valid_bands >= 250)  # keep pixels with >=250/350 valid bands
    )
    if not valid.any(): continue

    X = cube[valid].astype(np.float32)
    y = label[valid].astype(np.int64)

    obs_num = len(all_obs_ids)
    all_obs_ids.append(oid)
    all_X.append(X)
    all_y.append(y)
    all_obs_idx.append(np.full(len(y), obs_num, dtype=np.int32))
    obs_metadata[oid] = {"cube_path": name, "lbl_file": os.path.basename(lp),
                          "cube_rows": int(cube.shape[0]), "cube_cols": int(cube.shape[1])}

    elapsed = time.time() - t0
    print(f"  [{fi+1}/{len(files)}] {oid}: {X.shape[0]} pixels  ({elapsed:.0f}s)", flush=True)

X_all = np.concatenate(all_X)
y_all = np.concatenate(all_y)
obs_idx = np.concatenate(all_obs_idx)
obs_ids = np.array(all_obs_ids, dtype="<U32")

print(f"\nTotal: {X_all.shape[0]:,} pixels from {len(all_obs_ids)} observations")
print(f"Classes: {sorted(set(y_all.tolist()))}")
print(f"Shape: X={X_all.shape}, y={y_all.shape}, obs_idx={obs_idx.shape}")
print(f"Saving to {OUT_PATH}...")

np.savez_compressed(
    OUT_PATH,
    X=X_all,                      # (N, 350) raw reflectance
    y=y_all,                      # (N,) class labels
    obs_idx=obs_idx,              # (N,) observation index per pixel
    obs_ids=obs_ids,              # (n_obs,) observation hex IDs
    bands=np.linspace(1.0213, 3.9206, 350).astype(np.float32),
)

sz = os.path.getsize(OUT_PATH) / 1e6
print(f"Done! {sz:.1f} MB  ({time.time()-t0:.0f}s total)")
