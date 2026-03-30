"""Configuration for VQA-based mineral classification."""
from pathlib import Path
import numpy as np

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
MODEL_DIR = PROJECT_ROOT / "models" / "vqa_lora"
SRC_DIR = PROJECT_ROOT / "src"

NPZ_PATH = PROJECT_ROOT / "crism_training_data_f16.npz"
SPECTRAL_IMAGES_DIR = DATA_DIR / "spectral_images"
VQA_DATASET_PATH = DATA_DIR / "vqa_dataset.json"

MINERAL_KB_PATH = KNOWLEDGE_DIR / "mineral_kb.json"
ABSORPTION_BANDS_PATH = KNOWLEDGE_DIR / "absorption_bands.json"

# --- Spectral Constants (from crism_v2_pipeline) ---
N_BANDS = 350
BANDS = np.linspace(1.0213, 3.9206, N_BANDS).astype(np.float32)

GROUPS_IDX = [
    (0, 83),    # B0  1.02-1.35 um  Fe2+ crystal field
    (83, 116),  # B1  1.35-1.75 um  H2O 1.4 um
    (116, 175), # B2  1.75-2.15 um  Al-OH, Mg-OH
    (175, 248), # B3  2.15-2.65 um  Metal-OH, CO3
    (248, 296), # B4  2.65-3.10 um  H2O 3 um broad
    (296, 320), # B5  3.10-3.48 um  CO3 v3
    (320, 350), # B6  3.48-3.92 um  Thermal tail
]

GROUP_NAMES = [
    "Fe2+ crystal field",
    "H2O 1.4um",
    "Al-OH / Mg-OH",
    "Metal-OH / CO3",
    "H2O 3um broad",
    "CO3 v3",
    "Thermal tail",
]

# --- Class Names ---
CLASS_NAME = {
    1: "Nontronite", 2: "Fe smectite", 3: "Saponite",
    4: "Montmorillonite", 6: "Vermiculite",
    7: "Fe/Mg smectite", 8: "Al smectite", 9: "Kaolinite",
    10: "Chlorite", 11: "Prehnite", 12: "Kieserite",
    14: "Gypsum", 15: "Bassanite",
    16: "Magnesite", 17: "Calcite", 18: "Hematite",
    19: "Goethite", 20: "LCP", 21: "HCP",
    22: "Olivine", 23: "Mg carbonate",
    24: "CO2 ice", 25: "Fe/Mg clay", 26: "Pyroxene",
    27: "HCP (high-Ca)", 28: "Zeolite", 29: "Sulfate mix",
    30: "Serpentine", 31: "Pyroxene (low-Ca)",
    38: "Olivine (Fo-rich)", 100: "Water-unrelated",
}

# --- VQA Config ---
MAX_SAMPLES = 50000           # max spectra to convert to images
IMAGE_SIZE = (384, 384)       # BLIP-2 input size
IMAGE_DPI = 100

# --- Training Config ---
BASE_MODEL = "Salesforce/blip2-opt-2.7b"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_LENGTH = 128
GRAD_ACCUM_STEPS = 8

# --- Split (observation-wise, matching existing pipeline) ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
