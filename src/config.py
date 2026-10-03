"""Central configuration: model IDs come from .env, thresholds live here."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)  # .env wins over stale shell variables

DATASET_ROOT = Path(os.getenv("DATASET_ROOT") or (ROOT / "dataset"))
if not DATASET_ROOT.exists():
    DATASET_ROOT = ROOT / "dataset"
XCEPTION_TIMM_NAME = os.getenv("XCEPTION_TIMM_NAME", "")
WAV2VEC2_MODEL_ID = os.getenv("WAV2VEC2_MODEL_ID", "")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
EMB_DIR = OUT_DIR / "embeddings"
LOG_DIR = ROOT / "logs"
SUBSET_CSV = DATA_DIR / "subset.csv"

# --- segmentation (SPEC 4.2) ---
WINDOW_S = 4.0
HOP_S = 2.0
FRAMES_PER_WINDOW = 2
CROP_MARGIN = 0.25
SAMPLE_RATE = 16000
W2V_LAYERS = (6, 9, 12)
MAX_VIDEO_S = 90.0

# --- reliability / fusion thresholds (SPEC 5-7) ---
R_GATE = 0.15
R_SUM_CAP = 0.5
CAP_VALUE = 0.64
INSUFFICIENT_R_SUM = 0.30
INSUFFICIENT_S = 0.35
SYNERGY_THRESH = 0.65
SYNERGY_BONUS = 0.10
BAND_LOW = 0.35
BAND_HIGH = 0.65
LOW_CONF_AUC = 0.65      # head with val AUC below this gets weight * LOW_CONF_WEIGHT
LOW_CONF_WEIGHT = 0.3
TOP_K_WINDOWS = 3
EVIDENCE_MIN_SCORE = 0.5   # a media detector window is shown as 'evidence' only if its score >= this

# --- reliability formula parameters (documented in src/reliability.py) ---
FACE_SIZE_FULL = 0.15    # face-height/frame-height fraction at which size score saturates
BLUR_FULL = 100.0        # Laplacian variance at which blur score saturates
CLIP_BAD = 0.02          # clipping fraction at which clipping score hits 0
SNR_FULL_DB = 20.0
SPEECH_FULL = 0.6


