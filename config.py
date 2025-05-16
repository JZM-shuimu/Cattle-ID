# ============================================================
# 🗂️  config.py
# ============================================================
"""Global hyper‑parameters and path settings.
Edit here once – every module reads from the same source."""
from pathlib import Path

# 🛣️  Paths ----------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = PROJECT_ROOT / "datasets_v2"          # change if needed
LOG_DIR      = PROJECT_ROOT / "logs"
CHECKPOINTS  = PROJECT_ROOT / "checkpoints"
ARTIFACTS    = PROJECT_ROOT / "artifacts"             # metrics csv, figures …

for _p in (LOG_DIR, CHECKPOINTS, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

# 🎚️  Hyper‑parameters ----------------------------------------
IMG_SIZE        = 112 #112 224 384
BATCH_SIZE      = 32
LR              = 1e-4
WEIGHT_DECAY    = 5e-2
EPOCHS          = 500
ARC_S, ARC_M    = 30.0, 0.50
RANDOM_STATE    = 42