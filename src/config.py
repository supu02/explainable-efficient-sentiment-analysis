"""
config.py — Central project configuration

Centralises all cross-file paths and shared constants.
Edit only here; model files import from this module.

Usage note:
    Scripts should be run from the project root:
        python -m src.tfidf_logreg_model
    or imported as part of the src package:
        from src.tfidf_logreg_model import TfidfLogRegModel
"""

from pathlib import Path

# Project root = directory that contains src/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_TRAIN_PATH = str(PROJECT_ROOT / "data" / "train.csv")
DATA_EVAL_PATH  = str(PROJECT_ROOT / "data" / "eval_clean.csv")

# ── Model artifacts ───────────────────────────────────────────────────────────
MODELS_DIR               = str(PROJECT_ROOT / "models")
TFIDF_LOGREG_MODEL_PATH  = str(PROJECT_ROOT / "models" / "tfidf_logreg.joblib")
TINYBERT_MODEL_PATH      = str(PROJECT_ROOT / "models" / "tinybert_quantized.pt")
TINYBERT_TOKENIZER_PATH  = str(PROJECT_ROOT / "models" / "tokenizer")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
