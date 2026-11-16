
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models_trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
SRC_DIR = PROJECT_ROOT / "src"

# Dataset files
CHAT_DATA = RAW_DATA_DIR / "chat_dataset.csv"
GEM_DATA = RAW_DATA_DIR / "gem_dataset.csv"

## Models
MODEL_BASE = "answerdotai/ModernBERT-base"
# MODEL_BASE = MODELS_DIR / "ethan_resolver_model_10000"
MODEL_UNTRAINED = "bert-base-uncased"
MODEL_RESOLVER = MODELS_DIR / "ethan_multitask_modernbert_v2"

# Create directories if they don't exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)
