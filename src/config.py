from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
DEFAULT_SAMPLE_SIZE = os.getenv("SAMPLE_SIZE")
DEFAULT_SAMPLE_SIZE = int(DEFAULT_SAMPLE_SIZE) if DEFAULT_SAMPLE_SIZE else None

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215

PROCESSED_FILE = DATA_PROCESSED_DIR / "mtpl_claims_modeling.csv"
MODEL_BUNDLE_FILE = MODELS_DIR / "insurerisk_model_bundle.joblib"
PREDICTIONS_FILE = REPORTS_DIR / "policy_level_predictions.csv"
METRICS_FILE = REPORTS_DIR / "model_metrics.json"

for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
