from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))




import argparse

from src.data_ingestion import fetch_fremtpl2
from src.data_quality import run_quality_checks
from src.train import train_models
from src.evaluate import create_evaluation_artifacts
from src.explain import create_explainability_outputs
from src.config import PROCESSED_FILE, MODEL_BUNDLE_FILE, PREDICTIONS_FILE, METRICS_FILE, REPORTS_DIR
from src.utils import save_json

import joblib


def main():
    parser = argparse.ArgumentParser(description="Run InsureRisk AI pipeline end-to-end.")
    parser.add_argument("--sample-size", type=int, default=60000)
    parser.add_argument("--skip-explain", action="store_true")
    args = parser.parse_args()

    df = fetch_fremtpl2(sample_size=args.sample_size)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved processed data: {PROCESSED_FILE}")

    quality = run_quality_checks(df)
    quality.to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)
    print("Data quality report saved.")

    result = train_models(df)
    joblib.dump(result["bundle"], MODEL_BUNDLE_FILE)
    result["predictions"].to_csv(PREDICTIONS_FILE, index=False)
    save_json(result["metrics"], METRICS_FILE)
    print("Model training complete.")

    create_evaluation_artifacts(result["predictions"])
    print("Evaluation artifacts created.")

    if not args.skip_explain:
        create_explainability_outputs(sample_rows=2000)
        print("Explainability artifacts created.")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
