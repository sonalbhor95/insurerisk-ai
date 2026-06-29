from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_BUNDLE_FILE


def load_model_bundle(path=MODEL_BUNDLE_FILE) -> Dict[str, Any]:
    return joblib.load(path)


def assign_risk_tier(pure_premium: float, thresholds: Dict[str, float]) -> str:
    if pure_premium >= thresholds["very_high"]:
        return "Very High"
    if pure_premium >= thresholds["high"]:
        return "High"
    if pure_premium >= thresholds["medium"]:
        return "Medium"
    return "Low"


def predict_policy(policy: Dict[str, Any], bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if bundle is None:
        bundle = load_model_bundle()

    X = pd.DataFrame([policy])[bundle["feature_columns"]]
    exposure = float(X["Exposure"].iloc[0])

    frequency = float(np.clip(bundle["frequency_model"].predict(X)[0], 0, None))
    severity = float(np.clip(bundle["severity_model"].predict(X)[0], 0, None))
    pure_premium = float(frequency * severity)
    expected_loss = float(pure_premium * exposure)
    expected_claim_count = float(frequency * exposure)
    tweedie_pure_premium = float(np.clip(bundle["tweedie_model"].predict(X)[0], 0, None))

    return {
        "pred_claim_frequency": frequency,
        "pred_claim_severity": severity,
        "pred_pure_premium_two_stage": pure_premium,
        "pred_pure_premium_tweedie": tweedie_pure_premium,
        "expected_claim_count": expected_claim_count,
        "expected_loss": expected_loss,
        "risk_tier": assign_risk_tier(pure_premium, bundle["risk_thresholds"]),
    }


if __name__ == "__main__":
    sample_policy = {
        "Exposure": 0.75,
        "VehPower": 7,
        "VehAge": 4,
        "DrivAge": 42,
        "BonusMalus": 60,
        "Density": 1200,
        "Area": "C",
        "VehBrand": "B2",
        "VehGas": "Regular",
        "Region": "R24",
    }
    print(predict_policy(sample_policy))
