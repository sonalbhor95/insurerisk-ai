from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from src.config import MODELS_DIR, REPORTS_DIR


def normalized_gini(y_true, y_score) -> float:
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def create_lift_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "score": y_score})
    df["decile"] = pd.qcut(df["score"], q=n_bins, labels=False, duplicates="drop") + 1

    lift = (
        df.groupby("decile")
        .agg(
            policies=("y_true", "count"),
            claims=("y_true", "sum"),
            avg_score=("score", "mean"),
        )
        .reset_index()
        .sort_values("decile", ascending=False)
    )

    overall_claim_rate = df["y_true"].mean()
    lift["claim_rate"] = lift["claims"] / lift["policies"]
    lift["lift"] = lift["claim_rate"] / overall_claim_rate

    return lift


def create_calibration_and_lift_artifacts(
    processed_data_path: str | Path = "data/processed/mtpl_claims_modeling.csv",
    model_path: str | Path = "models/lightgbm_claim_classifier.joblib",
) -> None:
    df = pd.read_csv(processed_data_path)
    model = joblib.load(model_path)

    df["has_claim"] = (df["ClaimNb"] > 0).astype(int)

    drop_cols = [
        "IDpol",
        "ClaimNb",
        "ClaimAmount",
        "pure_premium",
        "frequency_target",
        "severity_target",
        "has_claim",
    ]

    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    y = df["has_claim"]

    y_score = model.predict_proba(X)[:, 1]

    reports_dir = Path(REPORTS_DIR)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_score, n_bins=10, strategy="quantile")

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Predicted claim probability")
    plt.ylabel("Observed claim rate")
    plt.title("Calibration Curve: Claim Probability Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "calibration_curve_claim_probability.png", dpi=150)
    plt.close()

    # Lift table and chart
    lift_table = create_lift_table(y, y_score, n_bins=10)
    lift_table.to_csv(reports_dir / "claim_probability_lift_table.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.bar(lift_table["decile"].astype(str), lift_table["lift"])
    plt.xlabel("Risk decile")
    plt.ylabel("Lift")
    plt.title("Lift Chart by Risk Decile")
    plt.tight_layout()
    plt.savefig(figures_dir / "lift_chart_claim_probability.png", dpi=150)
    plt.close()

    gini = normalized_gini(y, y_score)

    metrics = {
        "roc_auc": float(roc_auc_score(y, y_score)),
        "normalized_gini": float(gini),
        "overall_claim_rate": float(y.mean()),
    }

    with open(reports_dir / "risk_ranking_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Calibration, lift, and Gini artifacts created.")


if __name__ == "__main__":
    create_calibration_and_lift_artifacts()