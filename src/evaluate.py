from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src.config import PREDICTIONS_FILE, FIGURES_DIR, REPORTS_DIR


def create_evaluation_artifacts(predictions: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tier_summary = (
        predictions.groupby("risk_tier")
        .agg(
            policies=("IDpol", "count"),
            actual_claim_rate=("HasClaim", "mean"),
            avg_actual_loss=("ClaimAmount", "mean"),
            avg_expected_loss=("expected_loss", "mean"),
            avg_pred_pure_premium=("pred_pure_premium_two_stage", "mean"),
        )
        .reset_index()
    )
    tier_summary.to_csv(REPORTS_DIR / "risk_tier_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    predictions["pred_pure_premium_two_stage"].clip(upper=predictions["pred_pure_premium_two_stage"].quantile(0.99)).hist(bins=40)
    plt.title("Predicted pure premium distribution")
    plt.xlabel("Predicted pure premium")
    plt.ylabel("Policy count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pure_premium_distribution.png", dpi=160)
    plt.close()

    tier_order = ["Low", "Medium", "High", "Very High"]
    tier_summary = tier_summary.set_index("risk_tier").reindex(tier_order).dropna(how="all")
    plt.figure(figsize=(8, 5))
    tier_summary["actual_claim_rate"].plot(kind="bar")
    plt.title("Actual claim rate by predicted risk tier")
    plt.xlabel("Predicted risk tier")
    plt.ylabel("Actual claim rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "claim_rate_by_risk_tier.png", dpi=160)
    plt.close()


def main():
    predictions = pd.read_csv(PREDICTIONS_FILE)
    create_evaluation_artifacts(predictions)
    print("Saved evaluation artifacts in reports/ and reports/figures/.")


if __name__ == "__main__":
    main()
