from __future__ import annotations

import argparse
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import PROCESSED_FILE, MODEL_BUNDLE_FILE, METRICS_FILE, PREDICTIONS_FILE, RANDOM_STATE
from src.features import build_preprocessor, get_feature_columns, make_X
from src.utils import normalized_gini, save_json, rmse


def train_models(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    features, numeric_features, categorical_features = get_feature_columns(df)
    X = make_X(df)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["HasClaim"],
    )

    train = df.loc[train_idx].copy()
    test = df.loc[test_idx].copy()
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    frequency_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", PoissonRegressor(alpha=1e-4, max_iter=300)),
        ]
    )
    frequency_target = train["ClaimNb"] / train["Exposure"]
    frequency_model.fit(X_train, frequency_target, model__sample_weight=train["Exposure"])

    severity_train = train[(train["ClaimNb"] > 0) & (train["ClaimAmount"] > 0)].copy()
    if len(severity_train) < 100:
        raise ValueError("Not enough positive claims for severity model. Increase sample_size.")

    severity_model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", GammaRegressor(alpha=1e-4, max_iter=500)),
        ]
    )
    severity_target = severity_train["ClaimAmount"] / severity_train["ClaimNb"].clip(lower=1)
    severity_model.fit(
        make_X(severity_train),
        severity_target,
        model__sample_weight=severity_train["ClaimNb"].clip(lower=1),
    )

    tweedie_model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=500)),
        ]
    )
    tweedie_target = train["ClaimAmount"] / train["Exposure"]
    tweedie_model.fit(X_train, tweedie_target, model__sample_weight=train["Exposure"])

    predictions, metrics, thresholds = score_test_set(
        test=test,
        X_test=X_test,
        frequency_model=frequency_model,
        severity_model=severity_model,
        tweedie_model=tweedie_model,
    )

    bundle = {
        "frequency_model": frequency_model,
        "severity_model": severity_model,
        "tweedie_model": tweedie_model,
        "feature_columns": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "risk_thresholds": thresholds,
        "metrics": metrics,
    }

    return {"bundle": bundle, "metrics": metrics, "predictions": predictions}


def score_test_set(test, X_test, frequency_model, severity_model, tweedie_model):
    freq_pred = np.clip(frequency_model.predict(X_test), 0, None)
    sev_pred = np.clip(severity_model.predict(X_test), 0, None)
    pure_premium_two_stage = freq_pred * sev_pred
    pure_premium_tweedie = np.clip(tweedie_model.predict(X_test), 0, None)

    actual_pure_premium = test["ClaimAmount"] / test["Exposure"]
    actual_claim_count = test["ClaimNb"]
    expected_claim_count = freq_pred * test["Exposure"].to_numpy()
    expected_loss = pure_premium_two_stage * test["Exposure"].to_numpy()

    thresholds = {
        "medium": float(np.quantile(pure_premium_two_stage, 0.60)),
        "high": float(np.quantile(pure_premium_two_stage, 0.85)),
        "very_high": float(np.quantile(pure_premium_two_stage, 0.95)),
    }

    risk_tier = pd.cut(
        pure_premium_two_stage,
        bins=[-np.inf, thresholds["medium"], thresholds["high"], thresholds["very_high"], np.inf],
        labels=["Low", "Medium", "High", "Very High"],
    ).astype(str)

    pred_df = test[["IDpol", "Exposure", "ClaimNb", "ClaimAmount", "ClaimFrequency", "PurePremium", "HasClaim"]].copy()
    pred_df["pred_claim_frequency"] = freq_pred
    pred_df["pred_claim_severity"] = sev_pred
    pred_df["pred_pure_premium_two_stage"] = pure_premium_two_stage
    pred_df["pred_pure_premium_tweedie"] = pure_premium_tweedie
    pred_df["expected_claim_count"] = expected_claim_count
    pred_df["expected_loss"] = expected_loss
    pred_df["risk_tier"] = risk_tier

    has_claim_score = expected_claim_count
    try:
        claim_auc = float(roc_auc_score(test["HasClaim"], has_claim_score))
    except ValueError:
        claim_auc = None

    metrics = {
        "frequency_mae_expected_claim_count": float(mean_absolute_error(actual_claim_count, expected_claim_count)),
        "pure_premium_two_stage_mae": float(mean_absolute_error(actual_pure_premium, pure_premium_two_stage)),
        "pure_premium_two_stage_rmse": rmse(actual_pure_premium, pure_premium_two_stage),
        "pure_premium_tweedie_mae": float(mean_absolute_error(actual_pure_premium, pure_premium_tweedie)),
        "pure_premium_tweedie_rmse": rmse(actual_pure_premium, pure_premium_tweedie),
        "claim_occurrence_auc": claim_auc,
        "normalized_gini_expected_loss": normalized_gini(test["ClaimAmount"], expected_loss),
        "test_rows": int(len(test)),
        "actual_claim_rate_test": float(test["HasClaim"].mean()),
        "average_predicted_pure_premium": float(np.mean(pure_premium_two_stage)),
        "average_actual_pure_premium": float(np.mean(actual_pure_premium)),
    }
    return pred_df, metrics, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_FILE)
    result = train_models(df, test_size=args.test_size)

    MODEL_BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["bundle"], MODEL_BUNDLE_FILE)
    result["predictions"].to_csv(PREDICTIONS_FILE, index=False)
    save_json(result["metrics"], METRICS_FILE)

    print(f"Saved model bundle to {MODEL_BUNDLE_FILE}")
    print(f"Saved predictions to {PREDICTIONS_FILE}")
    print(f"Saved metrics to {METRICS_FILE}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
