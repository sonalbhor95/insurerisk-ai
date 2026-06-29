from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.sklearn
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None

from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_benchmark_models(
    processed_data_path: str | Path = "data/processed/mtpl_claims_modeling.csv",
) -> dict:
    df = pd.read_csv(processed_data_path)

    drop_cols = [
        "IDpol",
        "ClaimNb",
        "ClaimAmount",
        "pure_premium",
        "frequency_target",
        "severity_target",
        "has_claim",
    ]

    df["has_claim"] = (df["ClaimNb"] > 0).astype(int)

    X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
    y_freq = df["ClaimNb"]
    y_claim = df["has_claim"]

    X_train, X_test, y_freq_train, y_freq_test, y_claim_train, y_claim_test = train_test_split(
        X,
        y_freq,
        y_claim,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_claim,
    )

    preprocessor = _build_preprocessor(X_train)

    models = {
        "lightgbm_frequency": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LGBMRegressor(
                        objective="poisson",
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost_frequency": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        objective="count:poisson",
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "lightgbm_claim_classifier": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        num_leaves=31,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost_claim_classifier": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    metrics = {}

    mlflow.set_experiment("insurerisk-ai")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("n_rows", len(df))

            if "classifier" in name:
                model.fit(X_train, y_claim_train)
                pred_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_claim_test, pred_prob)

                metrics[name] = {
                    "task": "claim_probability",
                    "roc_auc": float(auc),
                }

                mlflow.log_metric("roc_auc", auc)

            else:
                model.fit(X_train, y_freq_train)
                pred = model.predict(X_test)
                pred = np.clip(pred, 0, None)

                if root_mean_squared_error is not None:
                    rmse = root_mean_squared_error(y_freq_test, pred)
                else:
                    rmse = mean_squared_error(y_freq_test, pred) ** 0.5

                mae = mean_absolute_error(y_freq_test, pred)

                metrics[name] = {
                    "task": "claim_frequency",
                    "rmse": float(rmse),
                    "mae": float(mae),
                }

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

            joblib.dump(model, Path(MODELS_DIR) / f"{name}.joblib")
            mlflow.sklearn.log_model(model, artifact_path=name)

    output_path = Path(REPORTS_DIR) / "benchmark_model_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Benchmark metrics saved to {output_path}")
    return metrics


if __name__ == "__main__":
    train_benchmark_models()