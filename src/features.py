from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
TARGET_COLUMNS = ["ClaimNb", "ClaimAmount", "ClaimFrequency", "PurePremium", "HasClaim"]
ID_COLUMNS = ["IDpol"]


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    features = numeric + categorical
    return features, numeric, categorical


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
    categorical_pipe = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    features, _, _ = get_feature_columns(df)
    return df[features].copy()
