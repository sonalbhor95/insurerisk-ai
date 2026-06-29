from __future__ import annotations

import warnings

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import MODEL_BUNDLE_FILE, PROCESSED_FILE, REPORTS_DIR
from src.features import make_X


def create_explainability_outputs(sample_rows: int = 3000) -> None:
    bundle = joblib.load(MODEL_BUNDLE_FILE)
    df = pd.read_csv(PROCESSED_FILE).sample(n=min(sample_rows, len(pd.read_csv(PROCESSED_FILE))), random_state=42)
    X = make_X(df)
    y = df["ClaimAmount"] / df["Exposure"]

    result = permutation_importance(
        bundle["tweedie_model"],
        X,
        y,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    importance = pd.DataFrame(
        {
            "feature": bundle["feature_columns"],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    # Optional SHAP. This may be slower for linear pipelines with one-hot encoders,
    # so failures are captured and the project still runs end-to-end.
    try:
        import shap
        transformed = bundle["tweedie_model"].named_steps["preprocess"].transform(X)
        model = bundle["tweedie_model"].named_steps["model"]
        explainer = shap.Explainer(model.predict, transformed[:200])
        shap_values = explainer(transformed[:200])
        shap_abs = pd.DataFrame({"mean_abs_shap": abs(shap_values.values).mean(axis=0)})
        shap_abs.to_csv(REPORTS_DIR / "shap_summary_values.csv", index=False)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"SHAP output skipped: {exc}")


def main():
    create_explainability_outputs()
    print("Saved explainability outputs to reports/.")


if __name__ == "__main__":
    main()
