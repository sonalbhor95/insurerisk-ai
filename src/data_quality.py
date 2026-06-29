from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_FILE, REPORTS_DIR


def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add_check(name: str, value, status: str, details: str = ""):
        checks.append({"check": name, "value": value, "status": status, "details": details})

    add_check("row_count", len(df), "PASS" if len(df) > 1000 else "WARN", "Expected more than 1,000 rows.")
    add_check("duplicate_policy_ids", int(df["IDpol"].duplicated().sum()), "PASS" if df["IDpol"].duplicated().sum() == 0 else "FAIL")
    add_check("missing_values", int(df.isna().sum().sum()), "PASS" if df.isna().sum().sum() == 0 else "WARN")
    add_check("invalid_exposure", int(((df["Exposure"] <= 0) | (df["Exposure"] > 1.5)).sum()), "PASS" if ((df["Exposure"] <= 0) | (df["Exposure"] > 1.5)).sum() == 0 else "FAIL")
    add_check("negative_claim_count", int((df["ClaimNb"] < 0).sum()), "PASS" if (df["ClaimNb"] < 0).sum() == 0 else "FAIL")
    add_check("negative_claim_amount", int((df["ClaimAmount"] < 0).sum()), "PASS" if (df["ClaimAmount"] < 0).sum() == 0 else "FAIL")
    add_check("claim_rate", float(df["HasClaim"].mean()), "PASS", "Portfolio claim occurrence rate.")

    return pd.DataFrame(checks)


def main():
    df = pd.read_csv(PROCESSED_FILE)
    report = run_quality_checks(df)
    out = REPORTS_DIR / "data_quality_report.csv"
    report.to_csv(out, index=False)
    print(report)
    print(f"Saved data quality report to {out}")


if __name__ == "__main__":
    main()
