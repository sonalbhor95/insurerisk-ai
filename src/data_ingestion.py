from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import (
    DATA_RAW_DIR,
    PROCESSED_FILE,
    FREQ_OPENML_ID,
    SEV_OPENML_ID,
    RANDOM_STATE,
)


def fetch_fremtpl2(sample_size: Optional[int] = None, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Fetch and merge freMTPL2 frequency/severity data from OpenML.

    The frequency table is policy-level. The severity table is claim-level and is
    aggregated to policy level before joining.
    """
    print("Downloading freMTPL2freq from OpenML...")
    freq = fetch_openml(data_id=FREQ_OPENML_ID, as_frame=True, parser="auto").frame
    print("Downloading freMTPL2sev from OpenML...")
    sev = fetch_openml(data_id=SEV_OPENML_ID, as_frame=True, parser="auto").frame

    freq.columns = [str(c).strip() for c in freq.columns]
    sev.columns = [str(c).strip() for c in sev.columns]

    if "IDpol" not in freq.columns or "IDpol" not in sev.columns:
        raise ValueError("Expected IDpol column in both freMTPL2 tables.")

    freq["IDpol"] = freq["IDpol"].astype(int)
    sev["IDpol"] = sev["IDpol"].astype(int)
    sev["ClaimAmount"] = pd.to_numeric(sev["ClaimAmount"], errors="coerce").fillna(0.0)

    sev_policy = (
        sev.groupby("IDpol", as_index=False)["ClaimAmount"]
        .sum()
        .rename(columns={"ClaimAmount": "TotalClaimAmount"})
    )

    df = freq.merge(sev_policy, on="IDpol", how="left")
    df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0.0)

    numeric_cols = ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"TotalClaimAmount": "ClaimAmount"})

    df = clean_claims(df)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    freq.to_csv(DATA_RAW_DIR / "freMTPL2freq_raw.csv", index=False)
    sev.to_csv(DATA_RAW_DIR / "freMTPL2sev_raw.csv", index=False)
    return df


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["IDpol"])

    df = df[df["Exposure"].notna()]
    df = df[(df["Exposure"] > 0) & (df["Exposure"] <= 1.5)]

    df["ClaimNb"] = df["ClaimNb"].fillna(0).clip(lower=0)
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0).clip(lower=0)

    # If the severity table has no amount for a non-zero count, make the target consistent.
    inconsistent = (df["ClaimAmount"] <= 0) & (df["ClaimNb"] > 0)
    df.loc[inconsistent, "ClaimNb"] = 0

    # Cap extreme claim amounts for a stable portfolio project demo.
    positive_amounts = df.loc[df["ClaimAmount"] > 0, "ClaimAmount"]
    if not positive_amounts.empty:
        cap = positive_amounts.quantile(0.995)
        df["ClaimAmount"] = df["ClaimAmount"].clip(upper=cap)

    df["ClaimFrequency"] = df["ClaimNb"] / df["Exposure"]
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)

    for col in ["Area", "VehBrand", "VehGas", "Region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for faster local runs.")
    args = parser.parse_args()

    df = fetch_fremtpl2(sample_size=args.sample_size)
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"Saved processed modeling data to {PROCESSED_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()
