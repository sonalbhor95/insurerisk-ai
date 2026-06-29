from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import MODEL_BUNDLE_FILE, PREDICTIONS_FILE, METRICS_FILE
from src.predict import load_model_bundle, predict_policy
from src.utils import load_json

st.set_page_config(page_title="InsureRisk AI", page_icon="🚗", layout="wide")

st.title("InsureRisk AI: Insurance Claims Pricing & Risk Scoring")
st.caption("Open-data motor insurance risk model: frequency × severity = pure premium")

@st.cache_resource
def get_bundle():
    return load_model_bundle(MODEL_BUNDLE_FILE)

bundle = get_bundle()

with st.sidebar:
    st.header("Policy Input")
    exposure = st.slider("Exposure", 0.05, 1.0, 0.75, 0.05)
    veh_power = st.slider("Vehicle Power", 4, 15, 7)
    veh_age = st.slider("Vehicle Age", 0, 30, 4)
    driv_age = st.slider("Driver Age", 18, 95, 42)
    bonus_malus = st.slider("Bonus Malus", 50, 230, 60)
    density = st.number_input("Population Density", min_value=1, max_value=30000, value=1200)
    area = st.selectbox("Area", ["A", "B", "C", "D", "E", "F"], index=2)
    veh_brand = st.selectbox("Vehicle Brand", ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"], index=1)
    veh_gas = st.selectbox("Vehicle Gas", ["Regular", "Diesel"], index=0)
    region = st.selectbox("Region", ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"], index=4)

policy = {
    "Exposure": exposure,
    "VehPower": veh_power,
    "VehAge": veh_age,
    "DrivAge": driv_age,
    "BonusMalus": bonus_malus,
    "Density": density,
    "Area": area,
    "VehBrand": veh_brand,
    "VehGas": veh_gas,
    "Region": region,
}

prediction = predict_policy(policy, bundle)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk Tier", prediction["risk_tier"])
c2.metric("Pure Premium", f"{prediction['pred_pure_premium_two_stage']:.2f}")
c3.metric("Expected Loss", f"{prediction['expected_loss']:.2f}")
c4.metric("Expected Claims", f"{prediction['expected_claim_count']:.4f}")

st.subheader("Prediction Details")
st.json(prediction)

st.subheader("Portfolio Model Performance")
if METRICS_FILE.exists():
    st.json(load_json(METRICS_FILE))
else:
    st.info("Run the training pipeline to generate metrics.")

st.subheader("Risk Tier Distribution")
if PREDICTIONS_FILE.exists():
    preds = pd.read_csv(PREDICTIONS_FILE)
    st.bar_chart(preds["risk_tier"].value_counts())
    st.dataframe(preds.head(50))
else:
    st.info("Run the training pipeline to generate policy-level predictions.")
