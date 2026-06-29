from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load_model_bundle, predict_policy
from src.config import MODEL_BUNDLE_FILE

app = FastAPI(title="InsureRisk AI API", version="1.0.0")
bundle = load_model_bundle(MODEL_BUNDLE_FILE) if MODEL_BUNDLE_FILE.exists() else None


class PolicyInput(BaseModel):
    Exposure: float = Field(default=0.75, gt=0, le=1.5)
    VehPower: int = Field(default=7, ge=1)
    VehAge: int = Field(default=4, ge=0)
    DrivAge: int = Field(default=42, ge=16)
    BonusMalus: int = Field(default=60, ge=1)
    Density: float = Field(default=1200, ge=0)
    Area: str = "C"
    VehBrand: str = "B2"
    VehGas: str = "Regular"
    Region: str = "R24"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}


@app.post("/predict")
def predict(payload: PolicyInput):
    global bundle
    if bundle is None:
        bundle = load_model_bundle(MODEL_BUNDLE_FILE)
    return predict_policy(payload.model_dump(), bundle)
