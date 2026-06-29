# InsureRisk AI: Claims Frequency, Severity & Premium Risk Engine

An end-to-end insurance machine learning project using **open French Motor Third-Party Liability claims data**.

This project is designed for a portfolio that targets **Data Scientist, Machine Learning Engineer, Senior Analytics, and Insurance/Finance analytics roles**. It is intentionally not a Kaggle notebook. It includes reproducible open-data ingestion, data validation, feature engineering, frequency/severity modeling, pure premium estimation, SHAP explainability, a Streamlit app, and a FastAPI scoring API.

---

## 1. Business problem

Insurance companies need to estimate how risky a policy is before pricing or renewal decisions. A useful pricing model should answer:

- How often is this policy expected to generate claims?
- If a claim happens, how severe could it be?
- What is the expected loss per unit exposure?
- Which risk drivers explain the prediction?
- How should policies be segmented into underwriting-style risk tiers?

This project models:

```text
Claim frequency = ClaimNb / Exposure
Claim severity  = ClaimAmount / ClaimNb, for policies with positive claims
Pure premium    = predicted frequency × predicted severity
Expected loss   = pure premium × exposure
```

---

## 2. Data source

Primary open data:

- `freMTPL2freq`: French motor third-party liability policy-level risk features and claim counts.
- `freMTPL2sev`: claim amount table linked by policy ID.

The scripts fetch the data directly from OpenML using scikit-learn:

```text
freMTPL2freq OpenML data_id = 41214
freMTPL2sev  OpenML data_id = 41215
```

No dataset file is included in this repo because the ingestion script downloads the data reproducibly.

---

## 3. Project architecture

```text
insurerisk_ai/
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
├── config/
│   └── project_config.yaml
│
├── data/
│   ├── raw/              # downloaded OpenML files saved here
│   └── processed/        # cleaned modeling dataset
│
├── models/               # trained joblib model bundle
│
├── reports/
│   ├── model_card.md
│   └── figures/
│
├── scripts/
│   └── run_pipeline.py
│
├── src/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_quality.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── features.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   └── test_metrics.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Setup

### Step 1 — Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the full pipeline

For a quick local run:

```bash
python scripts/run_pipeline.py --sample-size 60000
```

For a stronger portfolio run:

```bash
python scripts/run_pipeline.py --sample-size 250000
```

For full data, omit `--sample-size`, but expect longer training time.

---

## 5. What the pipeline does

### 5.1 Data ingestion

```bash
python -m src.data_ingestion --sample-size 60000
```

This downloads `freMTPL2freq` and `freMTPL2sev`, aggregates severity by policy ID, merges the data, cleans exposure and claim fields, and saves:

```text
data/processed/mtpl_claims_modeling.csv
```

### 5.2 Data quality checks

```bash
python -m src.data_quality
```

Creates:

```text
reports/data_quality_report.csv
```

Checks include row count, missing values, exposure validity, negative claims, and duplicate policy IDs.

### 5.3 Model training

```bash
python -m src.train
```

Trains three models:

1. **Poisson frequency model** for claim frequency.
2. **Gamma severity model** for claim severity.
3. **Tweedie pure premium model** as a single-stage benchmark.

Saves:

```text
models/insurerisk_model_bundle.joblib
reports/model_metrics.json
reports/policy_level_predictions.csv
```

### 5.4 Explainability

```bash
python -m src.explain
```

Creates permutation importance and SHAP-compatible outputs when SHAP is installed.

### 5.5 Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app lets a user input a policy profile and returns:

- Predicted claim frequency
- Predicted severity
- Pure premium
- Expected loss
- Risk tier
- Portfolio risk distribution

### 5.6 FastAPI scoring service

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Example payload:

```json
{
  "Exposure": 0.75,
  "VehPower": 7,
  "VehAge": 4,
  "DrivAge": 42,
  "BonusMalus": 60,
  "Density": 1200,
  "Area": "C",
  "VehBrand": "B2",
  "VehGas": "Regular",
  "Region": "R24"
}
```

---

## 6. Suggested portfolio write-up

**Short portfolio description**

Built an end-to-end insurance risk pricing engine using open motor claims data. The system predicts claim frequency and claim severity, estimates policy-level pure premium, segments policies into underwriting-style risk tiers, and explains risk drivers using model interpretation techniques.

**Resume bullet**

Built an insurance risk pricing engine using open motor claims data, combining Poisson/Gamma/Tweedie modeling, pure premium estimation, policy-level risk segmentation, and explainability to support underwriting-style decision making.

---

## 7. Next upgrades

- Add LightGBM or XGBoost benchmark models.
- Add calibration curves and Gini/lift charts.
- Add MLflow experiment tracking.
- Deploy Streamlit app to Streamlit Community Cloud.
- Deploy FastAPI to Render, Railway, or Azure App Service.
- Add model drift checks by region, vehicle age, driver age, and risk tier.
