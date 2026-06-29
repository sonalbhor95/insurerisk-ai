# InsureRisk AI: Claims Frequency, Severity & Premium Risk Engine

An end-to-end insurance machine learning project using **open French Motor Third-Party Liability claims data**.

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
