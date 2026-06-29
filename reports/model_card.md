# Model Card — InsureRisk AI

## Intended use

This project is for portfolio and learning purposes. It demonstrates open-data insurance risk modeling and is not intended for production pricing decisions.

## Dataset

French Motor Third-Party Liability claims data from OpenML/CASdatasets.

## Target definitions

- Frequency target: `ClaimNb / Exposure`
- Severity target: `ClaimAmount / ClaimNb` for policies with positive claims
- Pure premium target: `ClaimAmount / Exposure`

## Models

- PoissonRegressor for frequency
- GammaRegressor for severity
- TweedieRegressor as pure premium benchmark

## Evaluation

Metrics include MAE, RMSE, AUC for claim occurrence ranking, normalized Gini for expected loss ranking, and risk-tier summaries.

## Limitations

- Dataset is historical and geographically specific.
- Data fields are limited compared to real underwriting systems.
- Fairness, regulatory compliance, and pricing governance are not fully addressed.
- The model should not be used for real pricing decisions.
