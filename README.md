![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Project-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

# LoanLens

> Explainable Loan Default Risk Prediction using Machine Learning

**Problem Statement Code:** I1 — Tabular ML Zoo
**Segment:** Foundations of Applied Machine Learning
**Author:** Swarnim Singh (Registration No. 12410315)

## Demo

🔗 **Live app:** [(https://loanlens-swarnim.streamlit.app/)]

The app takes applicant details (income, credit amount, employment history,
external credit bureau scores) and returns a default risk score along with
a SHAP-based explanation of which factors drove the prediction.

## Overview

LoanLens is an end-to-end machine learning project that predicts the probability
of loan default using the Home Credit Default Risk dataset.

The project covers:
- Exploratory Data Analysis (EDA)
- Data Cleaning & Feature Engineering
- Model Comparison (5 algorithms)
- SHAP Explainability (global + local)
- Fairness Auditing (mini-extension)
- Streamlit Deployment

---

## Current Status

🚧 Project built as part of the Futurense Summer Internship 2026.

Current phase:
- ✅ Design Document
- ✅ Dataset selection, EDA, cleaning
- ✅ Feature engineering
- ✅ 5 models trained and compared — LightGBM selected as best performer
- ✅ SHAP explainability integrated
- ✅ Fairness Audit completed (mini-extension)
- ✅ Streamlit app built and **deployed live**
- 🔲 Threshold tuning / further imbalance refinement (planned)
- 🔲 Model card finalization

## Problem Statement

Financial institutions need to identify high-risk applicants before loan approval.

LoanLens predicts whether an applicant is likely to default within 90 days,
using only information available at application time, and explains *why*
the model made that prediction — not just the score itself.

---

## Project Architecture

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Preprocessing Pipeline (impute → scale/encode)
   ↓
Model Training (5 algorithms compared)
   ↓
Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
   ↓
SHAP Explainability
   ↓
Fairness Audit
   ↓
Streamlit App (live prediction + explanation)
```

See `docs/architecture.png` for a visual diagram.

---

## Tech Stack

| Category | Tools |
|-----------|---------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Missingno |
| Preprocessing | Scikit-Learn Pipeline, ColumnTransformer |
| ML Models | Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost |
| Imbalanced Learning | `class_weight="balanced"` (SMOTE explored, see ADR-002) |
| Explainability | SHAP (TreeExplainer) |
| Fairness | Custom disparate impact & equal opportunity audit |
| Deployment | Streamlit, Streamlit Cloud |
| Testing | Pytest |
| Version Control | Git, GitHub |

---

## Quickstart

### Prerequisites
- Python 3.11+
- pip

### Install
```bash
git clone https://github.com/Swarnim-Singh-byte/LoanLens.git
cd LoanLens
pip install -r app/requirements.txt --break-system-packages
```

### Run the app locally
```bash
cd app
streamlit run app.py
```
Opens at `http://localhost:8501`.

### Run tests
```bash
pytest tests/ -v
```

---

## Repository Structure

```text
LoanLens/
├── app/            # Streamlit app + saved model artifacts
├── data/           # Dataset (raw + cleaned + feature-engineered)
├── docs/           # ADRs, model card, architecture diagram
│   └── adr/
├── notebooks/      # EDA, cleaning, preprocessing, modeling, SHAP, fairness
├── src/            # Reusable pipeline scripts
├── tests/          # Pytest test suite
└── README.md
```

---

## Data Sources

Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(Kaggle competition dataset). ~307,511 applicants, 122 raw features.

---

## Architecture Decision Records

Key technical decisions are documented in `docs/adr/`:
- **ADR-001:** Baseline model and class imbalance discovery
- **ADR-002:** Handling class imbalance (`class_weight` vs SMOTE)
- **ADR-003:** Model selection (LightGBM vs 4 alternatives)

---

## Mini-Extension: Fairness Audit

Per the project's mini-extension requirement, a fairness audit was added on
top of the core model, evaluating:
- **Disparate Impact Ratio** — whether one group is flagged as "high risk"
  at a disproportionate rate compared to another
- **Equal Opportunity Difference** — whether the model catches actual
  defaulters equally well across groups

Protected attributes analyzed: `CODE_GENDER`, `AGE_YEARS` (bucketed).

**Why this matters:** fairness auditing is table-stakes for any real-world
credit risk model — a model that's accurate overall but systematically
disadvantages a protected group is not production-ready, regardless of its
ROC-AUC. Full findings and discussion: `notebooks/07_fairness_audit.ipynb`.

---

## Model Performance Summary

Five models were compared; LightGBM was selected for deployment.

| Model | ROC-AUC | Recall | F1 |
|-------|---------|--------|-----|
| Logistic Regression (balanced) | 0.749 | 0.682 | 0.261 |
| Random Forest | 0.72 | 0.001 | 0.002 |
| **LightGBM (selected)** | **0.759** | **0.677** | **0.270** |
| XGBoost | 0.745 | 0.616 | 0.272 |
| CatBoost | 0.750 | 0.654 | 0.269 |

Full comparison and reasoning: `notebooks/05_model_comparison.ipynb` and
`docs/adr/ADR-003-model-selection.md`.

---

## Known Limitations

- Precision/recall tradeoff is currently tuned via `class_weight="balanced"`
  only; explicit threshold tuning has not yet been explored (see ADR-002).
- The Streamlit app's input form exposes a subset of features; remaining
  columns are filled with training-set medians/modes rather than user input.
- Fairness audit currently covers 2 protected attributes independently;
  intersectional fairness (e.g. gender × age combined) is not yet analyzed.

---

## What I'd Do Differently

If I started this project over, I'd centralize feature engineering into a
single reusable function from the start instead of duplicating the logic
across notebooks — I hit a `ValueError` in the SHAP notebook specifically
because two notebooks had drifted out of sync on which columns existed.
I'd also set up the Streamlit app's file paths correctly the first time
(using `os.path.dirname(__file__)`) instead of discovering the
working-directory assumption broke only after deploying to the cloud.

---

## Documentation

- Design Document: `docs/design_doc.md`
- Model Card: `docs/model_card.md`
- ADRs: `docs/adr/`

---

## What I Learned This Week

- EDA is "detective work" — before touching models, systematically asking
  what data I have, whether it's healthy, and what the target looks like,
  rather than jumping straight to feature engineering.
- The mean of a binary (0/1) column equals its proportion of 1s — a fast
  way to read class imbalance (confirmed ~8% default rate) directly off
  `df.describe()`.
- `missingno` visualizes patterns of missingness, not just counts — useful
  for spotting when features are missing together in structural blocks,
  which matters for imputation strategy later.
- `df.describe()` flags data quality issues beyond summary stats —
  comparing mean vs median reveals skew, and min/max catches anomalies
  like a known `DAYS_EMPLOYED` placeholder value and an outlier
  `AMT_INCOME_TOTAL` of 117 million.
- Git hygiene matters as much as the analysis — learned to properly
  exclude large data files and OS-specific junk (`.DS_Store`) from
  version control using `.gitignore`, and to verify staged changes
  carefully before committing.
- A bad model on good EDA is still better than a good model on bad EDA.
- Model accuracy can be dangerously misleading on imbalanced datasets —
  our baseline hit 92% accuracy but only 0.01 recall on the minority
  (default) class.
- Preserving the "signal" in anomalous/missing data (e.g. a
  `DAYS_EMPLOYED_ANOM` flag) before cleaning it is important — you can fix
  a value without throwing away what its absence/anomaly told you.
- Building a Scikit-Learn Pipeline with ColumnTransformer keeps
  preprocessing consistent between training and future prediction, and
  helps avoid data leakage.
- `class_weight="balanced"` and SHAP's TreeExplainer output values in
  log-odds space, not raw probability — useful to know before trying to
  cross-check chart values against a displayed percentage.
- Relative file paths that work locally can silently break once deployed
  to the cloud, if the app's working directory isn't the same as where
  it's launched from — worth anchoring paths to the script's own location
  (`os.path.dirname(__file__)`) rather than assuming a working directory.

## Author

**Swarnim Singh**
B.Tech CSE (AI & Data Engineering)
