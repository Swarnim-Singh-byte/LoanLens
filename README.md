
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Project-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)# LoanLens


> Explainable Loan Default Risk Prediction using Machine Learning
**Problem Statement Code:** I1 — Tabular ML Zoo
**Segment:** Foundations of Applied Machine Learning
**Author:** Swarnim Singh (Registration No. 12410315)

## Overview

LoanLens is an end-to-end machine learning project that predicts the probability of loan default using the Home Credit Default Risk dataset.

The project covers:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Comparison
- SHAP Explainability
- Fairness Auditing
- Streamlit Deployment

---

## Current Status

🚧 Project currently in development as part of the Futurense Summer Internship 2026.

Current phase:
- Design Document Completed
- Dataset Selection Completed
- EDA In Progress

## Problem Statement

Financial institutions need to identify high-risk applicants before loan approval.

LoanLens predicts whether an applicant is likely to default within 90 days using only information available at application time.

---

## Project Architecture

```text
Raw Data
   ↓
Exploratory data analysis
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
SHAP Explainability
   ↓
Fairness Audit
   ↓
Streamlit App
```

---



## Tech Stack

| Category | Tools |
|-----------|---------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Missingno |
| Preprocessing | Scikit-Learn Pipeline, ColumnTransformer |
| ML Models | Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost |
| Hyperparameter Tuning | Optuna |
| Imbalanced Learning | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Fairness | Fairlearn |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |
---

## Repository Structure

```text
LoanLens/
├── docs/
├── notebooks/
├── src/
├── app/
├── tests/
└── README.md
```

---

## Roadmap

- [x] Design Document
- [ ] EDA
- [ ] Feature Engineering
- [ ] Baseline Models
- [ ] Hyperparameter Tuning
- [ ] SHAP Analysis
- [ ] Fairness Audit
- [ ] Streamlit Deployment

---

## Documentation

- Design Document: `docs/design_doc.md`

---
## What I Learned This Week

- EDA is "detective work" — before touching models, systematically asking what data I have, whether it's healthy, and what the target looks like, rather than jumping straight to feature engineering.
- The mean of a binary (0/1) column equals its proportion of 1s — a fast way to read class imbalance (confirmed ~8% default rate) directly off `df.describe()`.
- `missingno` visualizes patterns of missingness, not just counts — useful for spotting when features are missing together in structural blocks, which matters for imputation strategy later.
- `df.describe()` flags data quality issues beyond summary stats — comparing mean vs median reveals skew, and min/max catches anomalies like a known `DAYS_EMPLOYED` placeholder value and an outlier `AMT_INCOME_TOTAL` of 117 million.
- Git hygiene matters as much as the analysis — learned to properly exclude large data files and OS-specific junk (`.DS_Store`) from version control using `.gitignore`, and to verify staged changes carefully before committing.
- And more on how a bad model on good eda is still better than a good model on bad eda.(Have also added eveything i learnt understood implemented why how in the markdown of notebook).
## Author

**Swarnim Singh**
B.Tech CSE (AI & Data Engineering)

