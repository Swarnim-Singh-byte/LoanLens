# LoanLens

> **An Explainable Machine Learning System for Loan Default Risk Prediction**

Financial institutions rely on accurate risk assessment to minimize loan defaults and improve lending decisions. LoanLens is an end-to-end machine learning project that predicts the probability of loan default using historical applicant data while emphasizing explainability, fairness, and reproducibility.

The project follows the complete machine learning lifecycle—from exploratory data analysis and feature engineering to model deployment through an interactive web application.

---

## Project Information

| Field                   | Value                                   |
| ----------------------- | --------------------------------------- |
| **Author**              | Swarnim Singh                           |
| **Registration Number** | 12410315                                |
| **Segment**             | Foundations of Applied Machine Learning |
| **Problem Code**        | I1 — Tabular ML Zoo                     |
| **Internship**          | Futurense Summer Internship 2026        |
| **Duration**            | June 2026 – July 2026                   |

---

# Problem Statement

LoanEasy, a fictional fintech company, wants to predict whether a customer will default within 90 days of receiving a loan using only information available at the time of application.

The challenge involves:

* Large-scale relational datasets
* Highly imbalanced classes
* Missing and noisy data
* Explainability requirements for lending decisions
* Fairness concerns across demographic groups

LoanLens addresses these challenges through a transparent and production-inspired machine learning pipeline that prioritizes both predictive performance and responsible AI practices.

---

# Project Objectives

### Core Objectives

* Build a complete machine learning workflow from raw data to deployment
* Engineer meaningful features from multiple relational datasets
* Compare multiple classification algorithms using consistent evaluation metrics
* Prevent data leakage through proper validation strategies
* Optimize model performance through hyperparameter tuning
* Deploy a user-friendly web application for risk assessment

### Explainability Objectives

* Generate global and local feature explanations using SHAP
* Identify the most influential risk factors behind predictions
* Improve model transparency for business stakeholders

### Fairness Objectives

* Evaluate model performance across protected demographic groups
* Measure potential bias using established fairness metrics
* Document trade-offs and ethical considerations

---

# Dataset

## Source

**Home Credit Default Risk Dataset**

Kaggle Competition Dataset

https://www.kaggle.com/c/home-credit-default-risk

---

## Tables Used

| Table                       | Description                                     |
| --------------------------- | ----------------------------------------------- |
| `application_train.csv`     | Primary applicant information and target labels |
| `bureau.csv`                | Credit history from external institutions       |
| `previous_application.csv`  | Historical loan application records             |
| `installments_payments.csv` | Installment payment behavior                    |
| `credit_card_balance.csv`   | Credit card utilization history                 |
| `POS_CASH_balance.csv`      | Point-of-sale loan information                  |

---

## Target Variable

| Value | Meaning     |
| ----- | ----------- |
| `0`   | Non-default |
| `1`   | Default     |

**Target Column:** `TARGET`

### Dataset Characteristics

* ~300,000 loan applications
* Highly imbalanced dataset
* Approximately 8% default rate
* Multiple relational tables
* Significant missing values
* Real-world credit risk prediction problem

---

# Proposed Architecture

```text
Raw Data Sources (6 Tables)
            │
            ▼
Exploratory Data Analysis
            │
            ▼
Data Cleaning & Preprocessing
            │
            ▼
Feature Engineering
            │
            ▼
Train / Validation / Test Split
(Time-Based Validation)
            │
            ▼
Model Training
(Logistic Regression,
 Random Forest,
 LightGBM,
 XGBoost,
 CatBoost)
            │
            ▼
Hyperparameter Optimization
(Optuna)
            │
            ▼
Model Evaluation
(AUC, Recall@K,
 Calibration,
 Business Cost Matrix)
            │
            ▼
SHAP Explainability
(Global + Local)
            │
            ▼
Fairness Audit
            │
            ▼
Streamlit Deployment
            │
            ▼
Risk Prediction Dashboard
```

---

# Technology Stack

| Component             | Technology                     | Purpose                    |
| --------------------- | ------------------------------ | -------------------------- |
| Programming Language  | Python 3.11                    | Core development           |
| Data Processing       | Pandas, NumPy                  | Data manipulation          |
| Visualization         | Matplotlib, Seaborn, Missingno | Exploratory analysis       |
| Machine Learning      | Scikit-Learn                   | Modeling framework         |
| Gradient Boosting     | LightGBM, XGBoost, CatBoost    | Advanced models            |
| Hyperparameter Tuning | Optuna                         | Automated optimization     |
| Explainability        | SHAP                           | Model interpretation       |
| Fairness Analysis     | Fairlearn                      | Bias assessment            |
| Imbalanced Learning   | SMOTE (imbalanced-learn)       | Class balancing            |
| Deployment            | Streamlit                      | Web application            |
| Version Control       | Git & GitHub                   | Collaboration and tracking |

---

# Machine Learning Pipeline

## Feature Engineering

Planned features include:

* Bureau credit history aggregations
* Debt-to-income ratio
* Loan-to-income ratio
* Credit card utilization metrics
* Installment payment behavior indicators
* Employment duration features
* Age-based features
* High-cardinality categorical encodings

Target outcome: **50+ engineered features** suitable for risk modeling.

---

## Validation Strategy

To avoid data leakage and better simulate real-world deployment, a strict time-based validation strategy will be used.

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

**Important:** No random shuffling will be performed.

---

## Models Evaluated

1. Logistic Regression
2. Random Forest
3. LightGBM
4. XGBoost
5. CatBoost

The best-performing model will be selected based on validation performance and business relevance.

---

## Evaluation Metrics

Performance will be measured using:

* ROC-AUC
* Precision
* Recall
* F1 Score
* Recall@K
* Calibration Curves
* Confusion Matrix
* Business Cost Matrix

### Business Assumption

A false negative (approving a risky borrower) is significantly more expensive than a false positive (rejecting a safe borrower).

Therefore, recall-focused evaluation will receive special attention.

---

## Hyperparameter Optimization

The final candidate model will be optimized using **Optuna**.

Planned configuration:

* 100 optimization trials
* Validation AUC as objective metric
* Automated search space exploration

---

# Mini-Extension: Fairness Audit

To align with responsible AI principles, LoanLens will include a fairness evaluation component.

### Protected Attributes

* Gender (`CODE_GENDER`)
* Age Groups (`DAYS_BIRTH`)
* Region (`REGION_RATING_CLIENT`)

### Fairness Metrics

* Disparate Impact Ratio
* Equal Opportunity Difference

### Deliverable

A dedicated fairness section within the model card documenting:

* Metric results
* Observed disparities
* Potential causes
* Limitations and trade-offs

---

# Repository Structure

```text
LoanLens/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
│
├── app/
│   └── streamlit_app.py
│
├── docs/
│   ├── design_doc.md
│   ├── model_card.md
│   └── fairness_report.md
│
├── tests/
├── requirements.txt
└── README.md
```

---

# Deliverables

* Public GitHub Repository
* EDA Report
* Feature Engineering Pipeline
* Model Training Pipeline
* Fairness Audit Report
* SHAP Explainability Analysis
* Model Card
* Streamlit Application
* Live Deployment URL
* Technical Documentation
* 3-Minute Loom Walkthrough

---

# Project Roadmap

* [x] Design Document
* [ ] Exploratory Data Analysis
* [ ] Data Cleaning
* [ ] Feature Engineering
* [ ] Baseline Models
* [ ] Model Comparison
* [ ] Hyperparameter Optimization
* [ ] SHAP Explainability
* [ ] Fairness Audit
* [ ] Streamlit Deployment
* [ ] Documentation & Final Review

---

# Future Enhancements

* Automated retraining pipelines
* Model drift detection
* MLOps integration using Docker and CI/CD
* Real-time scoring APIs
* Ensemble modeling
* A/B testing of competing models
* Production monitoring dashboards

---

## Success Criteria

The project will be considered successful if it:

* Achieves strong predictive performance on unseen data
* Produces interpretable model explanations
* Demonstrates fairness awareness and transparency
* Is fully reproducible and deployable
* Can be effectively used by non-technical stakeholders

---

**LoanLens aims to combine predictive accuracy, explainability, and fairness to support responsible lending decisions in modern financial systems.**
