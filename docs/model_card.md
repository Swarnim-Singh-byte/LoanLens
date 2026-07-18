# Model Card – LoanLens

## Model Details

| Field | Value |
|-------|-------|
| Model Name | LoanLens |
| Version | v1.0 |
| Model Type | LightGBM Classifier |
| Framework | LightGBM |
| Language | Python 3.11 |
| Deployment | Streamlit |

---

## Purpose

LoanLens predicts whether a loan applicant is likely to default using information available at the time of application.

The model is intended to assist financial institutions by providing a risk score and an explanation for each prediction. It is designed as a decision-support tool and should not be used as the sole basis for approving or rejecting loan applications.

---

## Dataset

**Dataset:** Home Credit Default Risk (Kaggle)

### Features

- Applicant demographic information
- Financial information
- Credit history
- Bureau records
- Engineered ratio features

### Target

- **0** → No Default
- **1** → Default

---

## Data Preprocessing

- Missing value imputation (Median / Most Frequent)
- One-Hot Encoding
- Standard Scaling
- Train/Test Split (80/20)

---

## Feature Engineering

- Credit-to-Income Ratio
- Annuity-to-Income Ratio
- Age (Years)
- Employment Years
- Missing-value indicators
- External credit source features

---

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- ✅ LightGBM (Selected)

---

## Performance

Primary evaluation metrics:

- ROC-AUC
- Recall
- Precision
- F1 Score

LightGBM achieved the best overall balance between discrimination ability and recall and was selected as the final production model.

---

## Explainability

The project integrates SHAP (SHapley Additive Explanations).

Supported explanations:

- Global Feature Importance
- Local Prediction Explanations

---

## Fairness

A fairness audit was conducted to evaluate potential bias across protected groups.

The analysis examined fairness metrics and highlighted areas where prediction disparities may occur.

---

## Intended Use

Suitable for:

- Credit risk assessment
- Educational purposes
- Machine learning demonstrations

Not intended for:

- Fully automated lending decisions
- High-risk financial decisions without human review

---

## Limitations

- Performance depends on the quality of input data.
- Dataset imbalance may influence predictions.
- Fairness metrics should be periodically re-evaluated.
- The model should be retrained as new applicant data becomes available.

---

## Future Improvements

- Threshold optimization
- Automated retraining
- Expanded fairness evaluation
- Hyperparameter optimization
- Model monitoring and drift detection
