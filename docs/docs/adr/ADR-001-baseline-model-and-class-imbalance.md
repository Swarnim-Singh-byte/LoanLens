# ADR-001: Baseline Model Choice and Class Imbalance Handling

## Context

LoanLens needed a first working end-to-end pipeline (Week 2 milestone) to validate 
that data flows correctly from raw CSV to a trained, evaluated model. The dataset has 
a known ~8.1% default rate (confirmed during EDA), making this a heavily imbalanced 
binary classification problem.

## Decision

We chose Logistic Regression as the baseline model, trained through a Scikit-Learn 
Pipeline with ColumnTransformer handling imputation, categorical encoding, and 
numerical scaling. No class balancing technique (e.g. SMOTE, class weighting) was 
applied in this first pass — the goal was to get a simple, interpretable, working 
pipeline first, and measure the actual impact of imbalance before introducing 
correction techniques.

## Consequences

**Positive:**
- Fast to train and interpret; validated the full pipeline end-to-end
- Established a clear, honest baseline before adding complexity

**Negative:**
- The model achieved 92% overall accuracy, but recall for the default class (TARGET=1) 
  was only 0.01 — it almost never correctly identifies an actual defaulter
- This confirms class imbalance is a real, measured problem for this dataset, not 
  just a theoretical concern from the design doc
- Accuracy alone is a misleading metric here; a model that predicted "no default" 
  for everyone would score similarly high on accuracy while being useless for the 
  business goal

## Alternatives Considered

- **SMOTE oversampling**: deferred to a later iteration, to first measure the 
  unmitigated baseline
- **Class-weighted Logistic Regression**: a natural next experiment, likely cheaper 
  than SMOTE to test first
- **Tree-based models (LightGBM, XGBoost)**: planned per the project roadmap, expected 
  to handle imbalance and non-linear feature relationships better than plain 
  Logistic Regression

## Next Steps

Try `class_weight='balanced'` in Logistic Regression and/or SMOTE as the next 
experiment, and compare Recall@K and the business cost matric alongside accuracy, 
per the evaluation plan in the design doc.
