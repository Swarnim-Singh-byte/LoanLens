# ADR-003: Model Selection for Final Deployment

## Context
Five models were trained and evaluated: Logistic Regression, Random
Forest, LightGBM, XGBoost, and CatBoost — all using `class_weight="balanced"`
where applicable, on the same feature-engineered dataset.

## Decision
LightGBM was selected for deployment.

## Consequences
**Positive:** Highest ROC-AUC (0.759), strong recall (0.677), competitive
F1 (0.270) — best overall balance across the metrics that matter for this
imbalanced credit-risk problem. Also fast to train and natively supports
SHAP's TreeExplainer for exact, efficient explainability.

**Negative:** Less interpretable than logistic regression out of the box
— addressed by integrating SHAP for both global and local explainability
in the deployed app.

## Alternatives considered
- **Random Forest:** rejected — highest accuracy (0.919) but recall of
  only 0.001, meaning it essentially never predicted a default. This
  confirms accuracy is a misleading metric for imbalanced datasets.
- **XGBoost:** close competitor — highest F1 and precision among balanced
  models, but lower recall (0.616) than LightGBM. A reasonable alternative
  if precision were prioritized over recall.
- **CatBoost:** stable, balanced results but did not outperform LightGBM
  on ROC-AUC or recall.
- **Logistic Regression:** highest raw recall (0.682) but weaker ROC-AUC
  and precision — retained as the interpretable baseline for comparison.