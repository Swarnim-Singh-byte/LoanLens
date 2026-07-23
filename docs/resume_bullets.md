# Resume Bullets — LoanLens

- Built and deployed an end-to-end loan default prediction system (LightGBM, 
  SHAP, Streamlit) achieving 0.762 ROC-AUC; compared 5 ML algorithms and 
  tuned hyperparameters via Optuna to select the best-performing model.

- Designed and shipped a fairness audit mini-extension computing Disparate 
  Impact Ratio and Equal Opportunity Difference across protected attributes, 
  informing a data-driven decision to prioritize recall over a marginal 
  ROC-AUC gain from an experimental feature addition.

- Implemented a leakage-free Scikit-learn preprocessing pipeline 
  (ColumnTransformer, imputation, encoding) and integrated SHAP-based 
  local/global explainability into a live, publicly deployed Streamlit 
  application.
