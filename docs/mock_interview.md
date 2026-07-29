# Mock Interview — LoanLens

**Q1: Walk me through your ML pipeline end to end.**
Raw Home Credit data goes through EDA, cleaning (fixing sentinel values 
like the DAYS_EMPLOYED placeholder, capping outliers), feature engineering 
(ratio features, EXT_SOURCE aggregation), then a Scikit-learn 
ColumnTransformer pipeline that imputes and encodes/scales. I trained and 
compared 5 models, selected LightGBM based on ROC-AUC/recall/F1 balance, 
tuned it with Optuna, added SHAP explainability, ran a fairness audit, and 
deployed it as a Streamlit app.

**Q2: Your dataset is 92% non-default, 8% default. Why is accuracy a bad 
metric here, and what did you use instead?**
A model that always predicts "no default" would score 92% accuracy while 
being completely useless. I used recall as my primary metric, since 
missing an actual defaulter is more costly than a false alarm on a safe 
applicant, alongside precision, F1, and ROC-AUC for a fuller picture. I 
handled the imbalance with `class_weight="balanced"` rather than SMOTE, 
given the dataset size and time constraints.

**Q3: You mentioned adding bureau data changed your metrics. What happened, 
and how did you decide what to do?**
Adding aggregated bureau.csv features improved ROC-AUC, F1, and precision, 
but recall dropped from 0.662 to 0.628. Since recall is the metric that 
matters most for this problem, I didn't promote that model to production — 
I kept the version with better recall and documented the bureau experiment 
as a future improvement, rather than shipping whichever model had the 
"bigger number."

**Q4: How do you know your preprocessing pipeline doesn't leak information 
from the test set into training?**
I fit the ColumnTransformer only on `X_train`, never on the full dataset 
or `X_test`. The imputer's medians/modes and the scaler's mean/std are 
learned exclusively from training data, then applied — not re-fit — to 
the test set during `.transform()`.

**Q5: What would you do differently if you rebuilt this project?**
I'd centralize feature engineering into one shared function from the 
start. I had a bug where two notebooks drifted out of sync on which 
columns existed, causing a ValueError — a shared function used everywhere 
would have made that structurally impossible. I'd also proactively 
consider working-directory assumptions before deploying, not after — I hit 
a FileNotFoundError on Streamlit Cloud because a path that worked locally 
didn't hold once deployed.
