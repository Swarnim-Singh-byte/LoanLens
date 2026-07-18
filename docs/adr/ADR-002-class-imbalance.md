# ADR-002: Handling Class Imbalance in Loan Default Prediction

## Context
The target variable is heavily imbalanced (~8.1% default rate). A naive
baseline logistic regression achieved 92% accuracy but only 1.3% recall on
the default class — essentially useless for the actual business problem
of catching risky applicants.

## Decision
Used `class_weight="balanced"` during model training rather than SMOTE,
given the dataset size (~300K rows) and project time constraints.

## Consequences
**Positive:** Recall improved from 1.3% to 67.7% with minimal added
complexity — no synthetic data generation needed, faster to implement
and iterate on.

**Negative:** Precision dropped substantially (60% → 16%), meaning more
false positives (safe applicants flagged as risky). ROC AUC stayed
roughly constant (~0.749), confirming this technique shifts the decision
threshold behavior rather than improving the model's underlying ability
to rank risk.

## Alternatives considered
- **SMOTE:** rejected for now due to computational cost at this data
  scale and time constraints; noted as a future improvement.
- **Threshold tuning:** a complementary technique, not yet implemented —
  could be combined with `class_weight` for finer control over the
  precision/recall tradeoff.