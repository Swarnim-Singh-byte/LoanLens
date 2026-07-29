# LoanLens — 3rd Year Extension Roadmap

## What this project is today

LoanLens is a deployed, explainable loan default risk predictor: 5 models compared, LightGBM selected and hyperparameter-tuned, SHAP explainability integrated, a fairness audit completed as a mini-extension, and a live Streamlit app serving predictions with per-applicant explanations.

## The arc: where this could be by 3rd year internship (May 2027)

By 3rd year, I want LoanLens to be a production-shaped system, not just a working demo: multi-table feature integration properly wired end-to-end (not just experimented with), a retraining pipeline that can be re-run on new data without manual notebook surgery, threshold tuning that's actually configurable rather than hardcoded, and a fairness audit that runs automatically on every retrain rather than as a one-time notebook analysis. The goal is a project I could walk a 3rd-year internship interviewer through end-to-end, including the mistakes and tradeoffs, not just the polished final numbers.

## 3rd Year Semester Plan (Aug 2026 - Dec 2026)

### Milestone 1 (Aug-Sep 2026): Multi-Table Feature Integration
- What I'll add: Properly wire bureau.csv, previous_application.csv, and installments_payments.csv into the pipeline, addressing the recall tradeoff I found this year (likely via threshold tuning combined with the new features, not accepting the tradeoff as-is)
- Tools I'll learn: Feature stores (or a lightweight equivalent), more advanced pandas aggregation patterns for multi-table joins
- Time commitment: 4-5 hours/week
- Done looks like: Bureau + previous application features live in the deployed app, with recall equal to or better than the current baseline

### Milestone 2 (Oct-Nov 2026): Retraining Pipeline
- What I'll add: A script (not a manual notebook) that can retrain the model on updated data, re-run the fairness audit, and flag if performance degrades
- Tools I'll learn: Basic MLOps concepts — model versioning, a simple CI pipeline (GitHub Actions) to run tests + retraining checks
- Time commitment: 4-5 hours/week
- Done looks like: A single command retrains, re-evaluates, and re-runs the fairness audit, with results logged somewhere reviewable

### Milestone 3 (Nov-Dec 2026): Threshold Tuning & Monitoring
- What I'll add: Configurable decision thresholds (not hardcoded 0.5), and basic monitoring for prediction drift over time
- Tools I'll learn: Evidently AI or a similar lightweight monitoring tool, precision-recall curve-based threshold selection
- Time commitment: 3-4 hours/week
- Done looks like: The app supports adjustable risk thresholds, and there's a basic dashboard showing whether the model's predictions are drifting from training-time behavior

## 3rd Year Internship Plan (Jun-Jul 2027)

This project sets me up well for a 3rd year Applied ML internship problem statement — likely one involving productionizing an existing model (retraining pipelines, monitoring, feature stores) rather than building a model from scratch, since that's exactly the gap between where LoanLens is now and where I want it to be by then.

## What I'll need from the placement / mentor ecosystem

Access to a mentor with real MLOps/production ML experience (my 2nd year mentor was strong on the ML fundamentals side; I'll want someone who's shipped ML systems that run continuously, not just once). A community or study group for CI/CD and monitoring tools, since these weren't covered this year. Possibly a free-tier cloud credit program, since monitoring/retraining infrastructure will need more compute than a single Streamlit deploy.

## Risks & open questions

I don't yet know how much of "proper MLOps" is realistic to build solo versus what genuinely requires a team or paid infrastructure — that's something I want to get a mentor's read on early in 3rd year rather than discovering the hard way. I'm also unsure whether to deepen this specific project further or diversify into a second, different problem type (e.g. NLP or vision) for portfolio breadth — I'll use the Week 1 mentor 1:1 in 3rd year to make that call.
