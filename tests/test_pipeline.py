import joblib
import numpy as np
import pandas as pd


def test_preprocessor_no_nulls_after_transform():
    """Confirm the fitted preprocessor produces no NaNs on training data."""
    preprocessor = joblib.load("app/preprocessor.pkl")
    df = pd.read_csv("data/application_train_features.csv").head(100)
    X = df.drop(columns="TARGET")
    X_processed = preprocessor.transform(X)
    assert not np.isnan(X_processed).any(), "Pipeline output contains NaNs"


def test_model_loads_and_predicts():
    """Confirm the saved LightGBM model loads and produces valid probabilities."""
    model = joblib.load("app/lightgbm_model.pkl")
    preprocessor = joblib.load("app/preprocessor.pkl")
    df = pd.read_csv("data/application_train_features.csv").head(5)
    X = df.drop(columns="TARGET")
    X_processed = preprocessor.transform(X)
    probs = model.predict_proba(X_processed)[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all(), "Predicted probabilities out of range"