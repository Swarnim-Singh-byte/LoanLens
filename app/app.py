import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="LoanLens", page_icon="💳", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("lightgbm_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    explainer = joblib.load("shap_explainer.pkl")
    with open("defaults.json") as f:
        defaults = json.load(f)
    return model, preprocessor, explainer, defaults

model, preprocessor, explainer, defaults = load_artifacts()

st.title("💳 LoanLens")
st.caption("Loan default risk predictor with explainable AI")

st.markdown("### Applicant Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income (₹)", min_value=0, value=200000, step=10000)
    credit = st.number_input("Loan Credit Amount (₹)", min_value=0, value=500000, step=10000)
    annuity = st.number_input("Annuity / Yearly Payment (₹)", min_value=0, value=25000, step=1000)
    age = st.slider("Age", 18, 75, 35)

with col2:
    employed_years = st.slider("Years Employed", 0, 45, 5)
    gender = st.selectbox("Gender", ["M", "F"])
    ext_source_1 = st.slider("External Score 1 (credit bureau)", 0.0, 1.0, 0.5)
    ext_source_2 = st.slider("External Score 2 (credit bureau)", 0.0, 1.0, 0.5)
    ext_source_3 = st.slider("External Score 3 (credit bureau)", 0.0, 1.0, 0.5)

if st.button("Predict Risk", type="primary"):
    row = defaults.copy()

    row["AMT_INCOME_TOTAL"] = income
    row["AMT_CREDIT"] = credit
    row["AMT_ANNUITY"] = annuity
    row["DAYS_BIRTH"] = -age * 365
    row["DAYS_EMPLOYED"] = -employed_years * 365
    row["CODE_GENDER"] = gender
    row["EXT_SOURCE_1"] = ext_source_1
    row["EXT_SOURCE_2"] = ext_source_2
    row["EXT_SOURCE_3"] = ext_source_3

    row["AGE_YEARS"] = age
    row["EMPLOYMENT_YEARS"] = employed_years
    row["CREDIT_INCOME_RATIO"] = credit / income if income > 0 else 0
    row["ANNUITY_INCOME_RATIO"] = annuity / income if income > 0 else 0

    input_df = pd.DataFrame([row])

    processed = preprocessor.transform(input_df)
    prob = model.predict_proba(processed)[:, 1][0]

    st.markdown("---")
    st.markdown("### Result")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Default Risk", f"{prob*100:.1f}%")
    with col_b:
        if prob > 0.5:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")

    st.markdown("### Why this prediction?")
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed, columns=feature_names)

    shap_values = explainer.shap_values(processed_df)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    expected = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=sv,
            base_values=expected,
            data=processed_df.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)

    st.caption(
        "This chart shows which factors pushed this applicant's risk score "
        "up (red) or down (blue) from the average prediction."
    )
