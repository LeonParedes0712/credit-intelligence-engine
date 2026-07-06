from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "application_train.csv"
MODEL_PATH = BASE_DIR / "outputs" / "xgboost_model.pkl"


# -----------------------------
# Helper functions
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    return df


def probability_to_score(probability, min_score=300, max_score=850):
    score = max_score - probability * (max_score - min_score)
    return int(round(score))


def assign_risk_tier(score):
    if score >= 750:
        return "Very Low Risk"
    elif score >= 700:
        return "Low Risk"
    elif score >= 650:
        return "Medium Risk"
    elif score >= 600:
        return "High Risk"
    else:
        return "Very High Risk"


def threshold_decision(probability):
    if probability <= 0.30:
        return "Approve"
    elif probability >= 0.50:
        return "Reject"
    else:
        return "Manual Review"


def apply_business_rules(applicant, base_decision, probability, score):
    decision = base_decision
    reasons = []

    income = applicant.get("AMT_INCOME_TOTAL", np.nan)
    credit = applicant.get("AMT_CREDIT", np.nan)
    annuity = applicant.get("AMT_ANNUITY", np.nan)
    ext_source_2 = applicant.get("EXT_SOURCE_2", np.nan)
    days_employed = applicant.get("DAYS_EMPLOYED", np.nan)

    if score < 550:
        decision = "Reject"
        reasons.append("Credit score below 550")

    if probability >= 0.60:
        decision = "Reject"
        reasons.append("Default probability above 60%")

    if pd.notnull(income) and income > 0:
        credit_to_income_ratio = credit / income
        annuity_to_income_ratio = annuity / income

        if credit_to_income_ratio > 8:
            if decision == "Approve":
                decision = "Manual Review"
            reasons.append("Credit amount is high relative to income")

        if annuity_to_income_ratio > 0.40:
            if decision == "Approve":
                decision = "Manual Review"
            reasons.append("Annuity burden is high relative to income")

    if pd.notnull(ext_source_2) and ext_source_2 < 0.20:
        if decision == "Approve":
            decision = "Manual Review"
        reasons.append("Low external risk score")

    if pd.notnull(days_employed):
        years_employed = abs(days_employed) / 365

        if years_employed < 1:
            if decision == "Approve":
                decision = "Manual Review"
            reasons.append("Employment history below 1 year")

    if len(reasons) == 0:
        reasons.append("No business rule triggered")

    return decision, reasons


def calculate_expected_loss(probability, credit_amount, lgd=0.45):
    return probability * lgd * credit_amount


def calculate_expected_profit(credit_amount, expected_loss, interest_rate=0.12):
    expected_interest = credit_amount * interest_rate
    expected_profit = expected_interest - expected_loss
    return expected_interest, expected_profit


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Credit Intelligence Engine",
    page_icon="💳",
    layout="wide"
)

st.title("Credit Intelligence Engine")
st.write(
    "Interactive credit risk scoring and decision engine using XGBoost, "
    "credit score conversion, business rules, and expected loss simulation."
)


# -----------------------------
# Load data and model
# -----------------------------

try:
    model = load_model()
    df = load_reference_data()
except FileNotFoundError as error:
    st.error(f"Missing required file: {error}")
    st.stop()


feature_df = df.drop(columns=["TARGET", "SK_ID_CURR"])
sample_applicant = feature_df.median(numeric_only=True)

# Fill categorical columns with most frequent value
for col in feature_df.select_dtypes(include=["object", "string"]).columns:
    sample_applicant[col] = feature_df[col].mode(dropna=True)[0]


# -----------------------------
# Sidebar inputs
# -----------------------------

st.sidebar.header("Applicant Inputs")

income = st.sidebar.number_input(
    "Annual income",
    min_value=0.0,
    value=float(sample_applicant["AMT_INCOME_TOTAL"]),
    step=10000.0
)

credit_amount = st.sidebar.number_input(
    "Requested credit amount",
    min_value=0.0,
    value=float(sample_applicant["AMT_CREDIT"]),
    step=10000.0
)

annuity = st.sidebar.number_input(
    "Loan annuity",
    min_value=0.0,
    value=float(sample_applicant["AMT_ANNUITY"]),
    step=1000.0
)

goods_price = st.sidebar.number_input(
    "Goods price",
    min_value=0.0,
    value=float(sample_applicant["AMT_GOODS_PRICE"]),
    step=10000.0
)

days_birth = st.sidebar.number_input(
    "Age in years",
    min_value=18,
    max_value=100,
    value=35
)

years_employed = st.sidebar.number_input(
    "Years employed",
    min_value=0.0,
    max_value=60.0,
    value=5.0,
    step=0.5
)

ext_source_2 = st.sidebar.slider(
    "External source score 2",
    min_value=0.0,
    max_value=1.0,
    value=float(
        0.5 if pd.isnull(sample_applicant["EXT_SOURCE_2"]) else sample_applicant["EXT_SOURCE_2"]
    ),
    step=0.01
)

interest_rate = st.sidebar.slider(
    "Interest rate assumption",
    min_value=0.01,
    max_value=0.30,
    value=0.12,
    step=0.01
)

lgd = st.sidebar.slider(
    "LGD assumption",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)


# -----------------------------
# Build applicant row
# -----------------------------

applicant = sample_applicant.copy()

applicant["AMT_INCOME_TOTAL"] = income
applicant["AMT_CREDIT"] = credit_amount
applicant["AMT_ANNUITY"] = annuity
applicant["AMT_GOODS_PRICE"] = goods_price
applicant["DAYS_BIRTH"] = -days_birth * 365
applicant["DAYS_EMPLOYED"] = -years_employed * 365
applicant["EXT_SOURCE_2"] = ext_source_2

applicant_df = pd.DataFrame([applicant])
applicant_df = applicant_df[feature_df.columns]


# -----------------------------
# Prediction
# -----------------------------

if st.button("Evaluate Application", type="primary"):
    probability = model.predict_proba(applicant_df)[:, 1][0]
    score = probability_to_score(probability)
    risk_tier = assign_risk_tier(score)

    base_decision = threshold_decision(probability)
    final_decision, reasons = apply_business_rules(
        applicant=applicant,
        base_decision=base_decision,
        probability=probability,
        score=score
    )

    expected_loss = calculate_expected_loss(
        probability=probability,
        credit_amount=credit_amount,
        lgd=lgd
    )

    expected_interest, expected_profit = calculate_expected_profit(
        credit_amount=credit_amount,
        expected_loss=expected_loss,
        interest_rate=interest_rate
    )

    st.subheader("Credit Decision Summary")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.metric("Default Probability", f"{probability:.2%}")
    col2.metric("Credit Score", score)
    col3.metric("Risk Tier", risk_tier)
    col4.metric("Final Decision", final_decision)

    st.write(f"### Decision: {final_decision}")
    st.write(f"### Risk Tier: {risk_tier}")

    st.subheader("Financial Risk Simulation")

    col5, col6, col7 = st.columns(3)

    col5.metric("Expected Interest", f"${expected_interest:,.0f}")
    col6.metric("Expected Loss", f"${expected_loss:,.0f}")
    col7.metric("Expected Profit", f"${expected_profit:,.0f}")

    st.subheader("Decision Logic")

    st.write(f"**Base model decision:** {base_decision}")
    st.write(f"**Final decision after business rules:** {final_decision}")

    st.write("**Business rule reasons:**")
    for reason in reasons:
        st.write(f"- {reason}")

    st.subheader("Applicant Data Used by the Engine")

    display_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_2"
    ]

    st.dataframe(applicant_df[display_cols])

else:
    st.info("Adjust the applicant inputs in the sidebar and click Evaluate Application.")