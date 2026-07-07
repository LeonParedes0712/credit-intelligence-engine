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
# Helper functions (business logic)
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH)


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
# Visual tokens
# -----------------------------

DECISION_STYLE = {
    "Approve": {
        "bg": "#EEF8F1",
        "fg": "#166534",
        "border": "#15803D",
        "icon": "✓",
    },
    "Manual Review": {
        "bg": "#FFF7EB",
        "fg": "#A16207",
        "border": "#B7791F",
        "icon": "⏳",
    },
    "Reject": {
        "bg": "#FEF1F1",
        "fg": "#991B1B",
        "border": "#B91C1C",
        "icon": "✕",
    },
}

RISK_TIER_STYLE = {
    "Very Low Risk": "#15803D",
    "Low Risk": "#4D9A6A",
    "Medium Risk": "#B7791F",
    "High Risk": "#C2703A",
    "Very High Risk": "#B91C1C",
}


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: #F6F8FB;
        }

        [data-testid="stAppViewContainer"] {
            background: #F6F8FB;
        }

        [data-testid="stHeader"] {
            background: #F6F8FB;
        }

        .main .block-container {
            max-width: 1220px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        .cie-header {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            padding: 2rem 2.25rem 1.25rem 2.25rem;
            box-shadow: 0 2px 12px rgba(16,32,51,0.05);
            margin-bottom: 1rem;
        }

        .cie-header h1 {
            font-family: 'Space Grotesk', sans-serif;
            color: #102033;
            font-size: 2.1rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }

        .cie-header p.subtitle {
            color: #1D4ED8;
            font-size: 1rem;
            font-weight: 600;
            margin: 0 0 0.7rem 0;
        }

        .cie-header p.desc {
            color: #64748B;
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0;
            max-width: 780px;
        }

        .cie-helper {
            margin-top: 0.9rem;
            color: #64748B;
            font-size: 0.9rem;
        }

        .cie-section-label {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748B;
            margin: 1.5rem 0 0.7rem 0;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.45rem;
        }

        .cie-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 1px 3px rgba(16,32,51,0.04);
            height: 100%;
        }

        .cie-metric-label {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
            margin-bottom: 0.35rem;
        }

        .cie-metric-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.5rem;
            font-weight: 600;
            color: #102033;
            line-height: 1.2;
            white-space: nowrap;
        }

        .cie-pill {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .cie-banner {
            border-radius: 18px;
            padding: 1.4rem 1.5rem;
            border: 1.5px solid;
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1.2rem;
        }

        .cie-banner-icon {
            width: 3rem;
            height: 3rem;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.8rem;
            font-weight: 700;
            flex-shrink: 0;
        }

        .cie-banner-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0;
        }

        .cie-banner-sub {
            color: #64748B;
            font-size: 0.88rem;
            margin-top: 0.15rem;
        }

        .cie-score-track {
            position: relative;
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(
                90deg,
                #B91C1C 0%,
                #C2703A 25%,
                #B7791F 50%,
                #4D9A6A 75%,
                #15803D 100%
            );
            margin: 0.9rem 0 0.35rem 0;
        }

        .cie-score-marker {
            position: absolute;
            top: -6px;
            width: 3px;
            height: 22px;
            border-radius: 2px;
            background: #102033;
        }

        .cie-score-scale {
            display: flex;
            justify-content: space-between;
            color: #64748B;
            font-size: 0.7rem;
            font-family: 'IBM Plex Mono', monospace;
        }

        .cie-reason {
            display: inline-flex;
            align-items: center;
            width: 100%;
            padding: 0.7rem 0.85rem;
            border-radius: 12px;
            margin-bottom: 0.55rem;
            border-left: 3px solid;
            font-size: 0.86rem;
            color: #102033;
        }

        div.stButton > button {
            background: #2563EB;
            color: #FFFFFF;
            border: none;
            border-radius: 12px;
            padding: 0.65rem 1.35rem;
            font-weight: 600;
            font-size: 0.95rem;
            box-shadow: 0 1px 2px rgba(37,99,235,0.2);
        }

        div.stButton > button:hover {
            background: #1D4ED8;
            color: #FFFFFF;
        }

        /* ----------------------------- */
        /* Sidebar general */
        /* ----------------------------- */

        section[data-testid="stSidebar"] {
            background: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #102033 !important;
        }

        .cie-sidebar-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #102033 !important;
            margin-bottom: 0.2rem;
        }

        .cie-sidebar-caption {
            font-size: 0.79rem;
            color: #64748B !important;
            margin-bottom: 1rem;
        }

        .cie-sidebar-section {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #2563EB !important;
            margin: 1.1rem 0 0.35rem 0;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.35rem;
        }

        /* ----------------------------- */
        /* FIX: number_input black bars */
        /* ----------------------------- */

        section[data-testid="stSidebar"] div[data-baseweb="input"] {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
            background: #FFFFFF !important;
            border-radius: 12px !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
            background: #FFFFFF !important;
            border: none !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] input {
            background-color: #FFFFFF !important;
            color: #102033 !important;
            -webkit-text-fill-color: #102033 !important;
            caret-color: #102033 !important;
            font-weight: 500 !important;
        }

        section[data-testid="stSidebar"] input::placeholder {
            color: #94A3B8 !important;
            -webkit-text-fill-color: #94A3B8 !important;
        }

        section[data-testid="stSidebar"] button {
            background-color: #FFFFFF !important;
            color: #475569 !important;
            border: none !important;
            box-shadow: none !important;
        }

        section[data-testid="stSidebar"] button svg {
            fill: #475569 !important;
            color: #475569 !important;
        }

        /* ----------------------------- */
        /* FIX: slider contrast */
        /* ----------------------------- */

        section[data-testid="stSidebar"] div[data-baseweb="slider"] {
            color: #102033 !important;
        }

        section[data-testid="stSidebar"] div[role="slider"] {
            background-color: #2563EB !important;
            border-color: #2563EB !important;
        }

        /* ----------------------------- */
        /* Streamlit top decoration */
        /* ----------------------------- */

        div[data-testid="stDecoration"] {
            background: #F6F8FB !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_header():
    st.markdown(
        """
        <div class="cie-header">
            <h1>💳 Credit Intelligence Engine</h1>
            <p class="subtitle">Machine learning credit risk scoring, business rules, and expected loss simulation</p>
            <p class="desc">
                This engine scores incoming applications with a trained XGBoost model, converts the
                predicted default probability into a 300–850 credit score, applies underwriting
                business rules, and simulates expected interest, loss, and profit for the requested credit line.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_helper_message():
    st.markdown(
        """
        <div class="cie-helper">
            Adjust the applicant inputs in the sidebar and click <strong>Evaluate Application</strong> to run the engine.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, help_text=None):
    help_html = (
        f"<div style='font-size:0.72rem;color:#94A3B8;margin-top:0.3rem;'>{help_text}</div>"
        if help_text else ""
    )
    st.markdown(
        f"""
        <div class="cie-card">
            <div class="cie-metric-label">{label}</div>
            <div class="cie-metric-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(final_decision, probability, score):
    style = DECISION_STYLE.get(final_decision, DECISION_STYLE["Manual Review"])
    st.markdown(
        f"""
        <div class="cie-banner" style="background:{style['bg']}; border-color:{style['border']};">
            <div class="cie-banner-icon" style="background:{style['border']};">{style['icon']}</div>
            <div>
                <div class="cie-banner-title" style="color:{style['fg']};">{final_decision}</div>
                <div class="cie-banner-sub">Default probability {probability:.2%} · Credit score {score}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_gauge(score, risk_tier):
    min_s, max_s = 300, 850
    pct = max(0.0, min(1.0, (score - min_s) / (max_s - min_s))) * 100
    tier_color = RISK_TIER_STYLE.get(risk_tier, "#64748B")

    st.markdown(
        f"""
        <div class="cie-card">
            <div class="cie-metric-label">Credit Score · Risk Tier</div>
            <div style="display:flex; align-items:center; gap:0.65rem; flex-wrap:wrap;">
                <div class="cie-metric-value">{score}</div>
                <span class="cie-pill" style="background:{tier_color}1F; color:{tier_color};">{risk_tier}</span>
            </div>
            <div class="cie-score-track">
                <div class="cie-score-marker" style="left:calc({pct}% - 2px);"></div>
            </div>
            <div class="cie-score-scale">
                <span>300</span><span>550</span><span>650</span><span>750</span><span>850</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reason_badges(reasons, final_decision):
    style = DECISION_STYLE.get(final_decision, DECISION_STYLE["Manual Review"])
    for reason in reasons:
        st.markdown(
            f"""
            <div class="cie-reason" style="background:{style['bg']}; border-color:{style['border']};">
                {reason}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Credit Intelligence Engine",
    page_icon="💳",
    layout="wide",
)

inject_css()
render_header()


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

for col in feature_df.select_dtypes(include=["object", "string"]).columns:
    sample_applicant[col] = feature_df[col].mode(dropna=True)[0]


# -----------------------------
# Sidebar inputs
# -----------------------------

st.sidebar.markdown('<div class="cie-sidebar-title">Applicant Inputs</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="cie-sidebar-caption">Adjust the fields below to simulate an applicant profile.</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="cie-sidebar-section">Applicant Financials</div>', unsafe_allow_html=True)

income = st.sidebar.number_input(
    "Annual income",
    min_value=0.0,
    value=float(sample_applicant["AMT_INCOME_TOTAL"]),
    step=10000.0,
    help="Total annual income declared by the applicant.",
)

credit_amount = st.sidebar.number_input(
    "Requested credit amount",
    min_value=0.0,
    value=float(sample_applicant["AMT_CREDIT"]),
    step=10000.0,
    help="Total credit amount requested for the loan.",
)

annuity = st.sidebar.number_input(
    "Loan annuity",
    min_value=0.0,
    value=float(sample_applicant["AMT_ANNUITY"]),
    step=1000.0,
    help="Periodic loan payment amount.",
)

goods_price = st.sidebar.number_input(
    "Goods price",
    min_value=0.0,
    value=float(sample_applicant["AMT_GOODS_PRICE"]),
    step=10000.0,
    help="Price of the goods being financed, if applicable.",
)

st.sidebar.markdown('<div class="cie-sidebar-section">Employment Profile</div>', unsafe_allow_html=True)

days_birth = st.sidebar.number_input(
    "Age in years",
    min_value=18,
    max_value=100,
    value=35,
)

years_employed = st.sidebar.number_input(
    "Years employed",
    min_value=0.0,
    max_value=60.0,
    value=5.0,
    step=0.5,
    help="Applicants with under 1 year of employment trigger a manual review rule.",
)

st.sidebar.markdown('<div class="cie-sidebar-section">External Risk Signals</div>', unsafe_allow_html=True)

ext_source_2 = st.sidebar.slider(
    "External source score 2",
    min_value=0.0,
    max_value=1.0,
    value=float(0.5 if pd.isnull(sample_applicant["EXT_SOURCE_2"]) else sample_applicant["EXT_SOURCE_2"]),
    step=0.01,
    help="External bureau risk score. Below 0.20 triggers a manual review rule.",
)

st.sidebar.markdown('<div class="cie-sidebar-section">Financial Assumptions</div>', unsafe_allow_html=True)

interest_rate = st.sidebar.slider(
    "Interest rate assumption",
    min_value=0.01,
    max_value=0.30,
    value=0.12,
    step=0.01,
    help="Annual interest rate used to estimate expected interest income.",
)

lgd = st.sidebar.slider(
    "LGD assumption",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="Loss Given Default.",
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
# Controls
# -----------------------------

control_col1, control_col2 = st.columns([1, 5])
with control_col1:
    evaluate = st.button("Evaluate Application", type="primary")
with control_col2:
    if not evaluate:
        render_helper_message()


# -----------------------------
# Prediction & dashboard
# -----------------------------

if evaluate:
    probability = model.predict_proba(applicant_df)[:, 1][0]
    score = probability_to_score(probability)
    risk_tier = assign_risk_tier(score)

    base_decision = threshold_decision(probability)
    final_decision, reasons = apply_business_rules(
        applicant=applicant,
        base_decision=base_decision,
        probability=probability,
        score=score,
    )

    expected_loss = calculate_expected_loss(
        probability=probability,
        credit_amount=credit_amount,
        lgd=lgd,
    )

    expected_interest, expected_profit = calculate_expected_profit(
        credit_amount=credit_amount,
        expected_loss=expected_loss,
        interest_rate=interest_rate,
    )

    st.markdown('<div class="cie-section-label">Credit Decision Summary</div>', unsafe_allow_html=True)
    render_decision_banner(final_decision, probability, score)

    top_left, top_right = st.columns([1.45, 1.0], gap="large")
    with top_left:
        render_score_gauge(score, risk_tier)
    with top_right:
        dec_a, dec_b = st.columns(2, gap="medium")
        with dec_a:
            render_metric_card("Base Decision", base_decision)
        with dec_b:
            render_metric_card("Final Decision", final_decision)

    st.markdown('<div class="cie-section-label">Financial Risk Simulation</div>', unsafe_allow_html=True)

    profit_color = "#15803D" if expected_profit >= 0 else "#B91C1C"

    fin1, fin2, fin3 = st.columns(3, gap="medium")
    with fin1:
        render_metric_card("Expected Interest", f"${expected_interest:,.0f}")
    with fin2:
        render_metric_card("Expected Loss", f"${expected_loss:,.0f}")
    with fin3:
        st.markdown(
            f"""
            <div class="cie-card">
                <div class="cie-metric-label">Expected Profit</div>
                <div class="cie-metric-value" style="color:{profit_color};">${expected_profit:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="cie-section-label">Decision Logic</div>', unsafe_allow_html=True)

    logic_col, reason_col = st.columns([1, 2], gap="large")
    with logic_col:
        render_metric_card("Base Model Decision", base_decision)
        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
        render_metric_card("Final Decision (post rules)", final_decision)
    with reason_col:
        st.markdown(
            "<div class='cie-metric-label' style='margin-bottom:0.55rem;'>Business Rule Reasons</div>",
            unsafe_allow_html=True,
        )
        render_reason_badges(reasons, final_decision)

    st.markdown('<div class="cie-section-label">Applicant Data Used by the Engine</div>', unsafe_allow_html=True)

    display_cols = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_2",
    ]

    with st.expander("View feature values passed to the model", expanded=False):
        st.dataframe(applicant_df[display_cols], use_container_width=True)