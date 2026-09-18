from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
from arch import arch_model
from scipy.stats import chi2, genpareto
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Metropol Analytics | Risk Intelligence Prototype",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#102A43"
BLUE = "#2F80ED"
GOLD = "#C99A3E"
BG = "#F5F7FA"
TEXT = "#1A202C"
GREEN = "#1B8A5A"
RED = "#C0392B"
MUTED = "#64748B"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_kenya_credit_data.csv"

st.markdown(
    f"""
    <style>
        .stApp {{ background: {BG}; color: {TEXT}; }}
        [data-testid="stSidebar"] {{ background: {NAVY}; }}
        [data-testid="stSidebar"] * {{ color: #FFFFFF; }}
        .block-container {{ max-width: 1450px; padding-top: 1.4rem; padding-bottom: 2rem; }}
        .app-header {{
            background: linear-gradient(125deg, {NAVY} 0%, #173F6D 72%, #205493 100%);
            border-radius: 16px; color: #FFFFFF; margin-bottom: .9rem; padding: 1.45rem 1.8rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, .12);
        }}
        .app-header h1 {{ font-size: 1.85rem; line-height: 1.15; margin: 0; }}
        .app-header p {{ color: #D9E7F5; margin: .45rem 0 0; font-size: .98rem; }}
        .metric-card {{
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 13px;
            box-shadow: 0 3px 11px rgba(15, 23, 42, .055); min-height: 122px; padding: 1rem 1.1rem;
        }}
        .metric-label {{ color: {MUTED}; font-size: .74rem; font-weight: 700; letter-spacing: .055em; text-transform: uppercase; }}
        .metric-value {{ color: {NAVY}; font-size: 1.68rem; font-weight: 750; line-height: 1.25; margin-top: .38rem; }}
        .metric-note {{ color: {MUTED}; font-size: .78rem; margin-top: .25rem; }}
        .prototype-notice {{
            background: #FFF7E6; border-left: 4px solid {GOLD}; border-radius: 7px; color: #6B4D12;
            margin: .75rem 0 1.3rem; padding: .78rem 1rem;
        }}
        .section-card {{
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 13px;
            margin: .55rem 0; padding: 1rem 1.15rem;
        }}
        .section-card h3 {{ color: {NAVY}; font-size: 1.02rem; margin: 0 0 .35rem; }}
        .section-card p {{ color: {MUTED}; font-size: .9rem; margin: 0; }}
        .sidebar-brand {{ font-size: 1.12rem; font-weight: 750; margin-bottom: .2rem; }}
        .sidebar-subtitle {{ color: #C8D9EC !important; font-size: .77rem; margin-bottom: 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="prototype-notice">
            <strong>Prototype environment.</strong> All records and outputs use synthetic demonstration data.
            This application is not a production credit-decisioning, consumer-scoring, or fraud-adjudication system.
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def make_synthetic_credit_data(n_rows: int = 3500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 69, n_rows)
    income = np.clip(rng.lognormal(np.log(72000), 0.56, n_rows), 12000, 600000)
    loan_amount = np.clip(income * rng.uniform(0.25, 2.8, n_rows), 5000, 1500000)
    loan_term = rng.choice([6, 12, 18, 24, 36, 48, 60], n_rows, p=[.06, .21, .10, .30, .15, .10, .08])
    existing_loans = rng.poisson(1.25, n_rows)
    late_payments = np.clip(rng.poisson(0.8 + 0.34 * existing_loans, n_rows), 0, 12)
    inquiries = np.clip(rng.poisson(1.7, n_rows), 0, 12)
    credit_history_months = np.clip((age - 18) * 12 + rng.normal(0, 35, n_rows), 6, 600)
    debt_to_income = np.clip(0.10 + 0.10 * existing_loans + 0.027 * late_payments + rng.normal(0.12, 0.085, n_rows), 0.03, 0.94)
    mobile_money_months = rng.integers(0, 120, n_rows)
    utility_on_time_pct = np.clip(rng.beta(8, 2.0, n_rows), 0.15, 1.0)
    employment_months = np.clip(rng.lognormal(np.log(38), 0.62, n_rows), 1, 360)
    sector = rng.choice(
        ["Salaried", "SME", "Agriculture", "Transport", "Trade", "Professional"],
        n_rows,
        p=[.31, .22, .14, .11, .15, .07],
    )
    sector_risk = pd.Series(sector).map(
        {
            "Salaried": -0.18,
            "SME": 0.12,
            "Agriculture": 0.22,
            "Transport": 0.17,
            "Trade": 0.08,
            "Professional": -0.13,
        }
    ).to_numpy()
    logit = (
        -3.10
        + 3.15 * debt_to_income
        + 0.25 * late_payments
        + 0.09 * inquiries
        + 0.18 * existing_loans
        - 0.0030 * credit_history_months
        - 0.0060 * employment_months
        - 0.95 * utility_on_time_pct
        - 0.0022 * mobile_money_months
        + sector_risk
        + rng.normal(0, 0.60, n_rows)
    )
    probability_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability_default)
    return pd.DataFrame(
        {
            "borrower_id": [f"BRW-{i:05d}" for i in range(1, n_rows + 1)],
            "age": age,
            "monthly_income_kes": income.round(0),
            "loan_amount_kes": loan_amount.round(0),
            "loan_term_months": loan_term,
            "existing_loans": existing_loans,
            "late_payments_24m": late_payments,
            "credit_inquiries_12m": inquiries,
            "credit_history_months": credit_history_months.round(0),
            "debt_to_income": debt_to_income,
            "mobile_money_months": mobile_money_months,
            "utility_on_time_pct": utility_on_time_pct,
            "employment_months": employment_months.round(0),
            "sector": sector,
            "default": default,
        }
    )


@st.cache_data(show_spinner=False)
def load_credit_data() -> tuple[pd.DataFrame, str]:
    if DATA_PATH.exists():
        try:
            data = pd.read_csv(DATA_PATH)
            return data, f"Loaded repository dataset: {DATA_PATH.name}"
        except Exception as exc:
            return make_synthetic_credit_data(), f"Repository data could not be read ({exc}); generated fallback data used."
    return make_synthetic_credit_data(), "Repository data file not found; generated fallback data used."


@st.cache_data(show_spinner=False)
def prepare_model_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    required_features = [
        "age",
        "monthly_income_kes",
        "loan_amount_kes",
        "loan_term_months",
        "existing_loans",
        "late_payments_24m",
        "credit_inquiries_12m",
        "credit_history_months",
        "debt_to_income",
        "mobile_money_months",
        "utility_on_time_pct",
        "employment_months",
    ]
    target_candidates = ["default", "target", "loan_default", "default_flag"]
    target = next((name for name in target_candidates if name in data.columns), None)

    # Use repository data only if its columns support the full demo model.
    if target is None or not set(required_features).issubset(data.columns):
        data = make_synthetic_credit_data()
        target = "default"

    x = data[required_features].copy().apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    y = pd.to_numeric(data[target], errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:
        return prepare_model_data(make_synthetic_credit_data())

    return x, y, required_features


@st.cache_resource(show_spinner="Training demonstration models…")
def train_models(data: pd.DataFrame) -> dict[str, Any]:
    x, y, features = prepare_model_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    logistic.fit(x_train_scaled, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    xgb_model.fit(x_train, y_train)

    return {
        "features": features,
        "x_test": x_test.reset_index(drop=True),
        "xgb": xgb_model,
        "xgb_auc": float(roc_auc_score(y_test, xgb_model.predict_proba(x_test)[:, 1])),
        "logistic_auc": float(roc_auc_score(y_test, logistic.predict_proba(x_test_scaled)[:, 1])),
        "probabilities": xgb_model.predict_proba(x_test)[:, 1],
    }


@st.cache_data(show_spinner=False)
def synthetic_fx_returns(n_obs: int = 1250, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    shocks = rng.standard_t(df=6, size=n_obs) * 0.0034
    volatility = np.zeros(n_obs)
    returns = np.zeros(n_obs)
    volatility[0] = 0.003
    for index in range(1, n_obs):
        volatility[index] = np.sqrt(0.000001 + 0.10 * returns[index - 1] ** 2 + 0.86 * volatility[index - 1] ** 2)
        returns[index] = 0.00010 + volatility[index] * shocks[index] / 0.0034
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_obs)
    return pd.Series(returns, index=dates, name="fx_return")


def run_stress_simulation(confidence: float, horizon: int, n_sims: int, seed: int) -> dict[str, float | np.ndarray]:
    returns = synthetic_fx_returns()
    try:
        garch = arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Constant", dist="t").fit(disp="off")
        annualized_vol = float(garch.conditional_volatility.iloc[-1] / 100 * np.sqrt(252))
    except Exception:
        annualized_vol = float(returns.std() * np.sqrt(252))

    losses = -returns.to_numpy()
    threshold = np.quantile(losses, 0.90)
    excess = losses[losses > threshold] - threshold
    try:
        shape, _, scale = genpareto.fit(excess, floc=0)
    except Exception:
        shape, scale = 0.1, float(np.std(excess) if len(excess) else np.std(losses))

    rng = np.random.default_rng(seed)
    daily_vol = max(annualized_vol / np.sqrt(252), 0.0001)
    paths = rng.standard_t(df=6, size=(n_sims, horizon)) * daily_vol
    tail_mask = rng.random((n_sims, horizon)) < 0.04
    tail_draws = threshold + genpareto.rvs(shape, loc=0, scale=max(scale, 0.0001), size=(n_sims, horizon), random_state=rng)
    paths[tail_mask] = -tail_draws[tail_mask]
    cumulative_depreciation = -paths.sum(axis=1)
    var = float(np.quantile(cumulative_depreciation, confidence))
    es = float(cumulative_depreciation[cumulative_depreciation >= var].mean())

    one_day_var = float(-np.quantile(returns, 1 - confidence))
    breaches = int((losses > one_day_var).sum())
    expected_breaches = len(losses) * (1 - confidence)
    observed_rate = max(min(breaches / len(losses), 1 - 1e-12), 1e-12)
    likelihood_null = max((1 - confidence) ** breaches * confidence ** (len(losses) - breaches), 1e-300)
    likelihood_alt = max(observed_rate ** breaches * (1 - observed_rate) ** (len(losses) - breaches), 1e-300)
    kupiec_statistic = -2 * np.log(likelihood_null / likelihood_alt)

    return {
        "simulations": cumulative_depreciation,
        "var": var,
        "es": es,
        "annualized_vol": annualized_vol,
        "kupiec_p": float(1 - chi2.cdf(kupiec_statistic, 1)),
        "breaches": breaches,
        "expected_breaches": expected_breaches,
    }


def executive_overview(model_results: dict[str, Any]) -> None:
    render_header("Metropol Analytics", "Integrated credit-risk intelligence, portfolio resilience, and data-governance prototype.")
    priority_cases = int((model_results["probabilities"] >= 0.25).sum())
    cols = st.columns(4)
    values = [
        ("Portfolio stress signal", "Elevated", "FX scenario sensitivity monitored"),
        ("Priority collection cases", f"{priority_cases:,}", "Illustrative ML triage output"),
        ("Data-quality exceptions", "8.5%", "Rule and anomaly-check estimate"),
        ("Identity-risk clusters", "6", "Synthetic shared-identifier networks"),
    ]
    for col, value in zip(cols, values):
        with col:
            metric_card(*value)

    st.markdown("## Decision intelligence across the credit lifecycle")
    st.write("This interactive prototype demonstrates portfolio stress testing, collection prioritization, data quality controls, policy retrieval, alternative-data scenarios, and fraud-network review in one executive-facing environment.")

    modules = [
        ("01", "Portfolio Stress Testing", "GARCH-EVT simulation, FX scenarios, tail risk, and default-rate sensitivity."),
        ("02", "Collections Intelligence", "XGBoost triage, model comparison, case review, and SHAP waterfall explanations."),
        ("03", "Data Quality Controls", "Credit-information-provider validation rules and statistical anomaly detection."),
        ("04", "Crystobol Assistant", "Transparent retrieval of relevant policy and scoring guidance."),
        ("05", "Alternative-Data Simulation", "Illustrative impacts from mobile-money, utility, and rent-payment signals."),
        ("06", "Fraud & Identity Networks", "Graph analytics for shared identifiers and suspicious linked-record clusters."),
    ]
    left, right = st.columns(2)
    for index, (number, title, description) in enumerate(modules):
        with (left if index % 2 == 0 else right):
            st.markdown(f"<div class='section-card'><h3>{number} · {title}</h3><p>{description}</p></div>", unsafe_allow_html=True)


def portfolio_stress_page() -> None:
    render_header("Portfolio Monitoring | GARCH-EVT Stress Simulator", "Simulate illustrative KES/USD depreciation scenarios and translate tail-risk conditions into portfolio stress signals.")
    controls, output = st.columns([1, 2])
    with controls:
        confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95, format_func=lambda value: f"{int(value * 100)}%")
        horizon = st.select_slider("Scenario horizon (business days)", options=[1, 5, 10, 20, 60], value=20)
        n_sims = st.select_slider("Monte Carlo paths", options=[2000, 5000, 10000], value=5000)
        seed = st.number_input("Simulation seed", min_value=1, max_value=99999, value=42)
    result = run_stress_simulation(confidence, horizon, n_sims, int(seed))
    with output:
        cols = st.columns(4)
        values = [
            ("Stress VaR", f"{result['var']:.2%}", f"{int(confidence * 100)}% confidence, {horizon} days"),
            ("Expected shortfall", f"{result['es']:.2%}", "Mean loss beyond VaR"),
            ("GARCH annual volatility", f"{result['annualized_vol']:.2%}", "Latest conditional estimate"),
            ("Kupiec p-value", f"{result['kupiec_p']:.3f}", "Illustrative coverage test"),
        ]
        for col, value in zip(cols, values):
            with col:
                metric_card(*value)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.hist(result["simulations"], bins=55, color=BLUE, alpha=0.78, edgecolor="white")
        ax.axvline(result["var"], color=RED, linewidth=2.2, label=f"VaR: {result['var']:.2%}")
        ax.axvline(result["es"], color=GOLD, linewidth=2.2, label=f"ES: {result['es']:.2%}")
        ax.set_title("Simulated cumulative FX depreciation distribution")
        ax.set_xlabel("Cumulative depreciation")
        ax.set_ylabel("Simulation paths")
        ax.legend()
        ax.grid(alpha=0.18)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def collections_page(model_results: dict[str, Any]) -> None:
    render_header("Collection Prioritization | ML Score Layer", "Prioritize illustrative collection cases using probability-of-default estimates and borrower-level model drivers.")
    x_test = model_results["x_test"].copy()
    x_test["predicted_risk"] = model_results["probabilities"]
    x_test["risk_band"] = pd.cut(x_test["predicted_risk"], bins=[-0.01, 0.10, 0.25, 1.0], labels=["Low", "Medium", "High"])
    threshold = st.slider("Priority threshold", min_value=0.05, max_value=0.75, value=0.25, step=0.01, format="%.0f%%")
    flagged = x_test[x_test["predicted_risk"] >= threshold].sort_values("predicted_risk", ascending=False).reset_index(drop=True)

    cols = st.columns(4)
    values = [
        ("XGBoost test AUC", f"{model_results['xgb_auc']:.3f}", "Synthetic-data holdout set"),
        ("Logistic test AUC", f"{model_results['logistic_auc']:.3f}", "Transparent benchmark model"),
        ("Priority cases", f"{len(flagged):,}", f"Risk threshold ≥ {threshold:.0%}"),
        ("Mean flagged risk", f"{flagged['predicted_risk'].mean():.1%}" if len(flagged) else "—", "Predicted default probability"),
    ]
    for col, value in zip(cols, values):
        with col:
            metric_card(*value)

    st.markdown("## Priority queue")
    if flagged.empty:
        st.info("No records meet the selected priority threshold. Lower the threshold to view illustrative cases.")
        return

    queue = flagged.head(100).copy()
    queue.insert(0, "case_rank", np.arange(1, len(queue) + 1))
    queue["predicted_risk"] = queue["predicted_risk"].map("{:.1%}".format)
    if "debt_to_income" in queue.columns:
        queue["debt_to_income"] = pd.to_numeric(queue["debt_to_income"], errors="coerce").map("{:.1%}".format)
    st.dataframe(queue, use_container_width=True, hide_index=True)
    st.download_button("Download priority queue (CSV)", flagged.to_csv(index=False).encode("utf-8"), "metropol_collection_priority_queue.csv", "text/csv")

    st.markdown("## Case explanation")
    selected_case = st.selectbox(
        "Select priority-queue row",
        options=list(range(len(flagged))),
        format_func=lambda index: f"Case {index + 1} | predicted risk {flagged.iloc[index]['predicted_risk']:.1%}",
    )

    # The selected borrower must stay as a one-row DataFrame for SHAP and XGBoost.
    row = flagged.iloc[[selected_case]][model_results["features"]].copy()

    try:
        st.markdown("### SHAP waterfall explanation")
        background = model_results["x_test"][model_results["features"]].sample(n=min(250, len(model_results["x_test"])), random_state=42)
        explainer = shap.Explainer(model_results["xgb"], background)
        shap_explanation = explainer(row)
        shap.plots.waterfall(shap_explanation[0], max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(11, 6.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        predicted_probability = float(model_results["xgb"].predict_proba(row)[0, 1])
        st.info(f"Selected case model-estimated default probability: **{predicted_probability:.1%}**.")
        st.caption("The waterfall starts at the model baseline and shows how each feature moves the selected borrower toward the final model output. Red contributions increase estimated risk; blue contributions reduce it. The SHAP axis can be in model-margin units rather than direct probability units.")
    except Exception as exc:
        st.warning(f"SHAP waterfall explanation was unavailable: {exc}")
        st.caption("The priority queue remains available. Confirm that SHAP and XGBoost were installed from requirements.txt, then restart Streamlit.")

    with st.expander("Model governance notes"):
        st.markdown("This model is trained only on synthetic observations. Production use requires approved target definitions, temporal validation, calibration, fairness testing, performance monitoring, challenger models, human override policies, and auditable model documentation.")


def data_quality_page() -> None:
    render_header("Validation & Processing | Data Quality Controls", "Combine deterministic validation rules with Isolation Forest anomaly detection for illustrative CIP-submission monitoring.")
    rng = np.random.default_rng(101)
    n = st.slider("Synthetic submissions", 100, 800, 250, 25)
    submissions = pd.DataFrame(
        {
            "submission_id": [f"CIP-{i:05d}" for i in range(1, n + 1)],
            "age": rng.integers(17, 86, n),
            "monthly_income_kes": np.clip(rng.lognormal(np.log(74000), 0.62, n), 0, 1200000),
            "loan_amount_kes": np.clip(rng.lognormal(np.log(120000), 0.85, n), 0, 3000000),
            "days_since_update": np.clip(rng.poisson(32, n), 0, 1200),
            "phone_length": rng.choice([9, 10, 12, 13], n, p=[.04, .78, .12, .06]),
            "duplicate_identifier": rng.random(n) < 0.055,
        }
    )
    injection = rng.choice(submissions.index, max(4, n // 18), replace=False)
    submissions.loc[injection[:len(injection) // 2], "monthly_income_kes"] = rng.choice([0, 4000000, 9000000], len(injection) // 2)
    submissions.loc[injection[len(injection) // 2:], "days_since_update"] = rng.integers(500, 1900, len(injection) - len(injection) // 2)
    submissions["rule_exception"] = ((submissions["age"] < 18) | (submissions["monthly_income_kes"] <= 0) | (submissions["monthly_income_kes"] > 2000000) | (submissions["loan_amount_kes"] <= 0) | (submissions["days_since_update"] > 365) | (~submissions["phone_length"].isin([10, 12])) | submissions["duplicate_identifier"])
    features = ["age", "monthly_income_kes", "loan_amount_kes", "days_since_update", "phone_length"]
    detector = IsolationForest(contamination=0.07, random_state=42)
    submissions["statistical_anomaly"] = detector.fit_predict(submissions[features]) == -1
    submissions["review_status"] = np.where(submissions["rule_exception"] | submissions["statistical_anomaly"], "Review", "Pass")
    exceptions = submissions[submissions["review_status"] == "Review"].copy()

    cols = st.columns(4)
    values = [
        ("Total submissions", f"{len(submissions):,}", "Synthetic CIP records"),
        ("Rule exceptions", f"{int(submissions['rule_exception'].sum()):,}", "Deterministic validation checks"),
        ("Statistical anomalies", f"{int(submissions['statistical_anomaly'].sum()):,}", "Isolation Forest output"),
        ("Records for review", f"{len(exceptions):,}", f"{len(exceptions) / len(submissions):.1%} of submissions"),
    ]
    for col, value in zip(cols, values):
        with col:
            metric_card(*value)
    st.markdown("## Exception queue")
    st.dataframe(exceptions, use_container_width=True, hide_index=True)
    st.download_button("Download exception queue (CSV)", exceptions.to_csv(index=False).encode("utf-8"), "metropol_data_quality_exceptions.csv", "text/csv")


def crystobol_page() -> None:
    render_header("Crystobol Assistant | Policy Explanation Retrieval", "Retrieve the most relevant internal guidance passage for an analyst or consumer-support question.")
    corpus = {
        "Portfolio stress methodology": "Portfolio stress outputs combine conditional volatility estimation, extreme-tail modelling and scenario simulation. They support risk assessment and do not by themselves set credit decisions.",
        "Collections prioritization policy": "Collection queues are decision-support tools. Assigned teams should use model scores together with account history, contact preferences, vulnerability indicators and approved operational policy.",
        "Consumer data correction": "Consumers should be provided with a clear channel to query or dispute inaccurate information. Investigation outcomes, evidence and correction actions must be recorded in accordance with applicable policy.",
        "Alternative data governance": "Alternative data sources require documented purpose, consent or lawful basis where applicable, data-quality assessment, fairness testing, explainability and ongoing monitoring before use in a decision process.",
        "Fraud investigation protocol": "Network alerts identify relationships for investigation, not proof of fraud. Analysts should corroborate alerts through approved evidence, controls and escalation processes before taking any action.",
        "Model risk management": "Models used in material workflows require a defined owner, documented development and validation, periodic performance monitoring, change controls, access restrictions and an escalation process for failures.",
    }
    question = st.text_input("Ask a question", placeholder="Example: How should a fraud-network alert be used?")
    if not question:
        question = "How should a fraud-network alert be used?"
    titles = list(corpus.keys())
    passages = list(corpus.values())
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(passages + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranking = np.argsort(scores)[::-1]
    top = int(ranking[0])
    left, right = st.columns([2, 1])
    with left:
        st.markdown("## Retrieved guidance")
        st.markdown(f"<div class='section-card'><h3>{titles[top]}</h3><p>{passages[top]}</p></div>", unsafe_allow_html=True)
        st.caption("This module performs retrieval only. It does not generate policy advice or connect to a live document-management system.")
    with right:
        st.markdown("## Retrieval confidence")
        for index in ranking[:3]:
            st.progress(float(scores[index]), text=f"{titles[index]} — {scores[index]:.0%}")


def alternative_data_page() -> None:
    render_header("Mobile Facilities | Alternative-Data Simulation", "Illustrate how consented payment-behaviour signals could be evaluated in a governed inclusion or affordability workflow.")
    left, right = st.columns([1, 1.35])
    with left:
        base_score = st.slider("Baseline bureau score", 250, 900, 580, 5)
        mobile_months = st.slider("Mobile-money history (months)", 0, 120, 36)
        utility_rate = st.slider("Utility payments on time", 0, 100, 82, 1)
        rent_rate = st.slider("Rent payments on time", 0, 100, 75, 1)
        verified_income = st.checkbox("Income pattern verified", value=True)
    mobile_adjustment = min(mobile_months / 120 * 24, 24)
    utility_adjustment = (utility_rate - 50) * 0.42
    rent_adjustment = (rent_rate - 50) * 0.32
    verification_adjustment = 18 if verified_income else 0
    adjustment = float(np.clip(mobile_adjustment + utility_adjustment + rent_adjustment + verification_adjustment, -55, 85))
    adjusted_score = int(np.clip(base_score + adjustment, 250, 900))
    with right:
        cols = st.columns(3)
        values = [
            ("Baseline score", str(base_score), "Illustrative bureau score"),
            ("Illustrative adjustment", f"{adjustment:+.0f}", "Alternative-data scenario"),
            ("Simulated score", str(adjusted_score), "Not a validated score"),
        ]
        for col, value in zip(cols, values):
            with col:
                metric_card(*value)
        contribution = pd.DataFrame({"Signal": ["Mobile-money history", "Utility payment behaviour", "Rent payment behaviour", "Income-pattern verification"], "Illustrative points": [mobile_adjustment, utility_adjustment, rent_adjustment, verification_adjustment]})
        fig, ax = plt.subplots(figsize=(8.5, 4.3))
        ax.barh(contribution["Signal"], contribution["Illustrative points"], color=[GREEN if value >= 0 else RED for value in contribution["Illustrative points"]])
        ax.axvline(0, color="#334155", linewidth=.8)
        ax.set_title("Illustrative contribution by signal")
        ax.set_xlabel("Score-point adjustment")
        ax.grid(axis="x", alpha=.18)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def fraud_network_page() -> None:
    render_header("Risk Exposure | Fraud & Identity Networks", "Explore synthetic shared-identifier relationships that could be prioritized for review under an approved investigation process.")
    rng = np.random.default_rng(202)
    graph = nx.Graph()
    identities = [f"ID-{i:03d}" for i in range(1, 45)]
    accounts = [f"ACC-{i:03d}" for i in range(1, 100)]
    for account in accounts:
        graph.add_edge(account, rng.choice(identities), relation="primary_identifier")
    for group_number, group in enumerate([rng.choice(identities, size=size, replace=False).tolist() for size in [4, 5, 6, 4]], start=1):
        hub = f"PHONE-HUB-{group_number}"
        for identity in group:
            graph.add_edge(identity, hub, relation="shared_phone")
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component_map = {node: index for index, component in enumerate(components) for node in component}
    component_sizes = {index: len(component) for index, component in enumerate(components)}
    suspicious_nodes = {node for node, index in component_map.items() if component_sizes[index] >= 8}

    cols = st.columns(4)
    values = [
        ("Synthetic accounts", str(len(accounts)), "Credit facilities / applications"),
        ("Identity records", str(len(identities)), "Synthetic identifiers"),
        ("Linked-record clusters", str(sum(size >= 8 for size in component_sizes.values())), "Threshold-based review clusters"),
        ("Nodes for review", str(len(suspicious_nodes)), "Relationship signal only"),
    ]
    for col, value in zip(cols, values):
        with col:
            metric_card(*value)

    pos = nx.spring_layout(graph, seed=19, k=0.48)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    account_nodes = [node for node in graph.nodes if node.startswith("ACC")]
    identity_nodes = [node for node in graph.nodes if node.startswith("ID")]
    hub_nodes = [node for node in graph.nodes if node.startswith("PHONE")]
    nx.draw_networkx_edges(graph, pos, alpha=.24, edge_color="#94A3B8", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=account_nodes, node_color="#9CC2E5", node_size=75, ax=ax, label="Accounts")
    nx.draw_networkx_nodes(graph, pos, nodelist=identity_nodes, node_color=[RED if node in suspicious_nodes else GOLD for node in identity_nodes], node_size=115, ax=ax, label="Identifiers")
    nx.draw_networkx_nodes(graph, pos, nodelist=hub_nodes, node_color=NAVY, node_size=210, ax=ax, label="Shared phone hubs")
    ax.set_title("Synthetic cross-record relationship network")
    ax.legend(loc="upper left", frameon=False)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def methodology_page() -> None:
    render_header("Methodology & Governance", "A concise record of what is live in the prototype, what is illustrative, and what a production implementation would require.")
    table = pd.DataFrame(
        [
            ["Portfolio Stress Testing", "GARCH(1,1), POT/EVT, Monte Carlo, illustrative Kupiec test", "Synthetic FX series and sector assumptions"],
            ["Collections Intelligence", "XGBoost, Logistic Regression, holdout AUC, SHAP waterfall", "Synthetic borrower observations and labels"],
            ["Data Quality Controls", "Rule checks and Isolation Forest", "Synthetic CIP submissions and thresholds"],
            ["Crystobol Assistant", "TF-IDF retrieval and cosine similarity", "Small demonstration corpus; no LLM generation"],
            ["Alternative-Data Simulation", "Interactive transparent scoring illustration", "Not calibrated, validated, or approved for decisions"],
            ["Fraud & Identity Networks", "NetworkX relationship graph analysis", "Synthetic records and investigation thresholds"],
        ],
        columns=["Module", "Live analytical element", "Illustrative / not production-ready"],
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("## Production readiness checklist")
    st.markdown("- Establish accountable business owners, intended use, risk tier, and approval governance.\n- Define data contracts, lineage, privacy controls, retention, security, and access controls.\n- Validate performance, calibration, robustness, stability, fairness, and operating thresholds on relevant historical data.\n- Document models, assumptions, transformations, limitations, change controls, human review, and escalation procedures.\n- Monitor data quality, drift, model performance, fairness, security, user activity, and incidents.")


def main() -> None:
    data, data_status = load_credit_data()
    model_results = train_models(data)
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>◈ METROPOL ANALYTICS</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-subtitle'>Risk Intelligence Prototype</div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            [
                "Executive Overview",
                "Portfolio Stress Testing",
                "Collections Intelligence",
                "Data Quality Controls",
                "Crystobol Assistant",
                "Alternative-Data Simulation",
                "Fraud & Identity Networks",
                "Methodology & Governance",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Synthetic-data demonstration only")
        st.caption(data_status)

    if page == "Executive Overview":
        executive_overview(model_results)
    elif page == "Portfolio Stress Testing":
        portfolio_stress_page()
    elif page == "Collections Intelligence":
        collections_page(model_results)
    elif page == "Data Quality Controls":
        data_quality_page()
    elif page == "Crystobol Assistant":
        crystobol_page()
    elif page == "Alternative-Data Simulation":
        alternative_data_page()
    elif page == "Fraud & Identity Networks":
        fraud_network_page()
    else:
        methodology_page()


if __name__ == "__main__":
    main()
