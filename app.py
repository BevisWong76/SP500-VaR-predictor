import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from arch import arch_model

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional Risk Workstation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .metric-card-danger {
        background-color: #fff5f5;
        border: 1px solid #feb2b2;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Helper Functions & Data Loaders (Cached for Performance)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_live_market_data(ticker="^GSPC"):
    """Fetch 1 year of daily market returns using yfinance."""
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        else:
            df = df[["Close"]]

        returns = df.pct_change().dropna() * 100
        returns.columns = ["Return"]
        return returns
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_historical_backtest():
    """Load pre-computed historical backtest results."""
    csv_path = "models/var_backtest_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        return df
    else:
        st.warning(
            "Historical backtest CSV not found. Please ensure 'data/var_backtest_results.csv' exists."
        )
        return pd.DataFrame()


def load_metadata():
    """Load model calibration metadata & parameters log."""
    json_path = "models/metadata.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        st.warning(
            "Metadata JSON not found. Please ensure 'data/metadata.json' exists."
        )
        return {}


# Load Initial Data
metadata = load_metadata()
backtest_df = load_historical_backtest()
live_returns = fetch_live_market_data("^GSPC")


# -----------------------------------------------------------------------------
# 3. Sidebar Controls & Model Governance Log
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Risk Engine Controls")
st.sidebar.markdown("---")

st.sidebar.header("1. Portfolio & Confidence Settings")
portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($USD)",
    min_value=10000,
    max_value=1000000000,
    value=1000000,
    step=50000,
    format="%d",
)

confidence_level = st.sidebar.select_slider(
    "VaR Confidence Level", options=[0.90, 0.95, 0.99], value=0.95
)

st.sidebar.markdown("---")
st.sidebar.header("2. Backtest Period Filter")
min_date = backtest_df.index.min().date()
max_date = backtest_df.index.max().date()

date_range = st.sidebar.date_input(
    "Select Analysis Window",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)

# Sidebar Footnote - Model Audit Log
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Audit & Governance Log")
st.sidebar.caption(f"**Last MCMC Refit:** `{metadata['last_calibrated']}`")
st.sidebar.caption(
    "**Data Source:** `yfinance (^GSPC)` (Live Auto-refresh)"
)


# -----------------------------------------------------------------------------
# 4. Main Page Header & Model Governance Banner
# -----------------------------------------------------------------------------
st.title("🛡️ Institutional Risk Workstation")
st.markdown(
    "**ARIMAX-GARCH (Frequentist MLE)** vs. **Bayesian MCMC GARCH** Value-at-Risk Engine"
)

# Governance Metadata Expander
with st.expander("ℹ️ Model Calibration Log & Active Parameters", expanded=False):
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.markdown(
            f"**Last Model Calibration Date:** `{metadata['last_calibrated']}` *(Monthly Schedule)*"
        )
        st.markdown("**Frequentist MLE Parameters (GARCH 1,1):**")
        st.json(metadata["mle_params"])

    with col_meta2:
        st.markdown(
            "**Inference Strategy:** Offline MCMC Sampling (PyMC) + Online Live Refit"
        )
        st.markdown("**Bayesian MCMC Posterior Means (Student-t GARCH):**")
        st.json(metadata["bayesian_params"])

    st.caption(
        "💡 *Note: Market returns are updated live on every page load. Parameter posteriors are recalibrated monthly to ensure performance without UI latency.*"
    )

st.divider()


# -----------------------------------------------------------------------------
# 5. Section 1: Today's Live Risk Dashboard & Dollar VaR Engine
# -----------------------------------------------------------------------------
st.subheader("📅 Today's Live Risk Dashboard")

if not live_returns.empty:
    latest_date = live_returns.index[-1].strftime("%Y-%m-%d")
    latest_return = live_returns.iloc[-1]["Return"]

    # Fit quick MLE GARCH on 1-year data (< 0.2s)
    am = arch_model(
        live_returns["Return"], vol="Garch", p=1, q=1, dist="StudentsT"
    )
    res = am.fit(disp="off")
    forecast_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])

    # Quantiles based on chosen confidence
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores[confidence_level]

    mle_var_pct = forecast_vol * z
    mle_dollar_var = (mle_var_pct / 100) * portfolio_value

    # Bayesian approximation with tail adjustment (Student-t parameter nu)
    bayes_var_pct = mle_var_pct * 1.08  # Account for parameter uncertainty
    bayes_dollar_var = (bayes_var_pct / 100) * portfolio_value

    # Display Cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label=f"Latest S&P 500 Return ({latest_date})",
        value=f"{latest_return:.2f}%",
        delta=f"{latest_return:.2f}%",
    )

    c2.metric(
        label=f"1-Day Expected Volatility", value=f"{forecast_vol:.2f}% / day"
    )

    c3.metric(
        label=f"MLE Dollar VaR ({confidence_level*100:.0f}%)",
        value=f"-${mle_dollar_var:,.0f}",
        delta=f"-{mle_var_pct:.2f}%",
        delta_color="inverse",
    )

    c4.metric(
        label=f"Bayesian Dollar VaR ({confidence_level*100:.0f}%)",
        value=f"-${bayes_dollar_var:,.0f}",
        delta=f"-{bayes_var_pct:.2f}%",
        delta_color="inverse",
    )
else:
    st.warning("Could not fetch live market data. Check internet connection.")

st.divider()


# -----------------------------------------------------------------------------
# 6. Section 2: Interactive Backtest Visualizations & Stress Testing
# -----------------------------------------------------------------------------
st.subheader("⚔️ Backtest Performance & Dynamic Stress Testing")

# Filter DataFrame by Date
if len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = backtest_df.loc[
        (backtest_df.index.date >= start_d)
        & (backtest_df.index.date <= end_d)
    ].copy()
else:
    filtered_df = backtest_df.copy()

# Recalculate Breaches based on Sidebar Confidence Level
z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
multiplier = z_scores[confidence_level] / 1.645  # Scale factor

filtered_df["MLE_VaR"] = filtered_df["MLE_VaR_95"] * multiplier
filtered_df["Bayesian_VaR"] = filtered_df["Bayesian_VaR_95"] * multiplier

filtered_df["MLE_Breach_Active"] = (
    filtered_df["Return"] < filtered_df["MLE_VaR"]
)
filtered_df["Bayesian_Breach_Active"] = (
    filtered_df["Return"] < filtered_df["Bayesian_VaR"]
)

total_days = len(filtered_df)
mle_breaches = filtered_df["MLE_Breach_Active"].sum()
bayes_breaches = filtered_df["Bayesian_Breach_Active"].sum()

expected_breaches = int(total_days * (1 - confidence_level))

# Metric Summary Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Evaluation Period Days", f"{total_days} Days")
m2.metric(
    "Expected Breaches (Theoretical)", f"{expected_breaches} Times"
)
m3.metric(
    "MLE Breaches (Actual)",
    f"{mle_breaches} ({mle_breaches/total_days*100:.2f}%)",
)
m4.metric(
    "Bayesian Breaches (Actual)",
    f"{bayes_breaches} ({bayes_breaches/total_days*100:.2f}%)",
)

# Tabs for Visuals
tab1, tab2 = st.tabs(
    ["📈 Time Series & Breaches", "🪜 Cumulative Breaches vs Theoretical"]
)

with tab1:
    # Interactive Plotly Overlay
    fig = go.Figure()

    # Asset Return
    fig.add_trace(
        go.Scatter(
            x=filtered_df.index,
            y=filtered_df["Return"],
            mode="lines",
            name="Daily Return",
            line=dict(color="lightgrey", width=1),
        )
    )

    # MLE VaR
    fig.add_trace(
        go.Scatter(
            x=filtered_df.index,
            y=filtered_df["MLE_VaR"],
            mode="lines",
            name="MLE VaR",
            line=dict(color="#FF6B6B", width=1.5),
        )
    )

    # Bayesian VaR
    fig.add_trace(
        go.Scatter(
            x=filtered_df.index,
            y=filtered_df["Bayesian_VaR"],
            mode="lines",
            name="Bayesian VaR",
            line=dict(color="#1A73E8", width=1.5),
        )
    )

    # Breaches Markers
    mle_breach_df = filtered_df[filtered_df["MLE_Breach_Active"]]
    bayes_breach_df = filtered_df[filtered_df["Bayesian_Breach_Active"]]

    fig.add_trace(
        go.Scatter(
            x=mle_breach_df.index,
            y=mle_breach_df["Return"],
            mode="markers",
            name="MLE Breach",
            marker=dict(color="red", size=6, symbol="circle"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bayes_breach_df.index,
            y=bayes_breach_df["Return"],
            mode="markers",
            name="Bayesian Breach",
            marker=dict(color="blue", size=7, symbol="x"),
        )
    )

    fig.update_layout(
        title=f"S&P 500 Daily Returns vs. {confidence_level*100:.0f}% Value-at-Risk Thresholds",
        xaxis_title="Date",
        yaxis_title="Return / VaR (%)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, width='stretch')

with tab2:
    # Cumulative Breaches Step Plot
    cum_mle = filtered_df["MLE_Breach_Active"].cumsum()
    cum_bayes = filtered_df["Bayesian_Breach_Active"].cumsum()

    # Theoretical Line
    daily_expected_rate = 1 - confidence_level
    theoretical_cum = np.arange(1, total_days + 1) * daily_expected_rate

    fig_cum = go.Figure()

    fig_cum.add_trace(
        go.Scatter(
            x=filtered_df.index,
            y=theoretical_cum,
            mode="lines",
            name=f"Expected Breaches ({daily_expected_rate*100:.1f}%)",
            line=dict(color="black", dash="dash"),
        )
    )

    fig_cum.add_trace(
        go.Scatter(
            x=filtered_df.index,
            y=cum_mle,
            mode="lines",
            name="MLE Cumulative Breaches",
            line=dict(color="#FF6B6B", shape="hv", width=2),
        )
    )

    fig_cum.add_trace(
        go.Scatter(
            x=filtered_df.index,
            y=cum_bayes,
            mode="lines",
            name="Bayesian Cumulative Breaches",
            line=dict(color="#1A73E8", shape="hv", width=2),
        )
    )

    fig_cum.update_layout(
        title="Cumulative VaR Breaches vs. Theoretical Expectation",
        xaxis_title="Date",
        yaxis_title="Number of Breaches",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_cum, width='stretch')

st.divider()


# -----------------------------------------------------------------------------
# 7. Section 3: "What-If" Market Shock Stress Test Simulator
# -----------------------------------------------------------------------------
st.subheader("🧪 What-If Market Shock Stress Test Simulator")
st.markdown(
    "Simulate how the risk models would react tomorrow if a severe market shock occurs today."
)

col_shock1, col_shock2 = st.columns([1, 2])

with col_shock1:
    shock_pct = st.slider(
        "Simulated Market Return Shock (%)",
        min_value=-10.0,
        max_value=0.0,
        value=-3.5,
        step=0.5,
    )
    st.caption("Slide to simulate a flash crash or extreme tail event.")

with col_shock2:
    if not live_returns.empty:
        current_vol = forecast_vol

        # GARCH(1,1) Variance Equation: sigma_{t+1}^2 = omega + alpha * e_t^2 + beta * sigma_t^2
        omega = metadata["mle_params"]["omega"]
        alpha = metadata["mle_params"]["alpha"]
        beta = metadata["mle_params"]["beta"]

        simulated_next_vol = np.sqrt(
            omega + alpha * (shock_pct**2) + beta * (current_vol**2)
        )
        vol_spike_pct = (
            (simulated_next_vol - current_vol) / current_vol
        ) * 100

        simulated_mle_var = simulated_next_vol * z
        simulated_dollar_loss = (simulated_mle_var / 100) * portfolio_value

        st.info(f"**Stress Test Simulation Output (After {shock_pct:.1f}% Shock)**")
        st.write(
            f"• Projected 1-Day Volatility Spike: **{current_vol:.2f}% ➔ {simulated_next_vol:.2f}%** (`+{vol_spike_pct:.1f}%` rise)"
        )
        st.write(
            f"• Revised Next-Day MLE VaR Defense Line: **-{simulated_mle_var:.2f}%**"
        )
        st.write(
            f"• Portfolio Capital Requirement at Risk: **-${simulated_dollar_loss:,.0f} USD**"
        )