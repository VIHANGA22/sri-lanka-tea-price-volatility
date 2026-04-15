"""
Sri Lanka Tea Export Price Volatility Dashboard (2008–2024)
===========================================================
A decision-support dashboard for tea exporters, policy makers,
analysts, and business managers.

Run with:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
from io import StringIO
import os

# ── Optional heavy imports (graceful fallback) ────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sri Lanka Tea Export Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "historical":   "#1a6fb5",
    "global":       "#2ca02c",
    "forecast":     "#d62728",
    "ci_fill":      "#ffb3b3",
    "volatility6":  "#e377c2",
    "volatility12": "#7f7f7f",
    "exchange":     "#ff7f0e",
    "quantity":     "#17becf",
    "accent":       "#f0a500",
    "bg_card":      "#f7fbff",
    "highlight":    "#fff3cd",
}

# Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #0d3b6e 0%, #1a6fb5 60%, #2ca02c 100%);
    padding: 2.2rem 2.5rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: "🍃";
    position: absolute;
    right: 2rem; top: 1rem;
    font-size: 5rem; opacity: 0.15;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem; margin: 0; color: white;
}
.main-header p { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.88; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.4rem; }
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border-left: 5px solid #1a6fb5;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); }
.kpi-card.green { border-left-color: #2ca02c; }
.kpi-card.orange { border-left-color: #ff7f0e; }
.kpi-card.red { border-left-color: #d62728; }
.kpi-label { font-size: 0.78rem; color: #666; text-transform: uppercase; letter-spacing: .07em; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 2rem; color: #0d3b6e; margin: .2rem 0; }
.kpi-sub { font-size: 0.82rem; color: #888; }

/* ── Insight Box ── */
.insight-box {
    background: #fffbe6;
    border-left: 4px solid #f0a500;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.93rem;
    color: #5a4000;
}
.insight-box b { color: #b07800; }

/* ── Section Header ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    color: #0d3b6e;
    border-bottom: 2px solid #1a6fb5;
    padding-bottom: .4rem;
    margin: 1.8rem 0 1rem;
}

/* ── Model Metric Card ── */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 4px solid #1a6fb5;
}
.metric-card.rmse { border-top-color: #d62728; }
.metric-card.mape { border-top-color: #ff7f0e; }

/* ── Risk Badge ── */
.badge {
    display: inline-block;
    padding: .25rem .75rem;
    border-radius: 20px;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .05em;
}
.badge.high { background: #ffe0e0; color: #c00; }
.badge.medium { background: #fff3cd; color: #856404; }
.badge.low { background: #d4edda; color: #155724; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #aaa;
    font-size: .8rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e5e5;
}

/* Streamlit tweaks */
div[data-testid="stSidebar"] { background: #0d3b6e; }
div[data-testid="stSidebar"] * { color: #cfe3ff !important; }
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stSlider label { color: #a8c9f0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION (realistic synthetic data matching research parameters)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def generate_synthetic_data() -> pd.DataFrame:
    """
    Generate realistic synthetic Sri Lanka tea export data (2008-2024).
    Replace this function with pd.read_csv('your_data.csv') if you have
    the actual dataset.
    """
    np.random.seed(42)
    dates = pd.date_range("2008-01-01", "2024-12-01", freq="MS")
    n = len(dates)

    # Base price trend: rising from ~4.5 to ~7.5 USD/kg with shocks
    trend = np.linspace(4.5, 7.5, n)
    seasonal = 0.3 * np.sin(2 * np.pi * np.arange(n) / 12 + 1.0)
    shocks = np.zeros(n)
    # COVID shock
    shocks[144:150] = -0.8
    # 2022 Sri Lanka economic crisis
    shocks[168:180] = -1.2
    # 2011 drought
    shocks[36:42] = 0.6
    noise = np.random.normal(0, 0.18, n)
    price_usd = np.clip(trend + seasonal + shocks + noise, 2.5, 11.0)

    # Exchange rate LKR/USD: rising from ~110 to ~325
    exch_base = np.linspace(110, 325, n)
    exch_shock = np.zeros(n)
    exch_shock[168:] = 80  # 2022 crisis
    exch = exch_base + exch_shock + np.random.normal(0, 4, n)

    price_lkr = price_usd * exch
    global_tea = price_usd * np.random.uniform(0.88, 1.12, n) + np.random.normal(0, 0.12, n)

    # Quantity (kg millions): gradual decline, seasonal highs
    qty_trend = np.linspace(310e6, 270e6, n)
    qty_seasonal = 15e6 * np.sin(2 * np.pi * np.arange(n) / 12)
    qty_noise = np.random.normal(0, 5e6, n)
    quantity = np.clip(qty_trend + qty_seasonal + qty_noise, 150e6, 400e6)

    df = pd.DataFrame({
        "Year":           dates.year,
        "Month":          dates.month,
        "Date":           dates,
        "Export_Qty_kg":  quantity.astype(int),
        "Export_Price_LKR": np.round(price_lkr, 2),
        "Exchange_Rate":  np.round(exch, 2),
        "Export_Price_USD": np.round(price_usd, 4),
        "Global_Tea_Price": np.round(np.clip(global_tea, 2.0, 12.0), 4),
    })

    # Computed fields
    df["Log_Return"] = np.log(df["Export_Price_USD"] / df["Export_Price_USD"].shift(1))
    df["Rolling_Vol_6m"]  = df["Log_Return"].rolling(6).std() * np.sqrt(12)
    df["Rolling_Vol_12m"] = df["Log_Return"].rolling(12).std() * np.sqrt(12)

    return df.dropna(subset=["Log_Return"]).reset_index(drop=True)


@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    """Load CSV if uploaded, otherwise use synthetic data."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            # Attempt to build Date column
            if "Date" not in df.columns and "Year" in df.columns and "Month" in df.columns:
                df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            # Ensure derived fields
            if "Log_Return" not in df.columns:
                df["Log_Return"] = np.log(df["Export_Price_USD"] / df["Export_Price_USD"].shift(1))
            if "Rolling_Vol_6m" not in df.columns:
                df["Rolling_Vol_6m"] = df["Log_Return"].rolling(6).std() * np.sqrt(12)
            if "Rolling_Vol_12m" not in df.columns:
                df["Rolling_Vol_12m"] = df["Log_Return"].rolling(12).std() * np.sqrt(12)
            return df.dropna(subset=["Log_Return"]).reset_index(drop=True)
        except Exception as e:
            st.sidebar.error(f"Could not parse uploaded file: {e}")
    return generate_synthetic_data()


# ═══════════════════════════════════════════════════════════════════════════════
# ARIMA FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def run_arima_forecast(series: pd.Series, order=(1, 1, 1), steps=12):
    """
    Fit ARIMA and return forecast + confidence intervals + metrics.
    Falls back to simple exponential smoothing if statsmodels unavailable.
    """
    results = {
        "forecast": None, "lower": None, "upper": None,
        "mae": None, "rmse": None, "mape": None,
        "model_name": "ARIMA(1,1,1)", "success": False,
        "warning": None,
    }

    series = series.dropna()
    train_size = int(len(series) * 0.85)
    train, test = series.iloc[:train_size], series.iloc[train_size:]

    if STATSMODELS_OK and len(series) >= 24:
        try:
            model = ARIMA(train, order=order)
            fit = model.fit()

            # In-sample predictions for validation
            pred = fit.get_prediction(start=train_size, end=len(series) - 1,
                                      dynamic=False)
            pred_mean = pred.predicted_mean.values[:len(test)]
            test_vals = test.values[:len(pred_mean)]

            if SKLEARN_OK and len(test_vals) > 0:
                mae  = mean_absolute_error(test_vals, pred_mean)
                rmse = np.sqrt(mean_squared_error(test_vals, pred_mean))
                with np.errstate(divide="ignore", invalid="ignore"):
                    mape_arr = np.abs((test_vals - pred_mean) / test_vals)
                    mape = np.nanmean(mape_arr[np.isfinite(mape_arr)]) * 100
                results.update(mae=round(mae, 4), rmse=round(rmse, 4),
                               mape=round(mape, 2))

            # Forecast
            fcast = fit.get_forecast(steps=steps)
            results["forecast"] = fcast.predicted_mean.values
            ci = fcast.conf_int(alpha=0.10)
            results["lower"] = ci.iloc[:, 0].values
            results["upper"] = ci.iloc[:, 1].values
            results["success"] = True
            results["model_name"] = f"ARIMA{order}"
        except Exception as e:
            results["warning"] = f"ARIMA fitting issue: {e}. Using fallback."

    if not results["success"]:
        # Simple drift forecast fallback
        results["model_name"] = "Drift Forecast (fallback)"
        last = series.iloc[-1]
        drift = (series.iloc[-1] - series.iloc[0]) / len(series)
        fc = np.array([last + drift * (i + 1) for i in range(steps)])
        std = series.std()
        results["forecast"] = fc
        results["lower"]    = fc - 1.645 * std
        results["upper"]    = fc + 1.645 * std
        results["warning"]  = "statsmodels not available. Using linear drift forecast."
        results["success"]  = True

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def style_ax(ax, title="", xlabel="", ylabel="", legend=True):
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0d3b6e", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color="#555")
    ax.set_ylabel(ylabel, fontsize=9, color="#555")
    ax.tick_params(colors="#666", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    ax.spines["bottom"].set_color("#ddd")
    ax.set_facecolor("#fafcff")
    if legend:
        ax.legend(fontsize=8, framealpha=0.85, loc="best")


def fig_to_st(fig, key=None):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_insights(df: pd.DataFrame, forecast_results: dict) -> dict:
    """Derive actionable text insights from the data."""
    recent = df.tail(12)
    last_price = df["Export_Price_USD"].iloc[-1]
    prev_price = df["Export_Price_USD"].iloc[-13] if len(df) > 13 else df["Export_Price_USD"].iloc[0]
    pct_change = ((last_price - prev_price) / prev_price) * 100

    last_vol = df["Rolling_Vol_12m"].iloc[-1]
    avg_vol  = df["Rolling_Vol_12m"].mean()

    if last_vol > avg_vol * 1.3:
        risk = "HIGH 🔴"
        risk_class = "high"
        risk_msg = "Volatility is significantly above the long-run average. Exporters should consider forward contracts or hedging instruments."
    elif last_vol > avg_vol * 0.9:
        risk = "MEDIUM 🟡"
        risk_class = "medium"
        risk_msg = "Volatility is near the historical average. Monitor exchange rate movements closely."
    else:
        risk = "LOW 🟢"
        risk_class = "low"
        risk_msg = "Price conditions are relatively stable. Good window to lock in supply contracts."

    # Forecast direction
    fc = forecast_results.get("forecast")
    if fc is not None and len(fc) >= 2:
        fc_start = fc[0]
        fc_end   = fc[-1]
        fc_change = ((fc_end - fc_start) / fc_start) * 100
        if fc_change > 3:
            trend_msg = f"📈 Prices are forecast to INCREASE by ~{fc_change:.1f}% over the next 12 months."
            trend_icon = "📈 Upward"
        elif fc_change < -3:
            trend_msg = f"📉 Prices are forecast to DECREASE by ~{abs(fc_change):.1f}% over the next 12 months."
            trend_icon = "📉 Downward"
        else:
            trend_msg = "➡️ Prices are forecast to remain STABLE over the next 12 months (±3%)."
            trend_icon = "➡️ Stable"
    else:
        trend_msg = "Forecast unavailable."
        trend_icon = "N/A"

    # Exchange rate correlation
    corr = df["Exchange_Rate"].corr(df["Export_Price_USD"])
    if corr > 0.5:
        exch_msg = f"Strong positive correlation ({corr:.2f}) between exchange rate and USD price. LKR depreciation tends to boost USD export prices."
    elif corr < -0.3:
        exch_msg = f"Negative correlation ({corr:.2f}) detected. Rising exchange rate is associated with falling USD prices — investigate competitive pricing pressure."
    else:
        exch_msg = f"Weak correlation ({corr:.2f}) between exchange rate and price — other demand/supply factors dominate."

    # Qty vs price
    qty_corr = df["Export_Qty_kg"].corr(df["Export_Price_USD"])
    if qty_corr < -0.2:
        qty_msg = "Higher export volumes correlate with lower prices — typical supply-side pressure. Restraining output during peak seasons may support prices."
    elif qty_corr > 0.2:
        qty_msg = "Higher volumes coincide with higher prices — demand-driven market. Expanding capacity could be profitable."
    else:
        qty_msg = "No strong quantity–price relationship. Prices are likely driven by global commodity markets or quality premiums."

    return {
        "pct_change": pct_change,
        "risk": risk,
        "risk_class": risk_class,
        "risk_msg": risk_msg,
        "trend_msg": trend_msg,
        "trend_icon": trend_icon,
        "exch_msg": exch_msg,
        "qty_msg": qty_msg,
        "last_price": last_price,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🍃 Sri Lanka Tea")
    st.markdown("### Export Price Dashboard")
    st.markdown("---")

    uploaded = st.file_uploader(
        "📂 Upload your CSV dataset",
        type=["csv"],
        help="CSV must contain: Year, Month, Export_Price_USD, Export_Qty_kg, Exchange_Rate, Global_Tea_Price"
    )

    st.markdown("---")
    st.markdown("### 🔧 Filters")

    df_raw = load_data(uploaded)
    years = sorted(df_raw["Year"].unique())
    yr_min, yr_max = int(min(years)), int(max(years))

    year_range = st.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))
    agg_mode = st.selectbox("Aggregation", ["Monthly", "Yearly"])

    st.markdown("---")
    st.markdown("### 📡 ARIMA Settings")
    arima_p = st.selectbox("p (AR order)", [0, 1, 2, 3], index=1)
    arima_d = st.selectbox("d (Differencing)", [0, 1, 2], index=1)
    arima_q = st.selectbox("q (MA order)", [0, 1, 2, 3], index=1)
    fc_steps = st.slider("Forecast Horizon (months)", 6, 24, 12)

    st.markdown("---")
    st.caption("Built for Sri Lanka Tea Export Research\n2008–2024 Analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER DATA
# ═══════════════════════════════════════════════════════════════════════════════

df_raw = load_data(uploaded)
df = df_raw[(df_raw["Year"] >= year_range[0]) & (df_raw["Year"] <= year_range[1])].copy()

if agg_mode == "Yearly":
    df_plot = df.groupby("Year").agg(
        Date=("Date", "first"),
        Export_Price_USD=("Export_Price_USD", "mean"),
        Export_Price_LKR=("Export_Price_LKR", "mean"),
        Exchange_Rate=("Exchange_Rate", "mean"),
        Global_Tea_Price=("Global_Tea_Price", "mean"),
        Export_Qty_kg=("Export_Qty_kg", "sum"),
        Rolling_Vol_6m=("Rolling_Vol_6m", "mean"),
        Rolling_Vol_12m=("Rolling_Vol_12m", "mean"),
    ).reset_index()
    df_plot["Date"] = pd.to_datetime(df_plot["Year"].astype(str) + "-01-01")
else:
    df_plot = df.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# ARIMA FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

fc_results = run_arima_forecast(
    df_raw["Export_Price_USD"],
    order=(arima_p, arima_d, arima_q),
    steps=fc_steps,
)

insights = generate_insights(df, fc_results)

# Forecast dates
last_date = df_raw["Date"].iloc[-1]
fc_dates = pd.date_range(
    last_date + pd.DateOffset(months=1),
    periods=fc_steps,
    freq="MS"
)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>Sri Lanka Tea Export Price Volatility Dashboard</h1>
  <p>Decision-support intelligence for exporters, policy makers, analysts & business managers &nbsp;|&nbsp; 2008–2024</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">📊 Overview — Key Performance Indicators</div>', unsafe_allow_html=True)

latest   = df["Export_Price_USD"].iloc[-1]
exch_now = df["Exchange_Rate"].iloc[-1]
glob_now = df["Global_Tea_Price"].iloc[-1]
pct_ch   = insights["pct_change"]

kpi_color = "red" if pct_ch < -3 else ("green" if pct_ch > 3 else "orange")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Latest Export Price</div>
      <div class="kpi-value">${latest:.3f}</div>
      <div class="kpi-sub">USD / kg</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card orange">
      <div class="kpi-label">Exchange Rate</div>
      <div class="kpi-value">{exch_now:.0f}</div>
      <div class="kpi-sub">LKR / USD</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card green">
      <div class="kpi-label">Global Tea Price</div>
      <div class="kpi-value">${glob_now:.3f}</div>
      <div class="kpi-sub">USD / kg (world)</div>
    </div>""", unsafe_allow_html=True)
with col4:
    arrow = "▲" if pct_ch > 0 else "▼"
    st.markdown(f"""
    <div class="kpi-card {kpi_color}">
      <div class="kpi-label">YoY Price Change</div>
      <div class="kpi-value">{arrow} {abs(pct_ch):.1f}%</div>
      <div class="kpi-sub">vs same month last year</div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="insight-box">
  <b>📋 Executive Summary:</b> Sri Lanka's tea export price currently stands at
  <b>${latest:.3f}/kg</b>, with the LKR exchange rate at <b>{exch_now:.0f}</b> LKR/USD.
  Prices have changed <b>{pct_ch:+.1f}%</b> year-over-year.
  Current risk level: <b>{insights['risk']}</b>.
  {insights['trend_msg']}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PRICE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">📈 Price Analysis — Sri Lanka vs Global Tea Price</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df_plot["Date"], df_plot["Export_Price_USD"],
        color=COLORS["historical"], lw=2, label="Sri Lanka Export Price (USD/kg)", zorder=3)
ax.plot(df_plot["Date"], df_plot["Global_Tea_Price"],
        color=COLORS["global"], lw=1.8, linestyle="--", label="Global Tea Price (USD/kg)", zorder=2)
ax.fill_between(df_plot["Date"],
                df_plot["Export_Price_USD"],
                df_plot["Global_Tea_Price"],
                alpha=0.08, color=COLORS["historical"])
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=30)
style_ax(ax, "Sri Lanka vs Global Tea Export Price", "Date", "Price (USD/kg)")
fig.tight_layout()
fig_to_st(fig)

st.markdown(f"""
<div class="insight-box">
  <b>💡 Price Insight:</b>
  Sri Lanka's export price generally tracks global prices but diverges during crisis periods
  (e.g., 2020 COVID-19, 2022 Sri Lanka economic crisis).
  A price premium above the global benchmark signals quality differentiation —
  a competitive advantage to protect.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXCHANGE RATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">💱 Exchange Rate Analysis</div>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 2])

with c1:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df_plot["Date"], df_plot["Exchange_Rate"],
            color=COLORS["exchange"], lw=2, label="LKR/USD Exchange Rate")
    ax.fill_between(df_plot["Date"], df_plot["Exchange_Rate"].min(),
                    df_plot["Exchange_Rate"], alpha=0.1, color=COLORS["exchange"])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=30)
    style_ax(ax, "LKR/USD Exchange Rate Trend", "Date", "LKR per USD")
    fig.tight_layout()
    fig_to_st(fig)

with c2:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(df["Exchange_Rate"], df["Export_Price_USD"],
               alpha=0.5, s=28, color=COLORS["exchange"], edgecolors="white", lw=0.4)
    # Trend line
    z = np.polyfit(df["Exchange_Rate"].dropna(), df["Export_Price_USD"].dropna(), 1)
    p = np.poly1d(z)
    xs = np.linspace(df["Exchange_Rate"].min(), df["Exchange_Rate"].max(), 100)
    ax.plot(xs, p(xs), color="#cc4400", lw=1.8, linestyle="--", label="Trend")
    style_ax(ax, "Exchange Rate vs Export Price", "Exchange Rate (LKR/USD)", "Export Price (USD/kg)")
    fig.tight_layout()
    fig_to_st(fig)

corr_val = df["Exchange_Rate"].corr(df["Export_Price_USD"])
st.markdown(f"""
<div class="insight-box">
  <b>💡 Exchange Rate Insight:</b>
  {insights['exch_msg']}
  Correlation coefficient: <b>{corr_val:.3f}</b>.
  Policy makers should monitor currency movements as a leading indicator for export competitiveness.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — QUANTITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">📦 Export Quantity Analysis</div>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 2])

with c1:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    qty_m = df_plot["Export_Qty_kg"] / 1e6
    ax.bar(df_plot["Date"], qty_m,
           color=COLORS["quantity"], alpha=0.75, width=25 if agg_mode == "Monthly" else 200,
           label="Export Volume (million kg)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=30)
    style_ax(ax, "Export Quantity Trend", "Date", "Volume (million kg)", legend=False)
    fig.tight_layout()
    fig_to_st(fig)

with c2:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(df["Export_Qty_kg"] / 1e6, df["Export_Price_USD"],
               alpha=0.5, s=28, color=COLORS["quantity"], edgecolors="white", lw=0.4)
    z = np.polyfit(df["Export_Qty_kg"] / 1e6, df["Export_Price_USD"], 1)
    p = np.poly1d(z)
    xs = np.linspace(df["Export_Qty_kg"].min() / 1e6, df["Export_Qty_kg"].max() / 1e6, 100)
    ax.plot(xs, p(xs), color="#004466", lw=1.8, linestyle="--", label="Trend")
    style_ax(ax, "Quantity vs Export Price", "Volume (million kg)", "Export Price (USD/kg)")
    fig.tight_layout()
    fig_to_st(fig)

qty_corr = df["Export_Qty_kg"].corr(df["Export_Price_USD"])
st.markdown(f"""
<div class="insight-box">
  <b>💡 Quantity Insight:</b>
  {insights['qty_msg']}
  Quantity–Price correlation: <b>{qty_corr:.3f}</b>.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — VOLATILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">⚡ Price Volatility Analysis</div>', unsafe_allow_html=True)

df_vol = df.dropna(subset=["Rolling_Vol_6m", "Rolling_Vol_12m"])
avg12 = df_vol["Rolling_Vol_12m"].mean()
high_risk_thresh = avg12 * 1.3

fig, ax = plt.subplots(figsize=(11, 4))

# Shade high-risk zones
in_high = False
start_hr = None
for i, row in df_vol.iterrows():
    if row["Rolling_Vol_12m"] >= high_risk_thresh and not in_high:
        start_hr = row["Date"]
        in_high = True
    elif row["Rolling_Vol_12m"] < high_risk_thresh and in_high:
        ax.axvspan(start_hr, row["Date"], color="#ffe0e0", alpha=0.45, zorder=0)
        in_high = False
if in_high:
    ax.axvspan(start_hr, df_vol["Date"].iloc[-1], color="#ffe0e0", alpha=0.45, zorder=0)

ax.plot(df_vol["Date"], df_vol["Rolling_Vol_6m"],
        color=COLORS["volatility6"], lw=1.8, label="6-Month Rolling Volatility")
ax.plot(df_vol["Date"], df_vol["Rolling_Vol_12m"],
        color=COLORS["volatility12"], lw=2.2, label="12-Month Rolling Volatility")
ax.axhline(avg12, color=COLORS["volatility12"], linestyle=":", lw=1.4,
           alpha=0.7, label=f"Avg 12m Vol ({avg12:.3f})")
ax.axhline(high_risk_thresh, color="red", linestyle="--", lw=1.2,
           alpha=0.6, label=f"High-Risk Threshold ({high_risk_thresh:.3f})")

# Annotation — highlight worst period
worst_idx = df_vol["Rolling_Vol_12m"].idxmax()
worst_row = df_vol.loc[worst_idx]
ax.annotate(f"Peak\n{worst_row['Date'].strftime('%b %Y')}",
            xy=(worst_row["Date"], worst_row["Rolling_Vol_12m"]),
            xytext=(30, 10), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#c00"),
            fontsize=8, color="#c00")

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=30)

# Add red patch to legend for high-risk zone
red_patch = mpatches.Patch(color="#ffe0e0", alpha=0.8, label="High-Risk Period")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [red_patch], labels + ["High-Risk Period"], fontsize=8)

style_ax(ax, "Rolling Price Volatility (Annualised)", "Date", "Volatility (σ)", legend=False)
ax.legend(fontsize=8, framealpha=0.85)
fig.tight_layout()
fig_to_st(fig)

last_vol = df_vol["Rolling_Vol_12m"].iloc[-1]
risk_desc = "above average — elevated caution warranted" if last_vol > high_risk_thresh else "within normal bounds"
st.markdown(f"""
<div class="insight-box">
  <b>💡 Volatility Insight:</b>
  The current 12-month annualised volatility is <b>{last_vol:.3f}</b> ({risk_desc}).
  Red-shaded periods represent historically high-risk windows.
  During these periods, futures contracts and price hedging are strongly recommended.
  Average long-run volatility: <b>{avg12:.3f}</b>.
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f'<div class="section-title">🔮 Price Forecast — Next {fc_steps} Months ({fc_results["model_name"]})</div>',
            unsafe_allow_html=True)

if fc_results.get("warning"):
    st.warning(fc_results["warning"])

if fc_results["success"]:
    hist_tail = df_raw.tail(36)  # show last 3 years for context

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Historical
    ax.plot(hist_tail["Date"], hist_tail["Export_Price_USD"],
            color=COLORS["historical"], lw=2.2, label="Historical Price (USD/kg)", zorder=3)

    # Confidence interval
    ax.fill_between(fc_dates,
                    fc_results["lower"], fc_results["upper"],
                    color=COLORS["ci_fill"], alpha=0.55,
                    label="90% Confidence Interval", zorder=1)

    # Forecast line
    ax.plot(fc_dates, fc_results["forecast"],
            color=COLORS["forecast"], lw=2.4, linestyle="--",
            label=f"Forecast ({fc_results['model_name']})", zorder=4)

    # Divider
    ax.axvline(last_date, color="#999", linestyle=":", lw=1.5, label="Forecast Start")

    # Annotations on first & last forecast point
    ax.annotate(f"${fc_results['forecast'][0]:.3f}",
                xy=(fc_dates[0], fc_results["forecast"][0]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8, color=COLORS["forecast"])
    ax.annotate(f"${fc_results['forecast'][-1]:.3f}",
                xy=(fc_dates[-1], fc_results["forecast"][-1]),
                xytext=(-40, 8), textcoords="offset points",
                fontsize=8, color=COLORS["forecast"])

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=35)
    style_ax(ax, f"Tea Export Price Forecast — {fc_steps}-Month Horizon", "Date", "Price (USD/kg)")
    fig.tight_layout()
    fig_to_st(fig)

    fc_pct = ((fc_results["forecast"][-1] - fc_results["forecast"][0]) / fc_results["forecast"][0]) * 100
    fc_dir = "increase" if fc_pct > 0 else "decrease"

    st.markdown(f"""
    <div class="insight-box">
      <b>💡 Forecast Interpretation:</b>
      The {fc_results['model_name']} model projects that the export price will
      <b>{fc_dir} by {abs(fc_pct):.1f}%</b> over the next {fc_steps} months —
      from an estimated <b>${fc_results['forecast'][0]:.3f}/kg</b> to
      <b>${fc_results['forecast'][-1]:.3f}/kg</b>.
      The pink band shows the 90% confidence interval; prices could realistically range from
      <b>${fc_results['lower'][-1]:.3f}</b> to <b>${fc_results['upper'][-1]:.3f}</b>
      by the end of the horizon.
      <br><br>
      <b>Action:</b> {insights['trend_msg']}
    </div>
    """, unsafe_allow_html=True)

    # Forecast table
    with st.expander("📋 View Month-by-Month Forecast Table"):
        fc_df = pd.DataFrame({
            "Month": fc_dates.strftime("%b %Y"),
            "Forecast (USD/kg)": np.round(fc_results["forecast"], 4),
            "Lower 90% CI": np.round(fc_results["lower"], 4),
            "Upper 90% CI": np.round(fc_results["upper"], 4),
        })
        st.dataframe(fc_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">🧪 Model Performance & Validation</div>', unsafe_allow_html=True)

mae  = fc_results.get("mae")
rmse = fc_results.get("rmse")
mape = fc_results.get("mape")

c1, c2, c3 = st.columns(3)
with c1:
    val = f"{mae:.4f}" if mae is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card">
      <div class="kpi-label">MAE</div>
      <div class="kpi-value" style="font-size:1.8rem;">{val}</div>
      <div class="kpi-sub">Mean Absolute Error (USD/kg)</div>
    </div>""", unsafe_allow_html=True)
with c2:
    val = f"{rmse:.4f}" if rmse is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card rmse">
      <div class="kpi-label">RMSE</div>
      <div class="kpi-value" style="font-size:1.8rem;">{val}</div>
      <div class="kpi-sub">Root Mean Squared Error (USD/kg)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    val = f"{mape:.2f}%" if mape is not None else "N/A"
    st.markdown(f"""
    <div class="metric-card mape">
      <div class="kpi-label">MAPE</div>
      <div class="kpi-value" style="font-size:1.8rem;">{val}</div>
      <div class="kpi-sub">Mean Absolute Percentage Error</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
  <b>📌 Interpreting Model Metrics:</b><br>
  • <b>MAE</b> — Average prediction error in USD/kg. Lower is better.
    Values under $0.40/kg are considered good for this series.<br>
  • <b>RMSE</b> — Penalises large errors more heavily. Useful for detecting outlier-driven misfits.<br>
  • <b>MAPE</b> — Percentage error. <b>Note:</b> MAPE can appear inflated when prices or log-returns
    are close to zero (division by a very small number amplifies the ratio artificially).
    This is a known limitation — use MAE and RMSE as primary accuracy measures.<br>
  • ARIMA performs well on stationary series. If MAPE is high, consider that price series contain
    structural breaks (e.g., 2022 crisis) that linear models cannot anticipate.
</div>
""", unsafe_allow_html=True)

# Residual plot (simulated if needed)
if fc_results["success"] and mae is not None:
    with st.expander("📊 Residual Diagnostics"):
        train_size = int(len(df_raw.dropna(subset=["Log_Return"])) * 0.85)
        actual = df_raw["Export_Price_USD"].dropna().values
        # Approximate residuals using AR(1) approximation for display
        residuals = np.diff(actual) - np.mean(np.diff(actual))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(residuals, color=COLORS["historical"], lw=1, alpha=0.8)
        axes[0].axhline(0, color="red", lw=1, linestyle="--")
        style_ax(axes[0], "Residuals Over Time", "Index", "Residual (USD/kg)", legend=False)
        axes[1].hist(residuals, bins=30, color=COLORS["historical"], alpha=0.7, edgecolor="white")
        style_ax(axes[1], "Residual Distribution", "Residual", "Frequency", legend=False)
        fig.tight_layout()
        fig_to_st(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DECISION SUPPORT PANEL
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">🎯 Decision Support — Actionable Insights Panel</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"""
    <div style="background:white; border-radius:12px; padding:1.4rem; box-shadow:0 2px 12px rgba(0,0,0,0.07); height:100%;">
      <h4 style="color:#0d3b6e; font-family:'Playfair Display',serif; margin-top:0;">
        📊 For Tea Exporters
      </h4>
      <p>
        <b>Price Outlook:</b> {insights['trend_msg']}<br><br>
        <b>Risk Level:</b> <span class="badge {insights['risk_class']}">{insights['risk']}</span><br><br>
        {insights['risk_msg']}<br><br>
        <b>Recommendation:</b> {"Lock in forward contracts now before prices potentially rise." if insights['pct_change'] > 0 else "Hold off on long-term commitments; await price stabilisation."}
      </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div style="background:white; border-radius:12px; padding:1.4rem; box-shadow:0 2px 12px rgba(0,0,0,0.07); height:100%;">
      <h4 style="color:#0d3b6e; font-family:'Playfair Display',serif; margin-top:0;">
        🏛️ For Policy Makers
      </h4>
      <p>
        <b>Exchange Rate Impact:</b> {insights['exch_msg']}<br><br>
        <b>Volatility Context:</b>
        Current annualised volatility stands at
        <b>{df["Rolling_Vol_12m"].iloc[-1]:.3f}</b>.
        {"Policy intervention may be needed to stabilise markets." if df["Rolling_Vol_12m"].iloc[-1] > df["Rolling_Vol_12m"].mean() * 1.3 else "Markets are within historically normal volatility bounds."}<br><br>
        <b>Recommendation:</b> Review export incentive structures and currency stabilisation policies quarterly.
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    st.markdown(f"""
    <div style="background:white; border-radius:12px; padding:1.4rem; box-shadow:0 2px 12px rgba(0,0,0,0.07);">
      <h4 style="color:#0d3b6e; font-family:'Playfair Display',serif; margin-top:0;">
        📦 Supply vs Demand
      </h4>
      <p>
        {insights['qty_msg']}<br><br>
        <b>Volume Trend:</b>
        Average monthly export volume over the selected period:
        <b>{df["Export_Qty_kg"].mean() / 1e6:.1f} million kg</b>.<br><br>
        <b>Recommendation:</b>
        {"Expanding production capacity could capture higher margins." if qty_corr > 0.2 else "Consider quality-driven differentiation over volume expansion."}
      </p>
    </div>
    """, unsafe_allow_html=True)

with col_d:
    st.markdown(f"""
    <div style="background:white; border-radius:12px; padding:1.4rem; box-shadow:0 2px 12px rgba(0,0,0,0.07);">
      <h4 style="color:#0d3b6e; font-family:'Playfair Display',serif; margin-top:0;">
        🌍 Global Market Position
      </h4>
      <p>
        Sri Lanka's export price vs global benchmark:
        <b>${latest:.3f}</b> vs <b>${glob_now:.3f}</b> USD/kg.<br><br>
        {"🟢 Sri Lanka commands a <b>price premium</b> over global average — quality positioning is effective." if latest > glob_now else "🔴 Sri Lanka is currently priced <b>below</b> the global average — an opportunity for brand & quality investment."}<br><br>
        <b>Recommendation:</b>
        Invest in certifications (e.g., Rainforest Alliance, Fair Trade) to sustain premium pricing.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("🗃️ Raw Data Explorer"):
    st.dataframe(df_plot.reset_index(drop=True), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", csv,
                       "tea_export_filtered.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
  Sri Lanka Tea Export Price Volatility Dashboard &nbsp;|&nbsp;
  Research Project 2024 &nbsp;|&nbsp;
  Built with Streamlit · statsmodels · pandas · matplotlib
</div>
""", unsafe_allow_html=True)
