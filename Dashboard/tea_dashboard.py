"""
Sri Lankan Tea Price Volatility Analysis Dashboard
====================================================
A Streamlit dashboard for academic research on Sri Lankan tea export
price volatility and ARIMA-based forecasting (2008–2024).

Usage:
    streamlit run tea_dashboard.py

Dataset expected:  Cleaned_Merged_Tea_Dataset.csv
    Columns: Unnamed: 0 (date), Year, Month, Export Quantity (kg),
             Export Price (LKR/kg), Exchange Rate (LKR/USD),
             Export Price (USD/kg), Global_Tea_Price ($), Log_Return
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lankan Tea Price Volatility Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f7f9fc; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a2a3a;
        color: #e8eeF5;
    }
    section[data-testid="stSidebar"] * { color: #cdd8e6 !important; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 0.93rem; }

    /* Section header cards */
    .section-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2e6b9e 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        color: #ffffff !important;
    }
    .section-header h2 { color: #ffffff !important; margin: 0; font-size: 1.35rem; }
    .section-header p  { color: #cde4f5 !important; margin: 0.25rem 0 0; font-size: 0.88rem; }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #dce5ef;
        border-left: 4px solid #2e6b9e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .kpi-label { font-size: 0.80rem; color: #6b7f96; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.06em; }
    .kpi-value { font-size: 1.65rem; color: #1a3a5c; font-weight: 700;
                 margin-top: 0.2rem; }

    /* Metric boxes for model validation */
    .metric-box {
        background: #edf3fb;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        text-align: center;
        border: 1px solid #c8d9ed;
    }
    .metric-label { font-size: 0.78rem; color: #4a6080; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.5rem; color: #1a3a5c; font-weight: 700; margin-top: 0.15rem; }

    /* Info box */
    .info-box {
        background: #eef6ff;
        border-left: 4px solid #2e86de;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem;
        margin-top: 0.75rem;
        font-size: 0.87rem;
        color: #1a3a5c;
    }

    /* Warning box */
    .warn-box {
        background: #fff8e6;
        border-left: 4px solid #e0a800;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem;
        margin-top: 0.75rem;
        font-size: 0.87rem;
        color: #5a4000;
    }

    /* Hide default Streamlit hamburger & footer */
    #MainMenu, footer { visibility: hidden; }

    /* Chart container spacing */
    .chart-container { margin-bottom: 1.5rem; }

    div[data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLOUR PALETTE  (consistent across all plots)
# ─────────────────────────────────────────────
C_SL   = "#2e86de"   # Sri Lanka export price
C_GLOB = "#e84393"   # Global tea price
C_EX   = "#27ae60"   # Exchange rate
C_QTY  = "#e67e22"   # Export quantity
C_FORE = "#8e44ad"   # Forecast
C_CI   = "rgba(142,68,173,0.15)"  # Forecast CI
C_VOL6 = "#e74c3c"   # 6-month rolling vol
C_VOL12= "#c0392b"   # 12-month rolling vol

PLOT_LAYOUT = dict(
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(family="Georgia, serif", size=12, color="#1a2a3a"),
    title_font=dict(family="Georgia, serif", size=14, color="#1a2a3a"),
    legend=dict(bgcolor="#f7f9fc", bordercolor="#dce5ef", borderwidth=1),
    margin=dict(l=55, r=25, t=55, b=45),
    hovermode="x unified",
)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and prepare the tea dataset. Accepts both CSV and XLSX."""
    try:
        df = pd.read_csv("Cleaned_Merged_Tea_Dataset.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        try:
            df = pd.read_excel("Cleaned_Merged_Tea_Dataset.xlsx")
            df = df.rename(columns={"Unnamed: 0": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        except FileNotFoundError:
            st.error("❌ Dataset file not found. Place `Cleaned_Merged_Tea_Dataset.csv` "
                     "or `Cleaned_Merged_Tea_Dataset.xlsx` in the same folder as this script.")
            st.stop()

    # Normalise index
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Ensure log returns exist
    if "Log_Return" not in df.columns:
        df["Log_Return"] = np.log(df["Export Price (USD/kg)"]).diff()

    # Rolling volatility (std of log returns × √12 for annualised)
    df["Rolling_Vol_6M"]  = df["Log_Return"].rolling(6).std()  * np.sqrt(12)
    df["Rolling_Vol_12M"] = df["Log_Return"].rolling(12).std() * np.sqrt(12)

    return df


@st.cache_data
def fit_arima(series: pd.Series):
    """Fit ARIMA(1,0,0) on the export price series and return model results."""
    model  = ARIMA(series.dropna(), order=(1, 0, 0))
    result = model.fit()
    return result


@st.cache_data
def compute_forecast(_result, steps: int = 12):
    """Generate out-of-sample forecast with confidence intervals."""
    fc = _result.get_forecast(steps=steps)
    mean_fc  = fc.predicted_mean
    conf_int = fc.conf_int(alpha=0.05)
    return mean_fc, conf_int


@st.cache_data
def in_sample_metrics(_result, series: pd.Series):
    """Compute in-sample MAE, RMSE, MAPE for the fitted model."""
    fitted = _result.fittedvalues
    actual = series.dropna()

    # Align indices
    idx    = actual.index.intersection(fitted.index)
    actual = actual[idx]
    fitted = fitted[idx]

    mae  = mean_absolute_error(actual, fitted)
    rmse = np.sqrt(mean_squared_error(actual, fitted))

    # MAPE (guard against zero/near-zero prices)
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - fitted[nonzero]) / actual[nonzero])) * 100

    return mae, rmse, mape, actual, fitted


# ─────────────────────────────────────────────
# LOAD DATA & FIT MODEL
# ─────────────────────────────────────────────
df     = load_data()
price  = df["Export Price (USD/kg)"].dropna()
result = fit_arima(price)
fc_mean, fc_ci = compute_forecast(result, steps=12)
mae, rmse, mape, actual, fitted = in_sample_metrics(result, price)

# Forecast index (months after last data point)
last_date  = df.index[-1]
fc_index   = pd.date_range(start=last_date + pd.DateOffset(months=1),
                            periods=12, freq="MS")
fc_mean.index    = fc_index
fc_ci.index      = fc_index

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding:0.6rem 0 1rem;'>
            <span style='font-size:2.2rem;'>🍃</span><br>
            <span style='font-size:1.05rem; font-weight:700; color:#aed6f1;'>
                Tea Price Dashboard
            </span><br>
            <span style='font-size:0.75rem; color:#7fb3d3;'>
                Sri Lanka · 2008 – 2024
            </span>
        </div>
        <hr style='border-color:#2e4a68; margin:0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    section = st.radio(
        "Navigate to",
        options=[
            "📌 Overview",
            "📈 Time Series",
            "🌍 Exchange Rate",
            "📦 Export Quantity",
            "📊 Correlation Heatmap",
            "🔍 Volatility Analysis",
            "🤖 ARIMA Model",
            "📉 Forecast",
            "✅ Model Validation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#2e4a68; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.73rem; color:#7fb3d3; line-height:1.6;'>"
        "📚 <b>Research Project</b><br>"
        "Sri Lankan Tea Export Price<br>Volatility &amp; Forecasting<br>"
        "<br>Model: ARIMA(1, 0, 0)<br>"
        "Data: 2008 – 2024 (Monthly)"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HELPER: section header
# ─────────────────────────────────────────────
def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="section-header">
               <h2>{title}</h2>
               {"<p>" + subtitle + "</p>" if subtitle else ""}
            </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# 1.  OVERVIEW
# ══════════════════════════════════════════════
if section == "📌 Overview":
    st.markdown("""
        <h1 style='color:#1a3a5c; margin-bottom:0.2rem; font-size:1.9rem;'>
            Sri Lankan Tea Price Volatility Analysis Dashboard
        </h1>
        <p style='color:#4a6080; font-size:1rem; margin-top:0;'>
            An analytical tool supporting the study of export price dynamics,
            exchange rate interactions, and ARIMA-based price forecasting.
        </p>
        <hr style='border-color:#dce5ef; margin:0.8rem 0 1.4rem;'>
    """, unsafe_allow_html=True)

    # Research description
    st.markdown("""
        <div class="info-box">
        <b>Research Context</b><br>
        This dashboard supports a final-year research project investigating the volatility
        of Sri Lanka's tea export prices between January 2008 and December 2024.
        Sri Lanka is one of the world's largest tea exporters, and price fluctuations
        have significant economic implications. The analysis examines the relationship
        between export prices (USD/kg), exchange rate movements (LKR/USD), global
        benchmark tea prices, and export volumes. An ARIMA(1,0,0) model is used for
        short-term price forecasting, and rolling volatility metrics are employed to
        characterise risk over time.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI cards
    mean_price = price.mean()
    max_price  = price.max()
    min_price  = price.min()
    avg_qty    = df["Export Quantity (kg)"].mean()
    n_months   = len(df)
    avg_exrate = df["Exchange Rate (LKR/USD)"].mean()

    cols = st.columns(6)
    kpis = [
        ("Mean Export Price", f"${mean_price:.2f}", "USD/kg"),
        ("Max Export Price",  f"${max_price:.2f}",  "USD/kg"),
        ("Min Export Price",  f"${min_price:.2f}",  "USD/kg"),
        ("Avg Export Qty",    f"{avg_qty/1e6:.2f}M", "kg/month"),
        ("Avg Exchange Rate", f"{avg_exrate:.0f}",   "LKR/USD"),
        ("Months of Data",    str(n_months),          "Jan 2008 – Dec 2024"),
    ]
    for col, (label, value, unit) in zip(cols, kpis):
        col.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div style='font-size:0.75rem; color:#8096b0; margin-top:0.15rem;'>{unit}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # Quick mini-chart: full price series
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Price Series Overview", "Sri Lanka export price vs global benchmark tea price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Export Price (USD/kg)"],
                             name="SL Export Price (USD/kg)",
                             line=dict(color=C_SL, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["Global_Tea_Price ($)"],
                             name="Global Tea Price ($)",
                             line=dict(color=C_GLOB, width=2)))
    fig.update_layout(
        **PLOT_LAYOUT,
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# 2.  TIME SERIES VISUALISATION
# ══════════════════════════════════════════════
elif section == "📈 Time Series":
    section_header("📈 Time Series Analysis",
                   "Sri Lanka export price and global tea benchmark price (interactive)")

    # Date range selector
    col1, col2 = st.columns(2)
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    with col1:
        start_date = st.date_input("Start date", value=min_date,
                                   min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End date",   value=max_date,
                                  min_value=min_date, max_value=max_date)

    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    dff  = df.loc[mask]

    if dff.empty:
        st.warning("No data in selected date range.")
    else:
        # Dual-axis line chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=dff.index, y=dff["Export Price (USD/kg)"],
            name="SL Export Price (USD/kg)",
            line=dict(color=C_SL, width=2.2),
            hovertemplate="%{x|%b %Y}: $%{y:.3f}<extra>SL Export Price</extra>",
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=dff.index, y=dff["Global_Tea_Price ($)"],
            name="Global Tea Price ($)",
            line=dict(color=C_GLOB, width=2.2, dash="dot"),
            hovertemplate="%{x|%b %Y}: $%{y:.3f}<extra>Global Price</extra>",
        ), secondary_y=True)

        fig.update_layout(
            **PLOT_LAYOUT,
            title="Sri Lanka Export Price vs Global Tea Price",
            height=430,
        )
        fig.update_yaxes(title_text="SL Export Price (USD/kg)",  secondary_y=False,
                         gridcolor="#e8edf3")
        fig.update_yaxes(title_text="Global Tea Price (USD/kg)", secondary_y=True)
        fig.update_xaxes(title_text="Date", gridcolor="#e8edf3")

        st.plotly_chart(fig, use_container_width=True)

        # Descriptive statistics table
        st.markdown("**Descriptive Statistics for Selected Period**")
        stat_df = dff[["Export Price (USD/kg)", "Global_Tea_Price ($)"]].describe().T
        stat_df.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        st.dataframe(stat_df.style.format("{:.4f}"), use_container_width=True)


# ══════════════════════════════════════════════
# 3.  EXCHANGE RATE ANALYSIS
# ══════════════════════════════════════════════
elif section == "🌍 Exchange Rate":
    section_header("🌍 Exchange Rate Analysis",
                   "LKR/USD exchange rate over time and its relationship to export prices")

    # Line chart: Exchange Rate
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df.index, y=df["Exchange Rate (LKR/USD)"],
        name="Exchange Rate (LKR/USD)",
        line=dict(color=C_EX, width=2),
        fill="tozeroy",
        fillcolor="rgba(39,174,96,0.07)",
        hovertemplate="%{x|%b %Y}: %{y:.1f} LKR/USD<extra></extra>",
    ))
    fig1.update_layout(
        **PLOT_LAYOUT,
        title="LKR/USD Exchange Rate (2008 – 2024)",
        yaxis_title="LKR per USD",
        xaxis_title="Date",
        height=350,
    )
    fig1.update_yaxes(gridcolor="#e8edf3")
    fig1.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # Scatter: Exchange Rate vs Export Price
    fig2 = px.scatter(
        df.reset_index(),
        x="Exchange Rate (LKR/USD)",
        y="Export Price (USD/kg)",
        color=df.index.year,
        color_continuous_scale="Blues",
        labels={
            "Exchange Rate (LKR/USD)": "Exchange Rate (LKR/USD)",
            "Export Price (USD/kg)":   "Export Price (USD/kg)",
            "color":                   "Year",
        },
        title="Exchange Rate vs Export Price (USD/kg)",
        hover_data={"color": False},
        trendline="ols",
    )
    fig2.update_traces(marker=dict(size=7, opacity=0.8))
    fig2.update_layout(**PLOT_LAYOUT, height=400)
    fig2.update_yaxes(gridcolor="#e8edf3")
    fig2.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig2, use_container_width=True)

    corr_val = df["Exchange Rate (LKR/USD)"].corr(df["Export Price (USD/kg)"])
    st.markdown(
        f'<div class="info-box">Pearson correlation between Exchange Rate and '
        f'Export Price (USD/kg): <b>{corr_val:.4f}</b></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# 4.  EXPORT QUANTITY ANALYSIS
# ══════════════════════════════════════════════
elif section == "📦 Export Quantity":
    section_header("📦 Export Quantity Analysis",
                   "Monthly export volume trends and relationship to price")

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df.index,
        y=df["Export Quantity (kg)"] / 1e6,
        name="Export Quantity",
        marker_color=C_QTY,
        opacity=0.75,
        hovertemplate="%{x|%b %Y}: %{y:.2f}M kg<extra></extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=(df["Export Quantity (kg)"] / 1e6).rolling(12).mean(),
        name="12-Month Rolling Avg",
        line=dict(color="#1a3a5c", width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.2f}M kg<extra>12M Avg</extra>",
    ))
    fig1.update_layout(
        **PLOT_LAYOUT,
        title="Monthly Export Quantity (million kg)",
        yaxis_title="Export Quantity (million kg)",
        xaxis_title="Date",
        height=360,
        barmode="overlay",
    )
    fig1.update_yaxes(gridcolor="#e8edf3")
    fig1.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    fig2 = px.scatter(
        df.reset_index(),
        x="Export Quantity (kg)",
        y="Export Price (USD/kg)",
        color=df.index.year,
        color_continuous_scale="Oranges",
        labels={
            "Export Quantity (kg)":  "Export Quantity (kg)",
            "Export Price (USD/kg)": "Export Price (USD/kg)",
            "color":                 "Year",
        },
        title="Export Quantity vs Export Price",
        trendline="ols",
    )
    fig2.update_traces(marker=dict(size=7, opacity=0.8))
    fig2.update_layout(**PLOT_LAYOUT, height=400)
    fig2.update_yaxes(gridcolor="#e8edf3")
    fig2.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig2, use_container_width=True)

    corr_val = df["Export Quantity (kg)"].corr(df["Export Price (USD/kg)"])
    st.markdown(
        f'<div class="info-box">Pearson correlation between Export Quantity and '
        f'Export Price (USD/kg): <b>{corr_val:.4f}</b></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# 5.  CORRELATION HEATMAP
# ══════════════════════════════════════════════
elif section == "📊 Correlation Heatmap":
    section_header("📊 Correlation Heatmap",
                   "Pairwise Pearson correlations between key economic variables")

    cols_of_interest = {
        "Export Price (USD/kg)":  "Export Price\n(USD/kg)",
        "Export Quantity (kg)":   "Export Qty\n(kg)",
        "Exchange Rate (LKR/USD)":"Exchange Rate\n(LKR/USD)",
        "Global_Tea_Price ($)":   "Global Tea\nPrice ($)",
        "Export Price (LKR/kg)":  "Export Price\n(LKR/kg)",
    }

    corr_df = df[list(cols_of_interest.keys())].corr()
    corr_df.index   = list(cols_of_interest.values())
    corr_df.columns = list(cols_of_interest.values())

    # Annotated heatmap
    import plotly.figure_factory as ff

    labels = list(cols_of_interest.values())
    z      = corr_df.values

    text_vals = [[f"{v:.2f}" for v in row] for row in z]

    fig = ff.create_annotated_heatmap(
        z=z,
        x=labels,
        y=labels,
        annotation_text=text_vals,
        colorscale="RdBu",
        reversescale=True,
        zmin=-1, zmax=1,
        showscale=True,
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Pearson Correlation Matrix",
        height=500,
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="info-box">'
        '<b>Interpretation Guide:</b> Values close to <b>+1</b> indicate strong positive correlation; '
        'values close to <b>−1</b> indicate strong negative correlation; '
        'values near <b>0</b> suggest little linear relationship. '
        'The LKR-denominated price and USD-denominated price show high correlation mediated '
        'by the exchange rate.</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# 6.  VOLATILITY ANALYSIS
# ══════════════════════════════════════════════
elif section == "🔍 Volatility Analysis":
    section_header("🔍 Volatility Analysis",
                   "Log returns and annualised rolling volatility of Sri Lanka export price")

    # Log returns
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df.index,
        y=df["Log_Return"],
        name="Monthly Log Return",
        marker_color=np.where(df["Log_Return"] >= 0, C_SL, "#e74c3c"),
        hovertemplate="%{x|%b %Y}: %{y:.4f}<extra>Log Return</extra>",
    ))
    fig1.add_hline(y=0, line_width=1.2, line_color="#1a3a5c", line_dash="dash")
    fig1.update_layout(
        **PLOT_LAYOUT,
        title="Monthly Log Returns – Export Price (USD/kg)",
        yaxis_title="Log Return",
        xaxis_title="Date",
        height=330,
    )
    fig1.update_yaxes(gridcolor="#e8edf3")
    fig1.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig1, use_container_width=True)

    # Rolling volatility
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["Rolling_Vol_6M"],
        name="6-Month Rolling Volatility",
        line=dict(color=C_VOL6, width=2),
        hovertemplate="%{x|%b %Y}: %{y:.4f}<extra>6M Vol</extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["Rolling_Vol_12M"],
        name="12-Month Rolling Volatility",
        line=dict(color=C_VOL12, width=2.5, dash="dot"),
        hovertemplate="%{x|%b %Y}: %{y:.4f}<extra>12M Vol</extra>",
    ))
    fig2.update_layout(
        **PLOT_LAYOUT,
        title="Annualised Rolling Volatility of Export Price",
        yaxis_title="Annualised Volatility (std of log returns × √12)",
        xaxis_title="Date",
        height=360,
    )
    fig2.update_yaxes(gridcolor="#e8edf3")
    fig2.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig2, use_container_width=True)

    vol_max_6  = df["Rolling_Vol_6M"].idxmax()
    vol_max_12 = df["Rolling_Vol_12M"].idxmax()
    st.markdown(
        f'<div class="info-box">'
        f'Peak 6-month volatility: <b>{df["Rolling_Vol_6M"].max():.4f}</b> '
        f'({vol_max_6.strftime("%b %Y")})<br>'
        f'Peak 12-month volatility: <b>{df["Rolling_Vol_12M"].max():.4f}</b> '
        f'({vol_max_12.strftime("%b %Y")})'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# 7.  ARIMA MODEL RESULTS
# ══════════════════════════════════════════════
elif section == "🤖 ARIMA Model":
    section_header("🤖 ARIMA Model Results",
                   "ARIMA(1,0,0) — autoregressive model fitted to monthly export price (USD/kg)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Model Specification**")
        spec_data = {
            "Parameter": ["Model Order", "AR", "Integration", "MA",
                           "AIC", "BIC", "Log-Likelihood", "Observations"],
            "Value":     ["ARIMA(1, 0, 0)", "1", "0", "0",
                          f"{result.aic:.4f}", f"{result.bic:.4f}",
                          f"{result.llf:.4f}", str(int(result.nobs))],
        }
        st.dataframe(pd.DataFrame(spec_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Coefficient Table**")
        params   = result.params
        se       = result.bse
        tvalues  = result.tvalues
        pvalues  = result.pvalues
        ci       = result.conf_int()

        coef_df = pd.DataFrame({
            "Coefficient": params,
            "Std Error":   se,
            "t-Statistic": tvalues,
            "p-Value":     pvalues,
            "CI Lower (95%)": ci.iloc[:, 0],
            "CI Upper (95%)": ci.iloc[:, 1],
        })
        coef_df.index.name = "Term"
        st.dataframe(
            coef_df.style.format({
                "Coefficient":    "{:.6f}",
                "Std Error":      "{:.6f}",
                "t-Statistic":    "{:.4f}",
                "p-Value":        "{:.4f}",
                "CI Lower (95%)": "{:.6f}",
                "CI Upper (95%)": "{:.6f}",
            }).applymap(
                lambda v: "background-color: #d4edda;" if isinstance(v, float) and v < 0.05 else "",
                subset=["p-Value"],
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # Full model summary as pre-formatted text
    st.markdown("**Full Model Summary**")
    summary_text = result.summary().as_text()
    st.code(summary_text, language="text")

    st.markdown(
        '<div class="info-box">'
        '<b>Model Rationale:</b> ARIMA(1,0,0) is an AR(1) process, implying that each month\'s '
        'price depends linearly on the previous month\'s price plus a white-noise error term. '
        'The integration order of 0 indicates the series is treated as stationary (or '
        'pre-differenced). The AR(1) coefficient captures price persistence (momentum), '
        'which is characteristic of commodity markets.'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# 8.  FORECAST
# ══════════════════════════════════════════════
elif section == "📉 Forecast":
    section_header("📉 12-Month Price Forecast",
                   "ARIMA(1,0,0) out-of-sample forecast with 95% confidence interval")

    # Number of historical months to show alongside forecast
    history_months = st.slider(
        "Historical months to display", min_value=12, max_value=84,
        value=36, step=6,
        help="Select how many months of history to show alongside the forecast."
    )

    hist_slice = price.iloc[-history_months:]

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_slice.index,
        y=hist_slice.values,
        name="Historical Export Price",
        line=dict(color=C_SL, width=2.2),
        hovertemplate="%{x|%b %Y}: $%{y:.3f}<extra>Historical</extra>",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc_mean.index,
        y=fc_mean.values,
        name="12-Month Forecast",
        line=dict(color=C_FORE, width=2.5, dash="dash"),
        hovertemplate="%{x|%b %Y}: $%{y:.3f}<extra>Forecast</extra>",
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_ci.iloc[:, 0], fc_ci.iloc[::-1, 1]]).index,
        y=pd.concat([fc_ci.iloc[:, 0], fc_ci.iloc[::-1, 1]]).values,
        fill="toself",
        fillcolor=C_CI,
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval",
        hoverinfo="skip",
    ))

    # Vertical line at forecast start
    fig.add_vline(
        x=fc_mean.index[0].timestamp() * 1000,
        line_width=1.5, line_dash="dot", line_color="#8e44ad",
        annotation_text="Forecast begins",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#8e44ad",
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        title=f"ARIMA(1,0,0) — 12-Month Export Price Forecast",
        yaxis_title="Export Price (USD/kg)",
        xaxis_title="Date",
        height=450,
    )
    fig.update_yaxes(gridcolor="#e8edf3")
    fig.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.markdown("**Forecast Values**")
    fc_table = pd.DataFrame({
        "Month":             fc_mean.index.strftime("%B %Y"),
        "Forecast (USD/kg)": fc_mean.values,
        "Lower CI (95%)":    fc_ci.iloc[:, 0].values,
        "Upper CI (95%)":    fc_ci.iloc[:, 1].values,
    })
    st.dataframe(
        fc_table.style.format({
            "Forecast (USD/kg)": "${:.4f}",
            "Lower CI (95%)":    "${:.4f}",
            "Upper CI (95%)":    "${:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════
# 9.  MODEL VALIDATION
# ══════════════════════════════════════════════
elif section == "✅ Model Validation":
    section_header("✅ Model Validation",
                   "In-sample performance metrics and residual diagnostics")

    # Metric cards
    c1, c2, c3 = st.columns(3)
    for col, label, val, unit, note in [
        (c1, "MAE",  f"{mae:.4f}",  "USD/kg", "Mean Absolute Error"),
        (c2, "RMSE", f"{rmse:.4f}", "USD/kg", "Root Mean Squared Error"),
        (c3, "MAPE", f"{mape:.2f}%","",        "Mean Absolute Percentage Error"),
    ]:
        col.markdown(
            f"""<div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div style='font-size:0.75rem;color:#4a6080;margin-top:0.15rem;'>{note}{' · '+unit if unit else ''}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # MAPE note
    st.markdown(
        '<div class="warn-box">'
        '<b>⚠ Note on MAPE:</b> MAPE is computed here on the level price series (USD/kg). '
        'When applied to log-return series (where values may approach zero), MAPE can become '
        'arbitrarily large or undefined. For return-based models, alternative metrics such as '
        'Mean Directional Accuracy (MDA) or Theil\'s U-statistic are more appropriate. '
        'The MAPE reported above is therefore interpreted with caution and supplemented by '
        'MAE and RMSE, which are not sensitive to near-zero denominators.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Actual vs Fitted
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=actual.index, y=actual.values,
        name="Actual Price",
        line=dict(color=C_SL, width=2),
        hovertemplate="%{x|%b %Y}: $%{y:.3f}<extra>Actual</extra>",
    ))
    fig1.add_trace(go.Scatter(
        x=fitted.index, y=fitted.values,
        name="ARIMA Fitted Values",
        line=dict(color=C_FORE, width=2, dash="dot"),
        hovertemplate="%{x|%b %Y}: $%{y:.3f}<extra>Fitted</extra>",
    ))
    fig1.update_layout(
        **PLOT_LAYOUT,
        title="Actual vs ARIMA Fitted Values",
        yaxis_title="Export Price (USD/kg)",
        xaxis_title="Date",
        height=360,
    )
    fig1.update_yaxes(gridcolor="#e8edf3")
    fig1.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig1, use_container_width=True)

    # Residuals
    residuals = actual - fitted
    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=("Residuals over Time", "Residual Distribution"))

    fig2.add_trace(go.Scatter(
        x=residuals.index, y=residuals.values,
        mode="lines", name="Residual",
        line=dict(color="#2e86de", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:.4f}<extra>Residual</extra>",
    ), row=1, col=1)
    fig2.add_hline(y=0, line_width=1, line_dash="dash", line_color="#e74c3c", row=1, col=1)

    fig2.add_trace(go.Histogram(
        x=residuals.values, nbinsx=28, name="Frequency",
        marker_color=C_SL, opacity=0.75,
    ), row=1, col=2)

    fig2.update_layout(
        **PLOT_LAYOUT,
        title="Residual Diagnostics",
        showlegend=False,
        height=340,
    )
    fig2.update_yaxes(gridcolor="#e8edf3")
    fig2.update_xaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="info-box">'
        '<b>Residual Interpretation:</b> Well-specified models should produce residuals '
        'centred near zero with no visible autocorrelation pattern over time, and an '
        'approximately normal distribution (bell-curve histogram). Systematic patterns '
        'in residuals suggest the model may be missing structural breaks, seasonal effects, '
        'or non-linear dynamics in the data.'
        '</div>',
        unsafe_allow_html=True,
    )
