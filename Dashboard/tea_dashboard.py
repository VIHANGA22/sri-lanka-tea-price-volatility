"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Sri Lanka Tea Price Analysis Dashboard  —  Streamlit + Plotly       ║
║         Research: "Analysis and Forecasting of Sri Lankan Tea Export Prices" ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:  streamlit run tea_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Tea Price Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (dark, refined academic palette)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2a3347;
}
[data-testid="stSidebar"] * {
    color: #c8d4e8 !important;
}

/* ── Header banner ── */
.hero {
    background: linear-gradient(135deg, #0d2137 0%, #143352 50%, #1a4a3f 100%);
    border: 1px solid #2a4a6a;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(74,180,120,0.15) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #f0f6ff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: #8fa8c8;
    margin: 0;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-block;
    background: rgba(74,180,120,0.2);
    border: 1px solid rgba(74,180,120,0.4);
    color: #4ab478;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #161b27;
    border: 1px solid #2a3347;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 10px 10px;
}
.metric-card.green::after  { background: #4ab478; }
.metric-card.blue::after   { background: #4a90d9; }
.metric-card.amber::after  { background: #e8a838; }
.metric-card.rose::after   { background: #e86878; }
.metric-label {
    font-size: 0.72rem;
    color: #607090;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: #f0f6ff;
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: #607090;
    margin-top: 0.3rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #c8d8f0;
    margin: 0.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a3347;
}

/* ── Chart containers ── */
.chart-card {
    background: #161b27;
    border: 1px solid #2a3347;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #607090;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4ab478;
    border-bottom-color: #4ab478;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2a3347; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME  (shared config used on every figure)
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#a0b0c8", size=11),
    title_font=dict(family="Playfair Display, serif", color="#c8d8f0", size=14),
    legend=dict(
        bgcolor="rgba(22,27,39,0.8)",
        bordercolor="#2a3347",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="#1e2535",
        zerolinecolor="#2a3347",
        tickfont=dict(size=10),
    ),
    yaxis=dict(
        gridcolor="#1e2535",
        zerolinecolor="#2a3347",
        tickfont=dict(size=10),
    ),
    margin=dict(l=40, r=20, t=50, b=40),
    hovermode="x unified",
)

COLORS = {
    "sri_lanka": "#4ab478",   # green  — SL price
    "global":    "#4a90d9",   # blue   — global price
    "exchange":  "#e8a838",   # amber  — exchange rate
    "quantity":  "#a07ad9",   # violet — quantity
    "forecast":  "#e86878",   # rose   — forecast
    "ci":        "rgba(232,104,120,0.15)",
    "log_ret":   "#60c8c0",   # teal   — log returns
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    """Load dataset from uploaded file or fallback path."""
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
    else:
        # Try common default paths
        for path in [
            "Cleaned_Merged_Tea_Dataset.xlsx",
            "Cleaned_Merged_Tea_Dataset__1_.xlsx",
        ]:
            try:
                df = pd.read_excel(path)
                break
            except FileNotFoundError:
                continue
        else:
            st.error("❌ Please upload your dataset using the sidebar uploader.")
            st.stop()

    # ── Normalise column names ────────────────────────────────────────────
    df.columns = df.columns.str.strip()

    # Date column (first column is Unnamed: 0 = datetime index)
    date_col = df.columns[0]
    df["Date"] = pd.to_datetime(df[date_col])

    # Rename for convenience
    df = df.rename(columns={
        "Export Quantity (kg)":      "Quantity",
        "Export Price (LKR/kg)":     "Price_LKR",
        "Exchange Rate (LKR/USD)":   "ExchangeRate",
        "Export Price (USD/kg)":     "Price_USD",
        "Global_Tea_Price ($)":      "Global_Price",
        "Log_Return":                "Log_Return",
    })

    df = df.sort_values("Date").reset_index(drop=True)
    return df


@st.cache_data
def run_arima(series: pd.Series, order=(1, 1, 1), steps=12):
    """Fit ARIMA and return forecast + 95% CI."""
    model = ARIMA(series.dropna(), order=order)
    result = model.fit()
    forecast = result.get_forecast(steps=steps)
    mean   = forecast.predicted_mean
    ci     = forecast.conf_int(alpha=0.05)
    return mean, ci, result


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍃 Tea Dashboard")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Dataset (.xlsx)",
        type=["xlsx", "csv"],
        help="Upload your cleaned tea export dataset",
    )

    st.markdown("---")
    st.markdown("**Filters**")

    df_raw = load_data(uploaded)

    year_min, year_max = int(df_raw["Year"].min()), int(df_raw["Year"].max())
    year_range = st.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    MONTHS = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    ]
    month_sel = st.multiselect(
        "Months (leave blank = all)",
        options=MONTHS,
        default=[],
        help="Filter by specific months",
    )

    st.markdown("---")
    st.markdown("**ARIMA Settings**")
    p = st.selectbox("p (AR order)",  [0,1,2,3], index=1)
    d = st.selectbox("d (Integration)",[0,1,2],  index=1)
    q = st.selectbox("q (MA order)",  [0,1,2,3], index=1)
    horizon = st.slider("Forecast horizon (months)", 6, 24, 12)

    st.markdown("---")
    st.caption("Research Project · University of Colombo")

# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
mask = (df_raw["Year"] >= year_range[0]) & (df_raw["Year"] <= year_range[1])
if month_sel:
    mask &= df_raw["Month"].isin(month_sel)
df = df_raw[mask].copy()

# ─────────────────────────────────────────────────────────────────────────────
# ARIMA FORECAST
# ─────────────────────────────────────────────────────────────────────────────
fc_mean, fc_ci, arima_result = run_arima(
    df_raw["Price_USD"],   # always fit on full series
    order=(p, d, q),
    steps=horizon,
)
last_date = df_raw["Date"].iloc[-1]
fc_dates  = pd.date_range(start=last_date + pd.DateOffset(months=1),
                          periods=horizon, freq="MS")

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">Research Dashboard · 2008 – 2024</div>
  <h1 class="hero-title">🍃 Sri Lanka Tea Price Analysis</h1>
  <p class="hero-subtitle">
    Time-series analysis and ARIMA forecasting of Sri Lankan tea export prices &mdash;
    {year_range[0]} to {year_range[1]} &nbsp;|&nbsp; {len(df):,} monthly observations
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
latest_price   = df["Price_USD"].iloc[-1]
avg_price      = df["Price_USD"].mean()
volatility     = df["Log_Return"].std() * np.sqrt(12) * 100   # annualised %
latest_fx      = df["ExchangeRate"].iloc[-1]
next_fc        = fc_mean.iloc[0]
avg_qty_m      = df["Quantity"].mean() / 1e6

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card green">
    <div class="metric-label">Latest Export Price</div>
    <div class="metric-value">${latest_price:.2f}</div>
    <div class="metric-sub">USD / kg &nbsp;·&nbsp; {df["Date"].iloc[-1].strftime("%b %Y")}</div>
  </div>
  <div class="metric-card blue">
    <div class="metric-label">Period Average Price</div>
    <div class="metric-value">${avg_price:.2f}</div>
    <div class="metric-sub">USD / kg &nbsp;·&nbsp; {year_range[0]}–{year_range[1]}</div>
  </div>
  <div class="metric-card amber">
    <div class="metric-label">Annualised Volatility</div>
    <div class="metric-value">{volatility:.1f}%</div>
    <div class="metric-sub">Based on log returns</div>
  </div>
  <div class="metric-card rose">
    <div class="metric-label">12-Month Forecast</div>
    <div class="metric-value">${next_fc:.2f}</div>
    <div class="metric-sub">ARIMA({p},{d},{q}) next month</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price Trends",
    "🌍 Global Comparison",
    "💱 Exchange Rate & Quantity",
    "🔮 ARIMA Forecast",
    "📊 Statistical Analysis",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Sri Lanka Export Price (USD/kg)</p>', unsafe_allow_html=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["Date"], y=df["Price_USD"],
        mode="lines",
        name="Export Price (USD/kg)",
        line=dict(color=COLORS["sri_lanka"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(74,180,120,0.08)",
        hovertemplate="%{x|%b %Y}<br><b>$%{y:.3f}</b><extra></extra>",
    ))
    # Annotate max & min
    idx_max = df["Price_USD"].idxmax()
    idx_min = df["Price_USD"].idxmin()
    for idx, label, ay in [(idx_max, "Peak", -30), (idx_min, "Trough", 30)]:
        fig1.add_annotation(
            x=df.loc[idx, "Date"], y=df.loc[idx, "Price_USD"],
            text=f"{label}: ${df.loc[idx,'Price_USD']:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="#607090",
            font=dict(size=10, color="#a0b0c8"),
            ay=ay, ax=0,
        )
    fig1.update_layout(**PLOT_LAYOUT, title="Monthly Export Price — USD per kg", height=360)
    st.plotly_chart(fig1, use_container_width=True)

    # LKR price
    st.markdown('<p class="section-header">Export Price (LKR/kg)</p>', unsafe_allow_html=True)
    fig1b = go.Figure()
    fig1b.add_trace(go.Scatter(
        x=df["Date"], y=df["Price_LKR"],
        mode="lines",
        name="Export Price (LKR/kg)",
        line=dict(color="#a07ad9", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(160,122,217,0.08)",
        hovertemplate="%{x|%b %Y}<br><b>LKR %{y:,.0f}</b><extra></extra>",
    ))
    fig1b.update_layout(**PLOT_LAYOUT, title="Monthly Export Price — LKR per kg", height=300)
    st.plotly_chart(fig1b, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GLOBAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Sri Lanka vs Global Tea Price</p>', unsafe_allow_html=True)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(
        x=df["Date"], y=df["Price_USD"],
        name="SL Export Price (USD/kg)",
        line=dict(color=COLORS["sri_lanka"], width=2.5),
        hovertemplate="%{x|%b %Y}<br>SL: $%{y:.3f}<extra></extra>",
    ), secondary_y=False)
    fig2.add_trace(go.Scatter(
        x=df["Date"], y=df["Global_Price"],
        name="Global Tea Price (USD/kg)",
        line=dict(color=COLORS["global"], width=2, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>Global: $%{y:.3f}<extra></extra>",
    ), secondary_y=True)
    fig2.update_layout(
        **PLOT_LAYOUT,
        title="Sri Lanka Export Price vs Global Tea Price",
        height=400,
    )
    fig2.update_yaxes(title_text="SL Price (USD/kg)", secondary_y=False, gridcolor="#1e2535")
    fig2.update_yaxes(title_text="Global Price (USD/kg)", secondary_y=True, gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation scatter
    st.markdown('<p class="section-header">Correlation: SL Price vs Global Price</p>', unsafe_allow_html=True)
    corr = df["Price_USD"].corr(df["Global_Price"])
    fig2b = px.scatter(
        df, x="Global_Price", y="Price_USD",
        color="Year",
        color_continuous_scale="Teal",
        trendline="ols",
        labels={"Global_Price": "Global Tea Price (USD/kg)", "Price_USD": "SL Export Price (USD/kg)"},
        hover_data={"Year": True, "Month": True},
        title=f"SL Price vs Global Price  |  Pearson r = {corr:.3f}",
    )
    fig2b.update_traces(marker=dict(size=6, opacity=0.75))
    fig2b.update_layout(**PLOT_LAYOUT, height=360)
    fig2b.update_layout(coloraxis_colorbar=dict(title="Year", tickfont=dict(size=10)))
    st.plotly_chart(fig2b, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXCHANGE RATE & QUANTITY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Exchange Rate (LKR/USD)</p>', unsafe_allow_html=True)
        fig3a = go.Figure()
        fig3a.add_trace(go.Scatter(
            x=df["Date"], y=df["ExchangeRate"],
            mode="lines",
            name="Exchange Rate",
            line=dict(color=COLORS["exchange"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(232,168,56,0.08)",
            hovertemplate="%{x|%b %Y}<br><b>LKR %{y:.2f}</b><extra></extra>",
        ))
        fig3a.update_layout(**PLOT_LAYOUT, title="LKR / USD Exchange Rate", height=340)
        st.plotly_chart(fig3a, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Export Quantity (kg)</p>', unsafe_allow_html=True)
        fig3b = go.Figure()
        fig3b.add_trace(go.Scatter(
            x=df["Date"], y=df["Quantity"] / 1e6,
            mode="lines",
            name="Export Quantity",
            line=dict(color=COLORS["quantity"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(160,122,217,0.08)",
            hovertemplate="%{x|%b %Y}<br><b>%{y:.1f}M kg</b><extra></extra>",
        ))
        fig3b.update_layout(**PLOT_LAYOUT, title="Monthly Export Quantity (Million kg)", height=340)
        st.plotly_chart(fig3b, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Scatter: Price vs Quantity & Exchange Rate</p>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        corr_pq = df["Price_USD"].corr(df["Quantity"])
        fig3c = px.scatter(
            df, x="Quantity", y="Price_USD",
            color="Year", color_continuous_scale="Teal",
            trendline="ols",
            labels={"Quantity": "Export Quantity (kg)", "Price_USD": "Price (USD/kg)"},
            title=f"Price vs Quantity  |  r = {corr_pq:.3f}",
        )
        fig3c.update_traces(marker=dict(size=5, opacity=0.7))
        fig3c.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig3c, use_container_width=True)

    with col_d:
        corr_pe = df["Price_USD"].corr(df["ExchangeRate"])
        fig3d = px.scatter(
            df, x="ExchangeRate", y="Price_USD",
            color="Year", color_continuous_scale="Sunset",
            trendline="ols",
            labels={"ExchangeRate": "Exchange Rate (LKR/USD)", "Price_USD": "Price (USD/kg)"},
            title=f"Price vs Exchange Rate  |  r = {corr_pe:.3f}",
        )
        fig3d.update_traces(marker=dict(size=5, opacity=0.7))
        fig3d.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig3d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARIMA FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<p class="section-header">ARIMA({p},{d},{q}) Forecast — Next {horizon} Months</p>',
                unsafe_allow_html=True)

    fig4 = go.Figure()

    # Historical
    fig4.add_trace(go.Scatter(
        x=df_raw["Date"], y=df_raw["Price_USD"],
        mode="lines",
        name="Historical Price",
        line=dict(color=COLORS["sri_lanka"], width=2.5),
        hovertemplate="%{x|%b %Y}<br>Historical: $%{y:.3f}<extra></extra>",
    ))

    # Confidence interval shading
    fig4.add_trace(go.Scatter(
        x=list(fc_dates) + list(fc_dates[::-1]),
        y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[::-1, 0]),
        fill="toself",
        fillcolor=COLORS["ci"],
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Forecast line
    fig4.add_trace(go.Scatter(
        x=fc_dates, y=fc_mean,
        mode="lines+markers",
        name=f"ARIMA({p},{d},{q}) Forecast",
        line=dict(color=COLORS["forecast"], width=2.5, dash="dash"),
        marker=dict(size=7, color=COLORS["forecast"]),
        hovertemplate="%{x|%b %Y}<br>Forecast: $%{y:.3f}<extra></extra>",
    ))

    # Vertical divider at forecast start
    fig4.add_vline(
        x=last_date.timestamp() * 1000,
        line=dict(color="#2a3347", width=1.5, dash="dot"),
        annotation_text="Forecast →",
        annotation_font=dict(color="#607090", size=10),
        annotation_position="top right",
    )

    fig4.update_layout(
        **PLOT_LAYOUT,
        title=f"Export Price (USD/kg) — Historical & ARIMA({p},{d},{q}) Forecast",
        height=460,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Forecast table
    st.markdown('<p class="section-header">Forecast Values</p>', unsafe_allow_html=True)
    fc_df = pd.DataFrame({
        "Month":               fc_dates.strftime("%b %Y"),
        "Forecast (USD/kg)":   fc_mean.round(4).values,
        "Lower CI (95%)":      fc_ci.iloc[:, 0].round(4).values,
        "Upper CI (95%)":      fc_ci.iloc[:, 1].round(4).values,
    })
    st.dataframe(
        fc_df.style
            .format({"Forecast (USD/kg)": "${:.4f}", "Lower CI (95%)": "${:.4f}", "Upper CI (95%)": "${:.4f}"})
            .highlight_max(subset=["Forecast (USD/kg)"], color="#1a4a3f")
            .highlight_min(subset=["Forecast (USD/kg)"], color="#3a1a27"),
        use_container_width=True,
        hide_index=True,
    )

    # ARIMA summary
    with st.expander("📋 ARIMA Model Summary"):
        st.text(str(arima_result.summary()))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<p class="section-header">Log Returns Distribution</p>', unsafe_allow_html=True)
        lr = df["Log_Return"].dropna()
        fig5a = go.Figure()
        fig5a.add_trace(go.Histogram(
            x=lr,
            nbinsx=40,
            name="Log Returns",
            marker_color=COLORS["log_ret"],
            opacity=0.75,
            hovertemplate="Range: %{x:.4f}<br>Count: %{y}<extra></extra>",
        ))
        # Normal overlay
        x_line = np.linspace(lr.min(), lr.max(), 200)
        from scipy.stats import norm
        pdf = norm.pdf(x_line, lr.mean(), lr.std()) * len(lr) * (lr.max()-lr.min()) / 40
        fig5a.add_trace(go.Scatter(
            x=x_line, y=pdf,
            mode="lines",
            name="Normal Fit",
            line=dict(color="#e8a838", width=2, dash="dash"),
        ))
        fig5a.update_layout(**PLOT_LAYOUT, title="Distribution of Log Returns", height=340,
                            xaxis_title="Log Return", yaxis_title="Count")
        st.plotly_chart(fig5a, use_container_width=True)

    with col_f:
        st.markdown('<p class="section-header">Log Returns Over Time</p>', unsafe_allow_html=True)
        fig5b = go.Figure()
        fig5b.add_trace(go.Scatter(
            x=df["Date"], y=df["Log_Return"],
            mode="lines",
            name="Log Return",
            line=dict(color=COLORS["log_ret"], width=1.5),
            hovertemplate="%{x|%b %Y}<br>%{y:.4f}<extra></extra>",
        ))
        fig5b.add_hline(y=0, line=dict(color="#2a3347", width=1))
        fig5b.update_layout(**PLOT_LAYOUT, title="Monthly Log Returns", height=340,
                            yaxis_title="Log Return")
        st.plotly_chart(fig5b, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Rolling Volatility (12-Month)</p>', unsafe_allow_html=True)
    df_raw["Rolling_Vol"] = df_raw["Log_Return"].rolling(12).std() * np.sqrt(12) * 100
    fig5c = go.Figure()
    fig5c.add_trace(go.Scatter(
        x=df_raw["Date"], y=df_raw["Rolling_Vol"],
        mode="lines",
        name="12M Rolling Volatility",
        line=dict(color=COLORS["forecast"], width=2),
        fill="tozeroy",
        fillcolor="rgba(232,104,120,0.08)",
        hovertemplate="%{x|%b %Y}<br><b>%{y:.1f}%</b><extra></extra>",
    ))
    fig5c.update_layout(**PLOT_LAYOUT, title="12-Month Rolling Annualised Volatility (%)", height=300,
                        yaxis_title="Volatility (%)")
    st.plotly_chart(fig5c, use_container_width=True)

    # Descriptive stats table
    st.markdown('<p class="section-header">Descriptive Statistics</p>', unsafe_allow_html=True)
    desc = df[["Price_USD","Price_LKR","ExchangeRate","Quantity","Global_Price","Log_Return"]].describe().T
    desc.columns = ["Count","Mean","Std","Min","25%","50%","75%","Max"]
    st.dataframe(desc.round(4).style.format("{:.4f}"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#405060; font-size:0.78rem; padding:0.5rem 0;'>"
    "Sri Lanka Tea Price Analysis Dashboard &nbsp;·&nbsp; "
    "Data: Sri Lanka Tea Board / Central Bank &nbsp;·&nbsp; "
    "Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
