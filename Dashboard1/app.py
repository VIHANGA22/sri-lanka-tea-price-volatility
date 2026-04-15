"""
Sri Lankan Tea Export Price Volatility Dashboard
Decision-Support Analytics Tool | 2008–2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tea Export Analytics | Sri Lanka",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# DESIGN TOKENS
# ──────────────────────────────────────────────
NAVY       = "#0D2137"
BLUE       = "#1B4F72"
MID_BLUE   = "#2E6DA4"
ACCENT     = "#C0392B"
GREY_DARK  = "#2C3E50"
GREY_MID   = "#6B7A8D"
GREY_LIGHT = "#F4F6F8"
WHITE      = "#FFFFFF"
CI_COLOR   = "rgba(46,109,164,0.15)"
GRID_COLOR = "#E8ECF0"
FONT       = "IBM Plex Sans, sans-serif"

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'IBM Plex Sans', sans-serif;
      background-color: {GREY_LIGHT};
      color: {GREY_DARK};
  }}

  /* Remove Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 2rem 3rem 3rem; max-width: 1400px; }}

  /* Masthead */
  .masthead {{
      background: {NAVY};
      border-radius: 4px;
      padding: 1.6rem 2.2rem;
      margin-bottom: 1.8rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
  }}
  .masthead-title {{
      color: {WHITE};
      font-size: 1.15rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      margin: 0;
  }}
  .masthead-sub {{
      color: {GREY_MID};
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-top: 0.2rem;
  }}
  .masthead-badge {{
      background: {BLUE};
      color: {WHITE};
      font-size: 0.7rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      padding: 0.3rem 0.8rem;
      border-radius: 2px;
      font-weight: 500;
  }}

  /* Section headers */
  .section-label {{
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: {GREY_MID};
      margin: 0 0 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid {GRID_COLOR};
  }}

  /* KPI Cards */
  .kpi-card {{
      background: {WHITE};
      border: 1px solid {GRID_COLOR};
      border-radius: 4px;
      padding: 1.2rem 1.4rem;
      height: 100%;
  }}
  .kpi-label {{
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {GREY_MID};
      margin-bottom: 0.5rem;
  }}
  .kpi-value {{
      font-size: 1.8rem;
      font-weight: 600;
      color: {NAVY};
      line-height: 1;
      font-family: 'IBM Plex Mono', monospace;
  }}
  .kpi-delta-pos {{
      font-size: 0.78rem;
      color: #27AE60;
      margin-top: 0.4rem;
      font-weight: 500;
  }}
  .kpi-delta-neg {{
      font-size: 0.78rem;
      color: {ACCENT};
      margin-top: 0.4rem;
      font-weight: 500;
  }}
  .kpi-direction {{
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      color: {MID_BLUE};
      margin-top: 0.4rem;
  }}

  /* Insight bar */
  .insight-bar {{
      background: {WHITE};
      border-left: 3px solid {MID_BLUE};
      border-radius: 0 4px 4px 0;
      padding: 0.9rem 1.2rem;
      margin-bottom: 1.8rem;
      font-size: 0.84rem;
      color: {GREY_DARK};
      line-height: 1.5;
  }}

  /* Metric cards (performance) */
  .metric-box {{
      background: {WHITE};
      border: 1px solid {GRID_COLOR};
      border-radius: 4px;
      padding: 1rem 1.2rem;
      text-align: center;
  }}
  .metric-box-label {{
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: {GREY_MID};
      margin-bottom: 0.4rem;
  }}
  .metric-box-value {{
      font-size: 1.4rem;
      font-weight: 600;
      color: {NAVY};
      font-family: 'IBM Plex Mono', monospace;
  }}
  .metric-note {{
      font-size: 0.72rem;
      color: {GREY_MID};
      margin-top: 0.5rem;
      font-style: italic;
  }}

  /* Decision panel */
  .decision-card {{
      background: {WHITE};
      border: 1px solid {GRID_COLOR};
      border-radius: 4px;
      padding: 1rem 1.3rem;
      margin-bottom: 0.6rem;
  }}
  .decision-tag {{
      display: inline-block;
      font-size: 0.6rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 0.15rem 0.5rem;
      border-radius: 2px;
      margin-bottom: 0.4rem;
  }}
  .tag-price {{ background: #EBF5FB; color: {MID_BLUE}; }}
  .tag-fx    {{ background: #FDFEFE; color: #7D6608; border: 1px solid #F9E79F; }}
  .tag-vol   {{ background: #FDEDEC; color: {ACCENT}; }}
  .tag-qty   {{ background: #EAFAF1; color: #27AE60; }}
  .decision-text {{
      font-size: 0.82rem;
      color: {GREY_DARK};
      line-height: 1.4;
      margin: 0;
  }}

  /* Divider */
  hr.section-divider {{
      border: none;
      border-top: 1px solid {GRID_COLOR};
      margin: 1.8rem 0;
  }}

  /* Chart container */
  .chart-container {{
      background: {WHITE};
      border: 1px solid {GRID_COLOR};
      border-radius: 4px;
      padding: 0.5rem;
  }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING & ARIMA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Merged_Tea_Dataset.csv", index_col=0, parse_dates=True)
    df = df.sort_index()
    numeric_cols = [
        "Export Quantity (kg)", "Export Price (LKR/kg)",
        "Exchange Rate (LKR/USD)", "Export Price (USD/kg)",
        "Global_Tea_Price ($)", "Log_Return"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df["Roll_Vol_6M"]  = df["Log_Return"].rolling(6).std()
    df["Roll_Vol_12M"] = df["Log_Return"].rolling(12).std()
    return df


@st.cache_data
def run_arima(df):
    returns = df["Log_Return"].dropna()

    # Train/test split for metrics
    train_size = int(len(returns) * 0.8)
    train = returns[:train_size]
    test  = returns[train_size:]

    model_val = ARIMA(train, order=(1, 0, 0)).fit()
    pred_test = model_val.get_forecast(steps=len(test)).predicted_mean

    mae  = mean_absolute_error(test, pred_test)
    rmse = np.sqrt(mean_squared_error(test, pred_test))
    # Guard against near-zero returns in MAPE denominator
    nonzero = test[test.abs() > 1e-6]
    pred_nz = pred_test[nonzero.index]
    mape = np.mean(np.abs((nonzero - pred_nz) / nonzero)) * 100

    # Full model for forecasting
    full_model    = ARIMA(returns, order=(1, 0, 0)).fit()
    forecast_obj  = full_model.get_forecast(steps=12)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci   = forecast_obj.conf_int()

    # Convert log-return forecast → price levels
    last_price  = df["Export Price (USD/kg)"].iloc[-1]
    last_date   = df.index[-1]
    fc_dates    = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    forecast_mean.index = fc_dates
    forecast_ci.index   = fc_dates

    price_fc, price_lo, price_hi = [], [], []
    prev = last_price
    for i in range(12):
        p    = prev * np.exp(forecast_mean.iloc[i])
        lo_r = forecast_ci.iloc[i, 0]
        hi_r = forecast_ci.iloc[i, 1]
        lo   = prev * np.exp(lo_r)
        hi   = prev * np.exp(hi_r)
        price_fc.append(p)
        price_lo.append(lo)
        price_hi.append(hi)
        prev = p

    fc_df = pd.DataFrame({
        "forecast":  price_fc,
        "lower":     price_lo,
        "upper":     price_hi,
    }, index=fc_dates)

    direction = "Moderate Increase" if price_fc[-1] > last_price * 1.02 else \
                "Slight Decline"     if price_fc[-1] < last_price * 0.98 else "Stable"

    return fc_df, mae, rmse, mape, direction


# ──────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────
LAYOUT_BASE = dict(
    font=dict(family=FONT, color=GREY_DARK, size=12),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor=GRID_COLOR, showline=False, tickfont=dict(size=11)),
    yaxis=dict(gridcolor=GRID_COLOR, showline=False, tickfont=dict(size=11)),
    hovermode="x unified",
)


def chart_price_vs_global(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Export Price (USD/kg)"],
        name="Sri Lanka Export Price",
        line=dict(color=NAVY, width=2),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Global_Tea_Price ($)"],
        name="Global Benchmark Price",
        line=dict(color=MID_BLUE, width=1.5, dash="dot"),
        mode="lines",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        yaxis_title="USD / kg",
        title=dict(text="Sri Lanka vs Global Tea Prices", font=dict(size=13, color=NAVY), x=0),
    )
    return fig


def chart_exchange_rate(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Exchange Rate (LKR/USD)"],
        name="Exchange Rate",
        line=dict(color=MID_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(46,109,164,0.07)",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        yaxis_title="LKR / USD",
        title=dict(text="LKR / USD Exchange Rate", font=dict(size=13, color=NAVY), x=0),
    )
    return fig


def chart_fx_scatter(df):
    fig = px.scatter(
        df, x="Exchange Rate (LKR/USD)", y="Export Price (USD/kg)",
        opacity=0.55,
        trendline="ols",
        trendline_color_override=ACCENT,
    )
    fig.update_traces(marker=dict(color=MID_BLUE, size=5))
    fig.update_layout(
        **LAYOUT_BASE,
        xaxis_title="Exchange Rate (LKR/USD)",
        yaxis_title="Export Price (USD/kg)",
        title=dict(text="Exchange Rate vs Export Price", font=dict(size=13, color=NAVY), x=0),
        showlegend=False,
    )
    return fig


def chart_volatility(df):
    fig = go.Figure()
    # Highlight spikes > 2 std dev
    vol_mean = df["Roll_Vol_6M"].mean()
    vol_std  = df["Roll_Vol_6M"].std()
    spike_mask = df["Roll_Vol_6M"] > (vol_mean + 2 * vol_std)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Roll_Vol_12M"],
        name="12-Month Rolling Volatility",
        line=dict(color=GREY_MID, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Roll_Vol_6M"],
        name="6-Month Rolling Volatility",
        line=dict(color=NAVY, width=2),
    ))
    # Spike markers
    if spike_mask.any():
        fig.add_trace(go.Scatter(
            x=df.index[spike_mask],
            y=df["Roll_Vol_6M"][spike_mask],
            name="Volatility Spike",
            mode="markers",
            marker=dict(color=ACCENT, size=7, symbol="circle"),
        ))
    fig.update_layout(
        **LAYOUT_BASE,
        yaxis_title="Standard Deviation (Log Returns)",
        title=dict(text="Price Volatility — 6M & 12M Rolling Window", font=dict(size=13, color=NAVY), x=0),
    )
    return fig


def chart_forecast(df, fc_df):
    hist_tail = df["Export Price (USD/kg)"].iloc[-36:]
    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=fc_df.index.tolist() + fc_df.index.tolist()[::-1],
        y=fc_df["upper"].tolist() + fc_df["lower"].tolist()[::-1],
        fill="toself",
        fillcolor=CI_COLOR,
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval",
        showlegend=True,
        hoverinfo="skip",
    ))
    # Historical
    fig.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values,
        name="Historical (last 36M)",
        line=dict(color=NAVY, width=2),
        mode="lines",
    ))
    # Bridge dot
    bridge_x = [df.index[-1], fc_df.index[0]]
    bridge_y = [df["Export Price (USD/kg)"].iloc[-1], fc_df["forecast"].iloc[0]]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        line=dict(color=ACCENT, width=1.5, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df["forecast"],
        name="12-Month Forecast",
        line=dict(color=ACCENT, width=2.5),
        mode="lines+markers",
        marker=dict(size=5, color=ACCENT),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        yaxis_title="Export Price (USD/kg)",
        title=dict(text="ARIMA(1,0,0) — 12-Month Price Forecast", font=dict(size=13, color=NAVY), x=0),
    )
    return fig


# ──────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────
df = load_data()
fc_df, mae, rmse, mape, direction = run_arima(df)

latest_price    = df["Export Price (USD/kg)"].iloc[-1]
prev_price      = df["Export Price (USD/kg)"].iloc[-2]
price_delta     = latest_price - prev_price
latest_fx       = df["Exchange Rate (LKR/USD)"].iloc[-1]
prev_fx         = df["Exchange Rate (LKR/USD)"].iloc[-2]
fx_delta        = latest_fx - prev_fx
latest_global   = df["Global_Tea_Price ($)"].iloc[-1]
prev_global     = df["Global_Tea_Price ($)"].iloc[-2]
global_delta    = latest_global - prev_global
latest_date_str = df.index[-1].strftime("%B %Y")

# ── MASTHEAD ──────────────────────────────────
st.markdown(f"""
<div class="masthead">
  <div>
    <div class="masthead-title">Sri Lanka Tea Export Price Volatility</div>
    <div class="masthead-sub">Decision-Support Analytics · 2008 – 2024 · ARIMA Time Series</div>
  </div>
  <div class="masthead-badge">Latest Data: {latest_date_str}</div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────
st.markdown('<div class="section-label">Executive Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4, gap="small")

def _delta_html(val, fmt="{:+.3f}", unit=""):
    cls = "kpi-delta-pos" if val >= 0 else "kpi-delta-neg"
    arrow = "▲" if val >= 0 else "▼"
    return f'<div class="{cls}">{arrow} {fmt.format(abs(val))}{unit} vs prior month</div>'

with k1:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Export Price (USD/kg)</div>
      <div class="kpi-value">${latest_price:.3f}</div>
      {_delta_html(price_delta, "{:+.3f}", "")}
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Exchange Rate (LKR/USD)</div>
      <div class="kpi-value">{latest_fx:.1f}</div>
      {_delta_html(fx_delta, "{:+.1f}", "")}
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Global Benchmark Price ($)</div>
      <div class="kpi-value">${latest_global:.3f}</div>
      {_delta_html(global_delta, "{:+.3f}", "")}
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">12-Month Forecast Direction</div>
      <div class="kpi-value" style="font-size:1.3rem;">{direction}</div>
      <div class="kpi-direction">ARIMA(1,0,0) · 95% CI</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── INSIGHT BAR ───────────────────────────────
st.markdown(f"""
<div class="insight-bar">
  <strong>Market Insight &nbsp;·</strong>&nbsp;
  Sri Lanka export prices have demonstrated sustained divergence from global benchmarks since 2020,
  driven primarily by LKR depreciation pressure. Current forward trajectory indicates <strong>{direction.lower()}</strong>
  over the next 12 months based on ARIMA(1,0,0) modelling of log-return dynamics.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── PRICE & MARKET TRENDS ─────────────────────
st.markdown('<div class="section-label">Price &amp; Market Trends</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(chart_price_vs_global(df), use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── EXCHANGE RATE IMPACT ──────────────────────
st.markdown('<div class="section-label">Exchange Rate Impact</div>', unsafe_allow_html=True)
c_fx1, c_fx2 = st.columns([1.4, 1], gap="medium")
with c_fx1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(chart_exchange_rate(df), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
with c_fx2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(chart_fx_scatter(df), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── VOLATILITY & RISK ─────────────────────────
st.markdown('<div class="section-label">Volatility &amp; Risk</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(chart_volatility(df), use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── FORECASTING ───────────────────────────────
st.markdown('<div class="section-label">12-Month Price Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(chart_forecast(df, fc_df), use_container_width=True, config={"displayModeBar": False})
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── MODEL PERFORMANCE + DECISIONS ────────────
st.markdown('<div class="section-label">Model Performance &amp; Decision Insights</div>', unsafe_allow_html=True)
c_perf, c_gap, c_dec = st.columns([1, 0.08, 1.6], gap="small")

with c_perf:
    pm1, pm2, pm3 = st.columns(3)
    with pm1:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-box-label">MAE</div>
          <div class="metric-box-value">{mae:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with pm2:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-box-label">RMSE</div>
          <div class="metric-box-value">{rmse:.4f}</div>
        </div>""", unsafe_allow_html=True)
    with pm3:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-box-label">MAPE</div>
          <div class="metric-box-value">{mape:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <p class="metric-note">
      Model: ARIMA(1,0,0) on log returns · 80/20 train-test split.<br>
      MAPE may be elevated due to near-zero return values near inflection points.
      MAE and RMSE are the primary reliability indicators.
    </p>""", unsafe_allow_html=True)

with c_dec:
    insights = [
        ("price",  "tag-price", "Price Outlook",
         f"Prices expected to {direction.lower()} over the next 12 months. "
         f"Forecast terminal value: <strong>${fc_df['forecast'].iloc[-1]:.3f}/kg</strong> (95% CI: "
         f"${fc_df['lower'].iloc[-1]:.3f} – ${fc_df['upper'].iloc[-1]:.3f})."),
        ("fx",     "tag-fx",    "Exchange Rate Driver",
         "LKR/USD exchange rate is the primary structural driver of export price levels. "
         "Currency depreciation post-2020 has significantly amplified USD-denominated export returns."),
        ("vol",    "tag-vol",   "Volatility Regime",
         "Rolling volatility increased materially post-2020, consistent with macro disruptions "
         "(pandemic, currency crisis). This structural shift elevates forecast uncertainty and hedging requirements."),
        ("qty",    "tag-qty",   "Quantity Signal",
         "Export quantity shows weak correlation with price movements. Volume adjustments alone "
         "are insufficient as a price-support mechanism; quality and destination-market mix are more decisive."),
    ]
    for _, tag_cls, title, text in insights:
        st.markdown(f"""
        <div class="decision-card">
          <span class="decision-tag {tag_cls}">{title}</span>
          <p class="decision-text">{text}</p>
        </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="font-size:0.68rem; color:{GREY_MID}; text-align:center; padding:1rem 0; border-top:1px solid {GRID_COLOR};">
  Sri Lanka Tea Export Price Volatility Research &nbsp;·&nbsp; Data: 2008 – 2024 &nbsp;·&nbsp;
  ARIMA(1,0,0) Time Series Model &nbsp;·&nbsp; For analytical use only. Not financial advice.
</div>""", unsafe_allow_html=True)
