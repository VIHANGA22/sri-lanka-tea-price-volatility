"""
Sri Lanka Tea Export Price Volatility Dashboard
ARIMA Time Series · 2008 – 2024
Decision-Support Analytics for Business Professionals & Policy Makers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Ceylon Tea Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "Cleaned_Merged_Tea_Dataset.csv"

# ──────────────────────────────────────────────
# PALETTE  (slate-charcoal + copper-gold accent)
# ──────────────────────────────────────────────
C = dict(
    bg        = "#F7F8FA",
    white     = "#FFFFFF",
    navy      = "#1C2B3A",
    slate     = "#2E4057",
    mid       = "#4A6FA5",
    muted     = "#8DA5BF",
    copper    = "#B07D3A",
    gold      = "#D4A853",
    gold_lt   = "#F5E4BE",
    red       = "#C0392B",
    red_lt    = "#FBECEB",
    green     = "#1A6B4A",
    green_lt  = "#E8F5EE",
    border    = "#DEE4EC",
    grid      = "#EDF0F4",
    text      = "#1C2B3A",
    sub       = "#5A6B7D",
)

FONT = "DM Sans, sans-serif"

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: '{FONT}';
    background: {C['bg']};
    color: {C['text']};
}}

#MainMenu, footer {{ visibility: hidden; }}
.block-container {{
    padding: 0 2.5rem 3rem !important;
    max-width: 100% !important;
}}

section[data-testid="stSidebar"] {{
    background: {C['navy']} !important;
    border-right: none !important;
}}
section[data-testid="stSidebar"] * {{ color: #C8D8E8 !important; }}
section[data-testid="stSidebar"] .stRadio > div {{
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 6px !important;
    padding: 9px 13px !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(212,168,83,0.18) !important;
    border-color: {C['gold']} !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.12) !important;
}}
section[data-testid="stSidebar"] h3 {{
    color: {C['gold']} !important;
    font-size: .65rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
}}

.hero-band {{
    background: linear-gradient(120deg, {C['navy']} 0%, {C['slate']} 55%, {C['mid']} 100%);
    padding: clamp(18px,4vw,38px) clamp(18px,5vw,52px);
    margin-bottom: 0;
    border-bottom: 3px solid {C['gold']};
}}

.sec-label {{
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: {C['copper']};
    border-bottom: 1px solid {C['border']};
    padding-bottom: .45rem;
    margin-bottom: 1rem;
}}

.kpi {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-top: 3px solid {C['mid']};
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}
.kpi-label {{
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: {C['sub']};
    margin-bottom: .4rem;
}}
.kpi-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {C['navy']};
    font-family: 'DM Mono', monospace;
    line-height: 1;
}}
.kpi-sub  {{ font-size: .72rem; color: {C['copper']}; font-weight: 600; margin-top: .35rem; }}
.kpi-up   {{ color: {C['green']} !important; }}
.kpi-down {{ color: {C['red']}   !important; }}

.mbox {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 1rem 1.1rem;
    text-align: center;
}}
.mbox-label {{
    font-size: .62rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: {C['sub']}; margin-bottom: .35rem;
}}
.mbox-value {{
    font-size: 1.35rem; font-weight: 700; color: {C['navy']};
    font-family: 'DM Mono', monospace;
}}

.insight-bar {{
    background: {C['white']};
    border-left: 3px solid {C['gold']};
    border-radius: 0 6px 6px 0;
    padding: .9rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: .82rem;
    color: {C['slate']};
    line-height: 1.55;
}}

.dec-card {{
    background: {C['white']};
    border: 1px solid {C['border']};
    border-left: 4px solid {C['mid']};
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    margin-bottom: .7rem;
}}
.dec-tag {{
    display: inline-block; font-size: .58rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
    padding: .15rem .55rem; border-radius: 3px; margin-bottom: .4rem;
}}
.tag-price  {{ background: #E8EFF8; color: {C['mid']}; }}
.tag-fx     {{ background: {C['gold_lt']}; color: {C['copper']}; }}
.tag-vol    {{ background: {C['red_lt']};  color: {C['red']}; }}
.tag-qty    {{ background: {C['green_lt']}; color: {C['green']}; }}
.dec-text   {{ font-size: .8rem; color: {C['slate']}; line-height: 1.5; margin: 0; }}

.sb-stat {{
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-top: 2px solid {C['gold']};
    border-radius: 6px; padding: 10px 12px; margin-bottom: 8px;
}}
.sb-stat-label {{
    font-size: .58rem; font-weight: 700; color: {C['gold']} !important;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;
}}
.sb-stat-value {{ font-size: 1rem; font-weight: 700; color: #FFFFFF !important; font-family: 'DM Mono', monospace; }}
.sb-stat-sub   {{ font-size: .65rem; color: #A0BCD0 !important; margin-top: 2px; }}

.div {{ height: 1px; background: {C['border']}; margin: 1.8rem 0; }}
.chart-wrap {{ background: {C['white']}; border: 1px solid {C['border']}; border-radius: 6px; padding: .4rem; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    for c in ["Export Quantity (kg)", "Export Price (LKR/kg)",
              "Exchange Rate (LKR/USD)", "Export Price (USD/kg)",
              "Global_Tea_Price ($)", "Log_Return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df["Roll_Vol_6M"]  = df["Log_Return"].rolling(6).std()
    df["Roll_Vol_12M"] = df["Log_Return"].rolling(12).std()
    df["Year"]  = df.index.year
    df["Month"] = df.index.month
    return df


@st.cache_data
def run_arima(_df):
    returns  = _df["Log_Return"].dropna()
    train_n  = int(len(returns) * 0.8)
    train, test = returns[:train_n], returns[train_n:]
    m_val    = ARIMA(train, order=(1,0,0)).fit()
    pred     = m_val.get_forecast(steps=len(test)).predicted_mean
    mae      = mean_absolute_error(test, pred)
    rmse     = np.sqrt(mean_squared_error(test, pred))
    nz       = test[test.abs() > 1e-6]
    mape     = np.mean(np.abs((nz - pred[nz.index]) / nz)) * 100
    m_full   = ARIMA(returns, order=(1,0,0)).fit()
    fc       = m_full.get_forecast(steps=12)
    fm, ci   = fc.predicted_mean, fc.conf_int()
    last_p   = _df["Export Price (USD/kg)"].iloc[-1]
    last_d   = _df.index[-1]
    dates    = pd.date_range(last_d + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    fm.index = ci.index = dates
    fc_p, fc_lo, fc_hi, prev = [], [], [], last_p
    for i in range(12):
        p = prev * np.exp(fm.iloc[i])
        fc_p.append(p)
        fc_lo.append(prev * np.exp(ci.iloc[i, 0]))
        fc_hi.append(prev * np.exp(ci.iloc[i, 1]))
        prev = p
    fc_df     = pd.DataFrame({"forecast": fc_p, "lower": fc_lo, "upper": fc_hi}, index=dates)
    direction = ("Moderate Increase" if fc_p[-1] > last_p * 1.02
                 else "Slight Decline" if fc_p[-1] < last_p * 0.98
                 else "Stable")
    return fc_df, mae, rmse, mape, direction


# ──────────────────────────────────────────────
# PLOT DEFAULTS
# ──────────────────────────────────────────────
def base_layout(**kw):
    d = dict(
        font=dict(family=FONT, color=C["sub"], size=12),
        paper_bgcolor=C["white"], plot_bgcolor=C["white"],
        margin=dict(l=20, r=20, t=36, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=C["grid"], showline=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=C["grid"], showline=False, tickfont=dict(size=11)),
        hovermode="x unified",
    )
    d.update(kw)
    return d

PCFG = {"displayModeBar": False}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def divider():
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

def kpi_card(label, value, sub=None, sub_cls=""):
    sub_h = f'<div class="kpi-sub {sub_cls}">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{sub_h}</div>'

def mbox(label, value):
    return f'<div class="mbox"><div class="mbox-label">{label}</div><div class="mbox-value">{value}</div></div>'

def sec(label):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CHART FACTORIES
# ──────────────────────────────────────────────
def chart_price_global(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Export Price (USD/kg)"],
        name="Sri Lanka Export Price", line=dict(color=C["navy"], width=2.2),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}/kg<extra></extra>"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Global_Tea_Price ($)"],
        name="Global Benchmark", line=dict(color=C["muted"], width=1.6, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(
        title=dict(text="Sri Lanka Export Price vs Global Benchmark",
                   font=dict(size=13, color=C["navy"]), x=0),
        yaxis_title="USD / kg"))
    return fig


def chart_fx_trend(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Exchange Rate (LKR/USD)"],
        line=dict(color=C["mid"], width=2),
        fill="tozeroy", fillcolor="rgba(74,111,165,0.07)",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f}<extra></extra>",
        showlegend=False))
    fig.update_layout(**base_layout(
        title=dict(text="LKR / USD Exchange Rate", font=dict(size=13, color=C["navy"]), x=0),
        yaxis_title="LKR / USD"))
    return fig


def chart_fx_scatter(df):
    fig = px.scatter(df, x="Exchange Rate (LKR/USD)", y="Export Price (USD/kg)",
                     opacity=0.5, trendline="ols",
                     trendline_color_override=C["copper"])
    fig.update_traces(marker=dict(color=C["mid"], size=5))
    fig.update_layout(**base_layout(
        title=dict(text="Exchange Rate vs Export Price", font=dict(size=13, color=C["navy"]), x=0),
        xaxis_title="Exchange Rate (LKR/USD)", yaxis_title="Export Price (USD/kg)",
        showlegend=False))
    return fig


def chart_volatility(df):
    vol_mean = df["Roll_Vol_6M"].mean()
    vol_std  = df["Roll_Vol_6M"].std()
    spikes   = df["Roll_Vol_6M"] > (vol_mean + 2 * vol_std)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Roll_Vol_12M"],
        name="12-Month", line=dict(color=C["muted"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["Roll_Vol_6M"],
        name="6-Month", line=dict(color=C["navy"], width=2)))
    if spikes.any():
        fig.add_trace(go.Scatter(x=df.index[spikes], y=df["Roll_Vol_6M"][spikes],
            name="Spike", mode="markers",
            marker=dict(color=C["red"], size=7, symbol="circle")))
    fig.update_layout(**base_layout(
        title=dict(text="Rolling Price Volatility — 6M & 12M Windows",
                   font=dict(size=13, color=C["navy"]), x=0),
        yaxis_title="Std Dev (Log Returns)"))
    return fig


def chart_forecast(df, fc_df):
    hist = df["Export Price (USD/kg)"].iloc[-36:]
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc_df.index.tolist() + fc_df.index.tolist()[::-1],
        y=fc_df["upper"].tolist() + fc_df["lower"].tolist()[::-1],
        fill="toself", fillcolor="rgba(74,111,165,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values,
        name="Historical (36M)", line=dict(color=C["navy"], width=2.2)))
    fig.add_trace(go.Scatter(
        x=[df.index[-1], fc_df.index[0]],
        y=[df["Export Price (USD/kg)"].iloc[-1], fc_df["forecast"].iloc[0]],
        line=dict(color=C["red"], width=1.5, dash="dot"),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["forecast"],
        name="12-Month Forecast", line=dict(color=C["red"], width=2.5),
        mode="lines+markers", marker=dict(size=5, color=C["red"])))
    fig.update_layout(**base_layout(
        title=dict(text="ARIMA(1,0,0) — 12-Month Export Price Forecast",
                   font=dict(size=13, color=C["navy"]), x=0),
        yaxis_title="Export Price (USD/kg)"))
    return fig


def chart_annual_avg(df):
    aa = df.groupby("Year")["Export Price (USD/kg)"].mean().reset_index()
    fig = go.Figure(go.Bar(
        x=aa["Year"].astype(str), y=aa["Export Price (USD/kg)"].round(3),
        marker=dict(color=aa["Export Price (USD/kg)"],
                    colorscale=[[0, C["grid"]], [0.5, C["muted"]], [1, C["navy"]]],
                    showscale=False, line=dict(width=0)),
        text=aa["Export Price (USD/kg)"].round(2),
        texttemplate="$%{text}", textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg: $%{y:.3f}/kg<extra></extra>"))
    fig.update_layout(**base_layout(
        title=dict(text="Annual Average Export Price", font=dict(size=13, color=C["navy"]), x=0),
        yaxis=dict(gridcolor=C["grid"], range=[0, aa["Export Price (USD/kg)"].max() * 1.18])))
    return fig


def chart_seasonality_heatmap(df):
    mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    piv    = df.pivot_table(index="Year", columns="Month",
                             values="Export Price (USD/kg)", aggfunc="mean").reindex(columns=range(1,13))
    piv.columns = mnames
    zc = [[None if np.isnan(v) else round(v,3) for v in row] for row in piv.values]
    tx = [["$"+f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in piv.values]
    fig = go.Figure(go.Heatmap(
        z=zc, x=mnames, y=[str(y) for y in piv.index],
        colorscale=[[0, C["grid"]], [0.5, C["gold_lt"]], [1, C["copper"]]],
        text=tx, texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="<b>%{y} %{x}</b><br>%{text}<extra></extra>",
        showscale=True, colorbar=dict(title="$/kg", tickfont=dict(size=10)),
        zmin=df["Export Price (USD/kg)"].min(), zmax=df["Export Price (USD/kg)"].max()))
    fig.update_layout(**base_layout(
        title=dict(text="Monthly Average Price by Year",
                   font=dict(size=13, color=C["navy"]), x=0),
        margin=dict(l=20, r=20, t=36, b=20)))
    return fig


def chart_log_returns(df):
    r = df["Log_Return"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.index, y=r.values,
        line=dict(color=C["mid"], width=1.4), showlegend=False,
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.4f}<extra></extra>"))
    fig.add_hline(y=0, line_color=C["border"], line_width=1.5)
    fig.update_layout(**base_layout(
        title=dict(text="Monthly Log Returns", font=dict(size=13, color=C["navy"]), x=0),
        yaxis_title="Log Return"))
    return fig


def chart_qty_price(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df.index, y=df["Export Quantity (kg)"] / 1e6,
        name="Export Volume (M kg)", marker_color="rgba(74,111,165,0.35)",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f}M kg<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["Export Price (USD/kg)"],
        name="Export Price", line=dict(color=C["copper"], width=2.2),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}<extra></extra>"), secondary_y=True)
    fig.update_layout(**base_layout(
        title=dict(text="Export Volume vs Export Price", font=dict(size=13, color=C["navy"]), x=0)))
    fig.update_yaxes(title_text="Volume (M kg)", secondary_y=False, gridcolor=C["grid"])
    fig.update_yaxes(title_text="Price (USD/kg)", secondary_y=True, showgrid=False, tickprefix="$")
    return fig


def chart_premium(df):
    p_series = df["Export Price (USD/kg)"] - df["Global_Tea_Price ($)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=p_series,
        marker_color=[C["navy"] if v >= 0 else C["red"] for v in p_series],
        hovertemplate="<b>%{x|%b %Y}</b><br>Premium: $%{y:.3f}<extra></extra>",
        showlegend=False))
    fig.add_hline(y=0, line_color=C["border"], line_width=1.5)
    fig.update_layout(**base_layout(
        title=dict(text="SL Price Premium over Global Benchmark",
                   font=dict(size=13, color=C["navy"]), x=0),
        yaxis_title="USD / kg"))
    return fig


def chart_fx_dual(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["Exchange Rate (LKR/USD)"],
        name="Exchange Rate", line=dict(color=C["mid"], width=1.8),
        hovertemplate="<b>%{x|%b %Y}</b><br>LKR/USD: %{y:.1f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df["Export Price (USD/kg)"],
        name="Export Price", line=dict(color=C["copper"], width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.3f}/kg<extra></extra>"), secondary_y=True)
    fig.update_layout(**base_layout(
        title=dict(text="Exchange Rate & Export Price — Divergence",
                   font=dict(size=13, color=C["navy"]), x=0)))
    fig.update_yaxes(title_text="LKR / USD", secondary_y=False, gridcolor=C["grid"])
    fig.update_yaxes(title_text="Export Price (USD/kg)", secondary_y=True, showgrid=False, tickprefix="$")
    return fig


def chart_yoy(df, yrs):
    pal = [C["navy"], C["mid"], C["copper"], C["gold"], C["muted"], C["green"], C["red"], "#8E44AD"]
    mn  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure()
    for idx, yr in enumerate(yrs):
        yd = df[df["Year"] == yr].sort_values("Month")
        fig.add_trace(go.Scatter(
            x=[mn[m-1] for m in yd["Month"]], y=yd["Export Price (USD/kg)"],
            mode="lines+markers", name=str(yr),
            line=dict(color=pal[idx % len(pal)], width=2.2), marker=dict(size=6),
            hovertemplate=f"<b>{yr}</b> %{{x}}<br>$%{{y:.3f}}/kg<extra></extra>"))
    fig.update_layout(**base_layout(
        title=dict(text="Year-on-Year Price Comparison", font=dict(size=13, color=C["navy"]), x=0),
        yaxis_title="USD / kg"))
    return fig


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
df = load_data()
fc_df, mae, rmse, mape, direction = run_arima(df)

latest_price  = df["Export Price (USD/kg)"].iloc[-1]
prev_price    = df["Export Price (USD/kg)"].iloc[-2]
price_delta   = latest_price - prev_price
latest_fx     = df["Exchange Rate (LKR/USD)"].iloc[-1]
prev_fx       = df["Exchange Rate (LKR/USD)"].iloc[-2]
fx_delta      = latest_fx - prev_fx
latest_global = df["Global_Tea_Price ($)"].iloc[-1]
latest_qty    = df["Export Quantity (kg)"].iloc[-1]
latest_date   = df.index[-1].strftime("%B %Y")
avg_12m       = df["Export Price (USD/kg)"].iloc[-12:].mean()
premium       = latest_price - latest_global

cv_pct     = (df["Roll_Vol_6M"].iloc[-1] / avg_12m * 100) if avg_12m else 0
vol_regime = "Elevated" if cv_pct > 5 else "Warning" if cv_pct > 3 else "Stable"


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:24px 0 16px;
         border-bottom:1px solid rgba(255,255,255,0.12); margin-bottom:4px;'>
      <svg width="52" height="52" viewBox="0 0 64 64" fill="none"
           xmlns="http://www.w3.org/2000/svg" style="display:block; margin:0 auto 10px;">
        <circle cx="32" cy="32" r="30" fill="{C['slate']}" stroke="{C['gold']}" stroke-width="2"/>
        <path d="M20 44 Q32 12 44 44" stroke="{C['gold']}" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <path d="M24 36 Q32 20 40 36" stroke="#D4A853" stroke-width="2"
              fill="rgba(212,168,83,0.15)" stroke-linecap="round"/>
        <circle cx="32" cy="46" r="3" fill="{C['gold']}"/>
        <rect x="30.5" y="46" width="3" height="6" rx="1.5" fill="{C['copper']}"/>
        <rect x="14" y="46" width="36" height="3" rx="1.5" fill="{C['copper']}" opacity="0.6"/>
      </svg>
      <div style='font-size:1.25rem; font-weight:700; color:#FFFFFF; letter-spacing:.03em;'>
        Ceylon Tea
      </div>
      <div style='font-size:.62rem; color:{C["gold"]}; margin-top:3px;
           letter-spacing:2px; font-weight:600; text-transform:uppercase;'>
        Export Analytics
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Navigation")

    section = st.radio("", [
        "Overview & History",
        "Price & Market Trends",
        "Exchange Rate Impact",
        "Volatility & Risk",
        "Forecasting",
        "Model Performance",
        "Decision Insights",
        "Year Comparison",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Live Indicators")

    for sb_label, sb_val, sb_sub in [
        ("Latest Price",   f"${latest_price:.3f}/kg",   f"Delta {price_delta:+.3f} vs prior"),
        ("Exchange Rate",  f"{latest_fx:.1f} LKR",      f"Delta {fx_delta:+.1f} LKR/USD"),
        ("12M Average",    f"${avg_12m:.3f}/kg",         f"Premium ${premium:+.3f}"),
        ("12M Forecast",   direction,                    f"Terminal ${fc_df['forecast'].iloc[-1]:.3f}"),
    ]:
        st.markdown(f"""
        <div class="sb-stat">
          <div class="sb-stat-label">{sb_label}</div>
          <div class="sb-stat-value">{sb_val}</div>
          <div class="sb-stat-sub">{sb_sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12);
         border-radius:6px; padding:14px 12px; text-align:center;'>
      <div style='font-size:.58rem; font-weight:700; color:{C["gold"]};
           text-transform:uppercase; letter-spacing:1.5px; margin-bottom:8px;'>
        Dataset
      </div>
      <div style='font-size:.72rem; color:#A0BCD0; line-height:1.8;'>
        2008 – 2024 · Monthly<br>
        204 Observations<br>
        ARIMA(1,0,0) Model<br>
        80/20 Train-Test Split
      </div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="hero-band">
  <div style='display:inline-block; background:rgba(212,168,83,0.2);
       border:1px solid rgba(212,168,83,0.4); border-radius:20px;
       padding:4px 16px; font-size:.68rem; font-weight:700; color:{C["gold"]};
       letter-spacing:1px; margin-bottom:10px; text-transform:uppercase;'>
    Sri Lanka Tea Export Price Volatility
  </div>
  <h1 style='font-size:clamp(1.3rem,5vw,2.1rem); font-weight:700; color:#FFFFFF;
       margin:0 0 10px; line-height:1.2; letter-spacing:-.02em;'>
    Decision-Support Analytics &middot; 2008 &ndash; 2024
  </h1>
  <p style='color:#A8C0D6; font-size:clamp(.78rem,2.2vw,.9rem); max-width:580px;
       margin:0; line-height:1.7; font-weight:400; opacity:.9;'>
    ARIMA time series analysis of Sri Lankan tea export price dynamics.
    Covers price volatility, exchange rate impact, and 12-month forward forecasting
    for business professionals, analysts, and policy makers.
  </p>
</div>
<div style='margin-bottom:1.8rem;'></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE ROUTING
# ══════════════════════════════════════════════

# ── OVERVIEW & HISTORY ────────────────────────
if section == "Overview & History":
    sec("Executive Overview")

    k1, k2, k3, k4 = st.columns(4, gap="small")
    def arrow_p(v):
        return ("▲" if v >= 0 else "▼") + f" {abs(v):.3f} vs prior month"
    def arrow_f(v):
        return ("▲" if v >= 0 else "▼") + f" {abs(v):.1f} LKR vs prior month"

    with k1:
        cls = "kpi-up" if price_delta >= 0 else "kpi-down"
        st.markdown(kpi_card("Export Price (USD/kg)", f"${latest_price:.3f}",
                              arrow_p(price_delta), cls), unsafe_allow_html=True)
    with k2:
        cls = "kpi-down" if fx_delta >= 0 else "kpi-up"
        st.markdown(kpi_card("Exchange Rate (LKR/USD)", f"{latest_fx:.1f}",
                              arrow_f(fx_delta), cls), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Global Benchmark Price", f"${latest_global:.3f}",
                              f"SL Premium: ${premium:+.3f}/kg"), unsafe_allow_html=True)
    with k4:
        dir_cls = "kpi-up" if "Increase" in direction else "kpi-down" if "Decline" in direction else ""
        st.markdown(kpi_card("12-Month Forecast", direction,
                              f"ARIMA(1,0,0) · {latest_date}", dir_cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-bar">
      <strong>Market Context &nbsp;&middot;</strong>&nbsp;
      Sri Lanka export prices maintain a persistent premium over global benchmarks,
      driven primarily by LKR depreciation. Forward model indicates
      <strong>{direction.lower()}</strong>,
      terminal forecast <strong>${fc_df['forecast'].iloc[-1]:.3f}/kg</strong>.
      Post-2020 volatility is structurally elevated relative to the 2008–2019 baseline.
    </div>""", unsafe_allow_html=True)

    divider()
    sec("Price Trend & Quick Statistics")

    col_chart, col_stats = st.columns([2.2, 1], gap="medium")
    with col_chart:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart_price_global(df), use_container_width=True, config=PCFG)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_stats:
        t36 = df["Export Price (USD/kg)"].iloc[-36:]
        for lbl, val in [
            ("3-Year Average", f"${t36.mean():.3f}/kg"),
            ("3-Year High",    f"${t36.max():.3f}/kg"),
            ("3-Year Low",     f"${t36.min():.3f}/kg"),
            ("Std Deviation",  f"${t36.std():.3f}"),
            ("Latest Volume",  f"{latest_qty/1e6:.1f}M kg"),
        ]:
            st.markdown(f"""
            <div style='background:{C["white"]}; border:1px solid {C["border"]};
                 border-left:3px solid {C["mid"]}; border-radius:0 6px 6px 0;
                 padding:9px 13px; margin-bottom:8px;'>
              <div style='font-size:.65rem; color:{C["sub"]}; font-weight:700;
                   text-transform:uppercase; letter-spacing:.08em;'>{lbl}</div>
              <div style='font-size:1.2rem; font-weight:700; color:{C["navy"]};
                   font-family:"DM Mono",monospace;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    divider()
    sec("Seasonal Price Patterns")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(chart_seasonality_heatmap(df), use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)

    divider()
    sec("Trend Analysis")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart_annual_avg(df), use_container_width=True, config=PCFG)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart_log_returns(df), use_container_width=True, config=PCFG)
        st.markdown('</div>', unsafe_allow_html=True)

    divider()
    sec("Dataset Statistics")
    s1,s2,s3,s4,s5 = st.columns(5, gap="small")
    for col, (lbl, val) in zip([s1,s2,s3,s4,s5], [
        ("Observations",  str(len(df))),
        ("Max Price",     f"${df['Export Price (USD/kg)'].max():.3f}"),
        ("Min Price",     f"${df['Export Price (USD/kg)'].min():.3f}"),
        ("Mean Price",    f"${df['Export Price (USD/kg)'].mean():.3f}"),
        ("Coverage",      "2008–2024"),
    ]):
        with col:
            st.markdown(mbox(lbl, val), unsafe_allow_html=True)


# ── PRICE & MARKET TRENDS ─────────────────────
elif section == "Price & Market Trends":
    sec("Price & Market Trends")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(chart_price_global(df), use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)

    divider()
    sec("Volume & Price Dynamics")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(chart_qty_price(df), use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-bar" style="margin-top:1rem;">
      <strong>Finding:</strong> Export quantity shows weak correlation with price.
      Destination-market mix and global demand are stronger price drivers than domestic volume.
    </div>""", unsafe_allow_html=True)

    divider()
    sec("Price Premium over Global Benchmark")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(chart_premium(df), use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)


# ── EXCHANGE RATE IMPACT ──────────────────────
elif section == "Exchange Rate Impact":
    sec("Exchange Rate Impact Analysis")
    c1, c2 = st.columns([1.5, 1], gap="medium")
    with c1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart_fx_trend(df), use_container_width=True, config=PCFG)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart_fx_scatter(df), use_container_width=True, config=PCFG)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-bar" style="margin-top:1rem;">
      <strong>Exchange Rate Finding:</strong> LKR depreciation is the primary structural driver
      of USD-denominated export prices post-2020. The rate rose from ~108 (2008) to ~371 (peak 2022),
      amplifying apparent USD price while masking local supply-demand signals.
    </div>""", unsafe_allow_html=True)

    divider()
    sec("Rate & Price Divergence")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(chart_fx_dual(df), use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)

    divider()
    sec("Exchange Rate Statistics")
    m1,m2,m3,m4,m5 = st.columns(5, gap="small")
    for col, (lbl, val) in zip([m1,m2,m3,m4,m5], [
        ("Current Rate",  f"{latest_fx:.1f}"),
        ("2008 Rate",     f"{df['Exchange Rate (LKR/USD)'].iloc[0]:.1f}"),
        ("Peak Rate",     f"{df['Exchange Rate (LKR/USD)'].max():.1f}"),
        ("Depreciation",  f"{((latest_fx/df['Exchange Rate (LKR/USD)'].iloc[0])-1)*100:.0f}%"),
        ("Corr w/ Price", f"{df['Export Price (USD/kg)'].corr(df['Exchange Rate (LKR/USD)']):.3f}"),
    ]):
        with col:
            st.markdown(mbox(lbl, val), unsafe_allow_html=True)


# ── VOLATILITY & RISK ─────────────────────────
elif section == "Volatility & Risk":
    sec("Volatility & Risk Assessment")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(chart_volatility(df), use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)

    divider()
    sec("Current Volatility Regime")
    v1,v2,v3,v4,v5 = st.columns(5, gap="small")
    cv6  = (df["Roll_Vol_6M"].iloc[-1]  / avg_12m * 100) if avg_12m else 0
    cv12 = (df["Roll_Vol_12M"].iloc[-1] / avg_12m * 100) if avg_12m else 0
    spk  = int((df["Roll_Vol_6M"] > (df["Roll_Vol_6M"].mean() + 2*df["Roll_Vol_6M"].std())).sum())
    post20 = df[df.index >= "2020-01-01"]["Roll_Vol_6M"].mean() / avg_12m * 100
    for col, (lbl, val) in zip([v1,v2,v3,v4,v5], [
        ("6M Vol (CV%)",   f"{cv6:.1f}%"),
        ("12M Vol (CV%)",  f"{cv12:.1f}%"),
        ("Vol Regime",     vol_regime),
        ("Spike Events",   f"{spk}"),
        ("Post-2020 CV",   f"{post20:.1f}%"),
    ]):
        with col:
            st.markdown(mbox(lbl, val), unsafe_allow_html=True)

    divider()
    sec("Risk Indicators")
    risks = [
        (C["red"],   "Structural Volatility Shift",
         "Post-2020 volatility is materially higher than the 2008–2019 baseline, indicating a structural "
         "regime change in the pricing environment, not a temporary fluctuation."),
        (C["copper"],"Exchange Rate Transmission",
         "LKR depreciation creates rapid USD price spikes that may not reflect true supply-demand conditions, "
         "complicating forward contract pricing and margin planning."),
        (C["navy"],  "Global Price Divergence",
         "Sri Lanka prices periodically diverge from global benchmarks, creating export opportunity "
         "windows and competitiveness risk depending on direction."),
        (C["green"], "Volume Resilience",
         "Export volumes have remained relatively stable despite price volatility, suggesting "
         "inelastic demand from key importing markets."),
    ]
    r1, r2 = st.columns(2, gap="medium")
    for i, (dot_clr, title, desc) in enumerate(risks):
        col = r1 if i % 2 == 0 else r2
        with col:
            st.markdown(f"""
            <div style='background:{C["white"]}; border:1px solid {C["border"]};
                 border-radius:6px; padding:.9rem 1rem; margin-bottom:.6rem;
                 display:flex; gap:10px; align-items:flex-start;'>
              <div style='width:8px; height:8px; border-radius:50%;
                   background:{dot_clr}; flex-shrink:0; margin-top:4px;'></div>
              <div>
                <div style='font-size:.82rem; font-weight:700; color:{C["navy"]}; margin-bottom:4px;'>{title}</div>
                <div style='font-size:.75rem; color:{C["sub"]}; line-height:1.55;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    divider()
    sec("Return Distribution")
    returns = df["Log_Return"].dropna()
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=returns, nbinsx=40,
        marker_color=C["mid"], opacity=0.75, showlegend=False))
    fig_h.add_vline(x=0, line_color=C["border"], line_dash="dash")
    fig_h.update_layout(**base_layout(
        title=dict(text="Distribution of Monthly Log Returns",
                   font=dict(size=13, color=C["navy"]), x=0),
        xaxis_title="Log Return", yaxis_title="Frequency"))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_h, use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)


# ── FORECASTING ───────────────────────────────
elif section == "Forecasting":
    sec("12-Month Price Forecast — ARIMA(1,0,0)")
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(chart_forecast(df, fc_df), use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)

    divider()
    sec("Month-by-Month Forecast Detail")
    cols_fc = st.columns(6)
    for i, (date, row) in enumerate(fc_df.iterrows()):
        if i >= 12: break
        p   = row["forecast"]
        chg = ((p - latest_price) / latest_price) * 100
        clr = C["green"] if chg >= 0 else C["red"]
        with cols_fc[i % 6]:
            st.markdown(f"""
            <div style='background:{C["white"]}; border:1px solid {C["border"]};
                 border-top:3px solid {clr}; border-radius:6px;
                 padding:10px 8px; text-align:center; margin-bottom:8px;'>
              <div style='font-size:.62rem; color:{C["sub"]}; margin-bottom:3px;'>
                {date.strftime("%b %Y")}
              </div>
              <div style='font-size:.95rem; font-weight:700; color:{clr};
                   font-family:"DM Mono",monospace;'>${p:.3f}</div>
              <div style='font-size:.62rem; font-weight:600; color:{clr};'>
                {"▲" if chg >= 0 else "▼"} {abs(chg):.1f}%
              </div>
              <div style='font-size:.58rem; color:{C["muted"]}; margin-top:3px;'>
                ${row["lower"]:.2f}–${row["upper"]:.2f}
              </div>
            </div>""", unsafe_allow_html=True)

    divider()
    sec("Forecast Summary")
    fs1,fs2,fs3,fs4,fs5 = st.columns(5, gap="small")
    for col, (lbl, val) in zip([fs1,fs2,fs3,fs4,fs5], [
        ("Avg Forecast",  f"${fc_df['forecast'].mean():.3f}"),
        ("Peak Forecast", f"${fc_df['forecast'].max():.3f}"),
        ("Low Forecast",  f"${fc_df['forecast'].min():.3f}"),
        ("Range",         f"${fc_df['forecast'].max()-fc_df['forecast'].min():.3f}"),
        ("Direction",     direction),
    ]):
        with col:
            st.markdown(mbox(lbl, val), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-bar" style="margin-top:1.2rem;">
      <strong>Methodology:</strong> ARIMA(1,0,0) on monthly log returns.
      Forecast back-transformed via P(t) &times; exp(r&#770;).
      Terminal forecast: <strong>${fc_df['forecast'].iloc[-1]:.3f}/kg</strong>
      (95% CI: ${fc_df['lower'].iloc[-1]:.3f} &ndash; ${fc_df['upper'].iloc[-1]:.3f}).
    </div>""", unsafe_allow_html=True)


# ── MODEL PERFORMANCE ─────────────────────────
elif section == "Model Performance":
    sec("Model Performance — ARIMA(1,0,0)")
    p1, p2, p3 = st.columns(3, gap="medium")
    for col, (lbl, val) in zip([p1,p2,p3], [
        ("Mean Absolute Error (MAE)",  f"{mae:.5f}"),
        ("Root Mean Sq Error (RMSE)",  f"{rmse:.5f}"),
        ("Mean Abs Pct Error (MAPE)",  f"{mape:.1f}%"),
    ]):
        with col:
            st.markdown(f"""
            <div style='background:{C["white"]}; border:1px solid {C["border"]};
                 border-top:3px solid {C["copper"]}; border-radius:6px;
                 padding:1.4rem 1.6rem; text-align:center;'>
              <div style='font-size:.65rem; font-weight:700; letter-spacing:.1em;
                   text-transform:uppercase; color:{C["sub"]}; margin-bottom:.6rem;'>{lbl}</div>
              <div style='font-size:2rem; font-weight:700; color:{C["navy"]};
                   font-family:"DM Mono",monospace;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-bar" style="margin-top:1.2rem;">
      <strong>Performance Note:</strong> MAE and RMSE are the primary reliability indicators.
      MAPE is elevated due to near-zero log-return values at price inflection points —
      a known statistical artefact. Validated on held-out 20% test set.
    </div>""", unsafe_allow_html=True)

    divider()
    sec("Model Diagnostics — Test Set")
    returns = df["Log_Return"].dropna()
    train_n = int(len(returns) * 0.8)
    train_r, test_r = returns[:train_n], returns[train_n:]
    m_val = ARIMA(train_r, order=(1,0,0)).fit()
    pred_r = m_val.get_forecast(steps=len(test_r)).predicted_mean
    pred_r.index = test_r.index
    residuals = test_r.values - pred_r.values

    fig_diag = make_subplots(rows=1, cols=2,
        subplot_titles=["Actual vs Predicted (Test Set)", "Residuals"])
    fig_diag.add_trace(go.Scatter(x=test_r.index, y=test_r.values,
        name="Actual", line=dict(color=C["navy"], width=2)), row=1, col=1)
    fig_diag.add_trace(go.Scatter(x=pred_r.index, y=pred_r.values,
        name="Predicted", line=dict(color=C["red"], width=1.8, dash="dot")), row=1, col=1)
    fig_diag.add_trace(go.Bar(x=test_r.index, y=residuals, showlegend=False,
        marker_color=[C["mid"] if v >= 0 else C["red"] for v in residuals]), row=1, col=2)
    fig_diag.add_hline(y=0, line_color=C["border"], row=1, col=2)
    fig_diag.update_layout(
        font=dict(family=FONT, color=C["sub"]),
        paper_bgcolor=C["white"], plot_bgcolor=C["white"],
        height=300, margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor=C["grid"]), xaxis2=dict(gridcolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"]), yaxis2=dict(gridcolor=C["grid"]))
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_diag, use_container_width=True, config=PCFG)
    st.markdown('</div>', unsafe_allow_html=True)


# ── DECISION INSIGHTS ─────────────────────────
elif section == "Decision Insights":
    sec("Decision Insights Panel")
    st.markdown(f"""
    <div style='background:linear-gradient(120deg,{C["navy"]} 0%,{C["slate"]} 55%,{C["mid"]} 100%);
         border-radius:8px; padding:22px 28px; margin-bottom:20px;'>
      <div style='font-size:1.1rem; font-weight:700; color:#FFFFFF; margin-bottom:6px;'>
        Strategic Intelligence Summary
      </div>
      <div style='font-size:.84rem; color:#A8C0D6; line-height:1.7; max-width:720px;'>
        Evidence-based insights from ARIMA time series analysis of Sri Lanka tea export
        pricing dynamics (2008–2024). For
        <strong style='color:{C["gold"]};'>business professionals</strong>,
        <strong style='color:#A8C0D6;'>exporters</strong>, and
        <strong style='color:{C["gold"]};'>policy makers</strong>.
      </div>
    </div>""", unsafe_allow_html=True)

    sn1,sn2,sn3,sn4,sn5 = st.columns(5, gap="small")
    chg3m = ((latest_price / df["Export Price (USD/kg)"].iloc[-4]) - 1) * 100
    for col, (lbl, val) in zip([sn1,sn2,sn3,sn4,sn5], [
        ("Current Price",   f"${latest_price:.3f}"),
        ("3M Change",       f"{chg3m:+.1f}%"),
        ("Vol Regime",      vol_regime),
        ("FX Rate",         f"{latest_fx:.0f} LKR"),
        ("Forecast",        direction),
    ]):
        with col:
            st.markdown(mbox(lbl, val), unsafe_allow_html=True)

    divider()
    sec("Research-Based Insights")

    insights = [
        ("price", "tag-price", "Price Outlook",
         f"Export prices are forecast to show a <strong>{direction.lower()}</strong> over the next 12 months. "
         f"Terminal value: <strong>${fc_df['forecast'].iloc[-1]:.3f}/kg</strong> "
         f"(95% CI: ${fc_df['lower'].iloc[-1]:.3f} &ndash; ${fc_df['upper'].iloc[-1]:.3f}). "
         f"Current price of <strong>${latest_price:.3f}/kg</strong> sits "
         f"{'above' if latest_price > avg_12m else 'below'} the 12-month average of ${avg_12m:.3f}/kg."),
        ("fx", "tag-fx", "Exchange Rate Driver",
         "The LKR/USD exchange rate is the dominant structural driver of USD-denominated export prices. "
         "LKR depreciation from ~108 (2008) to ~296 (2024) is a primary amplifier of apparent price increases. "
         "Businesses should model FX risk independently from volume and quality dynamics."),
        ("vol", "tag-vol", "Volatility & Risk",
         f"Rolling 6-month volatility is currently at a <strong>{vol_regime.lower()}</strong> level. "
         f"Post-2020 volatility is structurally higher than the pre-2020 baseline — consistent with "
         f"macro disruptions (pandemic, currency crisis, supply chain shocks). "
         f"This elevated regime warrants active hedging and scenario planning."),
        ("qty", "tag-qty", "Export Quantity Signal",
         "Export quantity shows weak correlation with price movements across 2008–2024. "
         "Volume fluctuations are not reliable price predictors. Destination-market mix, "
         "quality grade, and global demand conditions are more decisive for price realisation."),
    ]

    for _, tag_cls, title, text in insights:
        st.markdown(f"""
        <div class="dec-card">
          <span class="dec-tag {tag_cls}">{title}</span>
          <p class="dec-text">{text}</p>
        </div>""", unsafe_allow_html=True)

    divider()
    sec("90-Day Action Plan")
    ap_cols = st.columns(4, gap="small")
    plan = [
        ("Week 1–2",  ["Review forward contracts vs. ARIMA forecast", "Reassess FX hedging position"]),
        ("Week 3–4",  ["Benchmark SL prices vs. global index", "Monitor 6M rolling volatility for regime shift"]),
        ("Month 2",   ["Review export destination mix", "Assess seasonal patterns on Q2–Q3 pricing"]),
        ("Month 3",   ["Validate ARIMA forecast vs. realised prices", "Refit model if structural break detected"]),
    ]
    for col, (period, actions) in zip(ap_cols, plan):
        items_html = "".join([
            f"<div style='font-size:.7rem; color:{C['sub']}; padding:5px 0; "
            f"border-bottom:1px solid {C['grid']}; line-height:1.4;'>{a}</div>"
            for a in actions])
        with col:
            st.markdown(f"""
            <div style='background:{C["white"]}; border:1px solid {C["border"]};
                 border-top:3px solid {C["copper"]}; border-radius:6px;
                 padding:14px 12px; min-height:160px;'>
              <div style='font-size:.62rem; font-weight:700; color:{C["copper"]};
                   text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>{period}</div>
              {items_html}
            </div>""", unsafe_allow_html=True)


# ── YEAR COMPARISON ───────────────────────────
elif section == "Year Comparison":
    sec("Year-on-Year Price Comparison")
    avail_years = sorted(df["Year"].unique().tolist())
    selected_years = st.multiselect("Select years to compare", avail_years,
                                     default=avail_years[-4:])
    if selected_years:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(chart_yoy(df, selected_years), use_container_width=True, config=PCFG)
        st.markdown('</div>', unsafe_allow_html=True)

        divider()
        sec("Volatility Distribution by Year")
        pal = [C["navy"], C["mid"], C["copper"], C["gold"], C["muted"], C["green"], C["red"], "#8E44AD"]
        fig_box = go.Figure()
        for idx, yr in enumerate(selected_years):
            fig_box.add_trace(go.Box(
                y=df[df["Year"] == yr]["Export Price (USD/kg)"], name=str(yr),
                marker_color=pal[idx % len(pal)], boxmean=True))
        fig_box.update_layout(**base_layout(
            title=dict(text="Price Distribution by Year", font=dict(size=13, color=C["navy"]), x=0),
            yaxis_title="USD / kg", showlegend=False))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_box, use_container_width=True, config=PCFG)
        st.markdown('</div>', unsafe_allow_html=True)

        divider()
        sec("Statistical Summary by Year")
        rows = []
        for yr in selected_years:
            yd = df[df["Year"] == yr]["Export Price (USD/kg)"]
            fx = df[df["Year"] == yr]["Exchange Rate (LKR/USD)"].mean()
            prev_avg = df[df["Year"] == yr - 1]["Export Price (USD/kg)"].mean()
            rows.append({
                "Year":       yr,
                "Avg ($/kg)": round(yd.mean(), 3),
                "Min ($/kg)": round(yd.min(),  3),
                "Max ($/kg)": round(yd.max(),  3),
                "Std Dev":    round(yd.std(),  3),
                "Avg FX":     round(fx, 1),
                "YoY Chg":   (f"{((yd.mean()/prev_avg)-1)*100:+.1f}%"
                               if yr > min(selected_years) and not np.isnan(prev_avg) else "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one year above to begin comparison.")


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
divider()

footer_cards = "".join([f"""
<div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12);
     border-top:2px solid {C['gold']}; border-radius:6px; padding:14px 12px;">
  <div style="font-size:.58rem; font-weight:700; color:{C['gold']};
       text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">{lbl}</div>
  <div style="font-size:.85rem; font-weight:600; color:#FFFFFF; margin-bottom:6px;">{name}</div>
  <div style="font-size:.72rem; color:#A0BCD0; line-height:1.8;">{info}</div>
</div>""" for lbl, name, info in [
    ("Data Coverage", "2008 – 2024",
     "204 monthly observations<br>Sri Lanka Tea Export Board<br>Central Bank of Sri Lanka"),
    ("Model", "ARIMA(1,0,0)",
     "Log return series<br>80/20 train-test split<br>12-month forecast horizon"),
    ("Key Metrics", "Performance",
     f"MAE: {mae:.5f}<br>RMSE: {rmse:.5f}<br>MAPE: {mape:.1f}%"),
    ("Technology", "Streamlit",
     "Python · statsmodels<br>pandas · scikit-learn<br>Plotly · DM Sans"),
]])

footer_stats = "".join([f"""
<div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12);
     border-radius:5px; padding:10px 20px; text-align:center; min-width:90px;">
  <div style="font-size:1.3rem; font-weight:700; color:#FFFFFF;">{val}</div>
  <div style="font-size:.6rem; color:{C['gold']}; text-transform:uppercase;
       letter-spacing:1px; font-weight:700;">{lbl}</div>
</div>""" for val, lbl in [
    ("204", "Observations"), ("17", "Years"),
    (f"${df['Export Price (USD/kg)'].max():.2f}", "Peak Price"),
    (f"{int(df['Exchange Rate (LKR/USD)'].max())}", "Peak FX"),
    ("ARIMA", "Model"), ("12M", "Horizon"),
]])

st.markdown(f"""
<div style='background:linear-gradient(120deg,{C["navy"]} 0%,{C["slate"]} 55%,{C["mid"]} 100%);
     border-radius:6px; padding:32px; margin-bottom:24px;'>
  <div style='text-align:center; padding-bottom:22px;
       border-bottom:1px solid rgba(255,255,255,0.12); margin-bottom:24px;'>
    <div style='font-size:1.5rem; font-weight:700; color:#FFFFFF; margin-bottom:6px;'>
      Sri Lanka Tea Export Analytics
    </div>
    <div style='font-size:.84rem; color:#A8C0D6;'>
      ARIMA Time Series Research &middot; Price Volatility Analysis &middot; 2008&ndash;2024
    </div>
  </div>
  <div style='display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px;'>
    {footer_cards}
  </div>
  <div style='display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-bottom:18px;'>
    {footer_stats}
  </div>
  <div style='text-align:center; font-size:.7rem; color:#6A8BA8;
       border-top:1px solid rgba(255,255,255,0.1); padding-top:16px;'>
    Ceylon Tea Export Analytics &middot; ARIMA Time Series Analysis &middot;
    Data: Sri Lanka Tea Export Board &amp; Central Bank &middot; For analytical use only
  </div>
</div>
""", unsafe_allow_html=True)
