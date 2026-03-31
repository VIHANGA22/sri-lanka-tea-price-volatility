"""
================================================================================
  Sri Lankan Tea Export Price Research Dashboard
  "Volatility, Structural Changes, and Forecasting of Sri Lankan Tea Export Prices"
  University Dissertation — Streamlit App
================================================================================

REQUIREMENTS (create a requirements.txt with these):
    streamlit
    pandas
    numpy
    plotly
    openpyxl

USAGE (local):
    streamlit run tea_dashboard.py

USAGE (Streamlit Cloud):
    Push to GitHub with both Excel files and requirements.txt in the same repo.
    Deploy from https://share.streamlit.io

FILE STRUCTURE:
    tea_dashboard.py                    <- this file
    Tea_Export_Master_2008_2024.xlsx    <- your data
    DateGlobal_Tea_Price.xlsx           <- your data
    requirements.txt                    <- pip dependencies
================================================================================
"""

import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Sri Lankan Tea Export Price Dashboard",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  THEME CONSTANTS
# ════════════════════════════════════════════════════════════════════════════
BLUE   = "#1a3f6f"
BLUE2  = "#2b5ca8"
ORANGE = "#c95d1a"
ORANGE2= "#e8793a"
GREEN  = "#1e7a4b"
RED    = "#a42020"
PURPLE = "#5b3f9c"
GREY   = "#6e7180"

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=11, color="#4a4a62"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font=dict(size=10)),
    hovermode="x unified",
)

# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Libre+Baskerville:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}

[data-testid="stSidebar"]{background:#1a3f6f}
[data-testid="stSidebar"],[data-testid="stSidebar"] *{color:white!important}
[data-testid="stSidebar"] label{font-size:.75rem;text-transform:uppercase;letter-spacing:.5px;opacity:.8}
[data-testid="stSidebar"] .stSelectbox>div>div{background:rgba(255,255,255,.12)!important;border-color:rgba(255,255,255,.25)!important}

.dash-header{background:linear-gradient(135deg,#1a3f6f 0%,#1e5090 100%);border-bottom:3px solid #c95d1a;padding:22px 28px 18px;border-radius:10px;margin-bottom:20px}
.dash-header h1{font-family:'Libre Baskerville',serif;font-size:1.4rem;font-weight:700;color:white;margin:0 0 5px 0;line-height:1.25}
.dash-header p{font-size:.76rem;color:rgba(255,255,255,.62);margin:0;letter-spacing:.3px}

.kpi-card{background:white;border-radius:10px;padding:17px 18px 14px;border-top:3px solid #1a3f6f;box-shadow:0 2px 10px rgba(0,0,0,.07);height:100%}
.kpi-card.orange{border-top-color:#c95d1a}
.kpi-card.green{border-top-color:#1e7a4b}
.kpi-card.purple{border-top-color:#5b3f9c}
.kpi-lbl{font-size:.66rem;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:#9090a8;margin-bottom:5px}
.kpi-val{font-family:'Libre Baskerville',serif;font-size:1.6rem;font-weight:700;color:#1a3f6f;line-height:1}
.kpi-card.orange .kpi-val{color:#c95d1a}
.kpi-card.green .kpi-val{color:#1e7a4b}
.kpi-card.purple .kpi-val{color:#5b3f9c}
.kpi-unit{font-size:.66rem;color:#9090a8;margin-top:3px}
.kpi-delta{font-size:.69rem;font-weight:600;margin-top:5px}
.kpi-delta.up{color:#1e7a4b}
.kpi-delta.dn{color:#a42020}

.insight-box{background:linear-gradient(135deg,#1a3f6f 0%,#1e5090 100%);color:white;border-radius:10px;padding:16px 20px;margin:4px 0 22px 0;font-size:.83rem;line-height:1.65}
.insight-box strong{color:#f5c97a}
.insight-label{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;opacity:.65;margin-bottom:6px}

.sec-hdr{display:flex;align-items:baseline;gap:10px;border-bottom:2px solid #d8d4cc;padding-bottom:8px;margin-bottom:14px;margin-top:10px}
.sec-num{background:#e6eef8;color:#2b5ca8;border-radius:20px;padding:2px 10px;font-size:.67rem;font-weight:700;letter-spacing:.8px;text-transform:uppercase}
.sec-title{font-family:'Libre Baskerville',serif;font-size:.98rem;font-weight:700;color:#1c1c2e}
.sec-sub{font-size:.70rem;color:#9090a8;font-style:italic;margin-left:auto}

.chip{display:inline-block;font-size:.66rem;font-weight:600;padding:2px 9px;border-radius:20px;margin:2px;letter-spacing:.3px}
.chip-red{background:#faeaea;color:#a42020;border:1px solid #e8b0b0}
.chip-orange{background:#faeee5;color:#c95d1a;border:1px solid #f5c9a8}
.chip-blue{background:#e6eef8;color:#2b5ca8;border:1px solid #c3d6ef}
.chip-green{background:#e4f4ec;color:#1e7a4b;border:1px solid #a8dbc0}

.stat-tbl{width:100%;border-collapse:collapse;font-size:.73rem}
.stat-tbl th{padding:6px 8px;text-align:left;font-size:.64rem;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#9090a8;border-bottom:1px solid #d8d4cc}
.stat-tbl td{padding:5px 8px;border-bottom:1px solid #ebe7e0;color:#4a4a62}
.stat-tbl td:first-child{color:#1c1c2e;font-weight:500}
.stat-tbl .num{text-align:right;font-weight:600;color:#1c1c2e;font-variant-numeric:tabular-nums}
.stat-tbl .pos{color:#1e7a4b;font-weight:700}
.stat-tbl .neg{color:#a42020;font-weight:700}
.stat-tbl tr:last-child td{border-bottom:none}

.regime-panel{border-radius:8px;padding:13px 15px;border:1px solid #d8d4cc;margin-bottom:9px}
.regime-panel.pre{background:linear-gradient(135deg,#f0f5fb 0%,#fff 100%)}
.regime-panel.post{background:linear-gradient(135deg,#fdf3ec 0%,#fff 100%)}
.rp-title{font-family:'Libre Baskerville',serif;font-size:.82rem;font-weight:700;margin-bottom:9px;color:#1c1c2e}
.rp-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.rp-stat{background:rgba(255,255,255,.75);border-radius:5px;padding:6px 8px}
.rp-stat-lbl{font-size:.6rem;color:#9090a8;font-weight:600;text-transform:uppercase;letter-spacing:.4px}
.rp-stat-val{font-size:.85rem;font-weight:700;color:#1c1c2e;font-family:'Libre Baskerville',serif}

#MainMenu{visibility:hidden}
footer{visibility:hidden}
.stDeployButton{display:none}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PROCESSING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading and processing data…")
def load_data():
    try:
        sl_df     = pd.read_excel("Tea_Export_Master_2008_2024.xlsx")
        global_df = pd.read_excel("DateGlobal_Tea_Price.xlsx")
    except FileNotFoundError as e:
        st.error(
            "Excel files not found. Please ensure these files are in the same "
            "folder as tea_dashboard.py:\n\n"
            "- Tea_Export_Master_2008_2024.xlsx\n"
            "- DateGlobal_Tea_Price.xlsx"
        )
        st.stop()

    for d in [sl_df, global_df]:
        d["Month_num"] = pd.to_datetime(d["Month"], format="%B").dt.month
        d.index = pd.to_datetime(
            d["Year"].astype(str) + "-" + d["Month_num"].astype(str),
            format="%Y-%m",
        )
        d.drop(columns=["Month_num"], inplace=True)

    global_df = global_df[["Global_Tea_Price ($)"]]
    df = sl_df.merge(global_df, left_index=True, right_index=True, how="left")

    num_cols = [
        "Export Quantity (kg)", "Export Price (LKR/kg)",
        "Exchange Rate (LKR/USD)", "Export Price (USD/kg)", "Global_Tea_Price ($)",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(
            df[col].astype(str)
                   .str.replace("\n", "", regex=False)
                   .str.replace(",", "", regex=False)
                   .str.strip(),
            errors="coerce",
        )

    df = df.sort_index()
    df[num_cols] = df[num_cols].ffill().bfill()

    df["Log_Return"] = np.log(
        df["Export Price (USD/kg)"] / df["Export Price (USD/kg)"].shift(1)
    )
    df["Vol_6M"]  = df["Log_Return"].rolling(6).std()  * np.sqrt(12) * 100
    df["Vol_12M"] = df["Log_Return"].rolling(12).std() * np.sqrt(12) * 100
    return df


@st.cache_data(show_spinner=False)
def compute_forecasts(_df):
    """ARIMA(1,0,0) via Yule-Walker — matches Colab notebook logic."""
    rets   = _df["Log_Return"].dropna().values
    mu     = rets.mean()
    c0     = np.var(rets)
    c1     = np.cov(rets[:-1], rets[1:])[0, 1]
    phi    = c1 / c0
    sigma2 = c0 * (1 - phi ** 2)

    def _fc(horizon):
        fc_mean = []
        val = rets[-1]
        for _ in range(horizon):
            val = mu + phi * (val - mu)
            fc_mean.append(val)

        fc_upper = [
            fc_mean[h - 1]
            + math.sqrt(sigma2 * sum(phi ** (2 * j) for j in range(h))) * 1.96
            for h in range(1, horizon + 1)
        ]
        fc_lower = [
            fc_mean[h - 1]
            - math.sqrt(sigma2 * sum(phi ** (2 * j) for j in range(h))) * 1.96
            for h in range(1, horizon + 1)
        ]

        lp = float(_df["Export Price (USD/kg)"].iloc[-1])
        pf, uf, lf = [], [], []
        p = u = l = lp
        for i in range(horizon):
            p = p * math.exp(fc_mean[i])
            u = u * math.exp(fc_upper[i])
            l = l * math.exp(fc_lower[i])
            pf.append(round(p, 4))
            uf.append(round(u, 4))
            lf.append(round(l, 4))

        li = _df.index[-1]
        idx = pd.date_range(
            start=pd.Timestamp(li.year + li.month // 12, li.month % 12 + 1, 1),
            periods=horizon, freq="MS",
        )
        return pd.DataFrame({"mean": pf, "upper": uf, "lower": lf}, index=idx)

    return {6: _fc(6), 12: _fc(12), 24: _fc(24)}


def ols_line(x, y):
    x, y = np.array(x, float), np.array(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return x, y
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.sort(x)
    return xs, slope * xs + intercept


def pearson_r(a, b):
    a, b = np.array(a, float), np.array(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ════════════════════════════════════════════════════════════════════════════
#  LOAD
# ════════════════════════════════════════════════════════════════════════════
df        = load_data()
forecasts = compute_forecasts(df)
pre_df    = df[df.index < "2020-01-01"]
post_df   = df[df.index >= "2020-01-01"]

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🍵 Tea Export Dashboard")
    st.caption("Dissertation Research Tool · 2008–2024")
    st.markdown("---")

    year_range = st.slider("Year Range", 2008, 2024, (2008, 2024))

    regime = st.radio(
        "Regime Filter",
        ["All Period", "Pre-2020 (Stable)", "Post-2020 (Crisis)"],
    )

    variable = st.selectbox(
        "Primary Variable",
        ["Export Price (USD/kg)", "Export Price (LKR/kg)",
         "Export Quantity (kg)", "Exchange Rate (LKR/USD)"],
    )

    currency = st.radio("Currency Display", ["USD/kg", "LKR/kg"], horizontal=True)

    fc_horizon = st.select_slider("Forecast Horizon (months)", [6, 12, 24], value=12)

    st.markdown("---")
    st.markdown("**Export Data**")
    dl_placeholder = st.empty()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.67rem;opacity:.55;line-height:1.6'>"
        "Model: ARIMA(1,0,0)<br>"
        "Method: Yule-Walker<br>"
        "n = 204 observations<br>"
        "Jan 2008 – Dec 2024"
        "</div>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════
#  FILTER DATA
# ════════════════════════════════════════════════════════════════════════════
if regime == "Pre-2020 (Stable)":
    mask = df.index < "2020-01-01"
elif regime == "Post-2020 (Crisis)":
    mask = df.index >= "2020-01-01"
else:
    mask = (df.index.year >= year_range[0]) & (df.index.year <= year_range[1])

dfs = df[mask].copy()
price_col = "Export Price (USD/kg)" if currency == "USD/kg" else "Export Price (LKR/kg)"
pfx = "$" if currency == "USD/kg" else "LKR "
n = len(dfs)

if n == 0:
    st.error("No data matches your filters. Adjust the year range or regime.")
    st.stop()

# CSV download
csv_bytes = dfs[[
    "Export Price (USD/kg)", "Export Price (LKR/kg)", "Global_Tea_Price ($)",
    "Export Quantity (kg)", "Exchange Rate (LKR/USD)",
    "Log_Return", "Vol_6M", "Vol_12M",
]].to_csv().encode("utf-8")

dl_placeholder.download_button(
    "⬇ Download Filtered CSV", csv_bytes,
    file_name="srilanka_tea_filtered.csv", mime="text/csv",
    use_container_width=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <h1>Volatility, Structural Changes &amp; Forecasting of Sri Lankan Tea Export Prices</h1>
  <p>University Dissertation Dashboard &nbsp;·&nbsp; Monthly Time Series Jan 2008 – Dec 2024
     &nbsp;·&nbsp; n = 204 observations &nbsp;·&nbsp; ARIMA(1,0,0) Forecast · Yule-Walker</p>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  § 01  EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr">
  <span class="sec-num">§ 01</span>
  <span class="sec-title">Executive Summary</span>
  <span class="sec-sub">Key Performance Indicators · Selected Period</span>
</div>""", unsafe_allow_html=True)

latest_p   = dfs[price_col].iloc[-1]
avg_p      = dfs[price_col].mean()
growth_pct = (dfs["Export Price (USD/kg)"].iloc[-1] /
              dfs["Export Price (USD/kg)"].iloc[0] - 1) * 100
avg_fx     = dfs["Exchange Rate (LKR/USD)"].mean()
vol12_vals = dfs["Vol_12M"].dropna()
latest_v   = float(vol12_vals.iloc[-1]) if len(vol12_vals) else 0.0

fmt_lp = f"${latest_p:.3f}" if currency == "USD/kg" else f"LKR {latest_p:,.0f}"
fmt_ap = f"${avg_p:.3f}"   if currency == "USD/kg" else f"LKR {avg_p:,.0f}"

def kpi(label, value, unit, delta, delta_cls="up", color=""):
    return (f'<div class="kpi-card {color}">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{value}</div>'
            f'<div class="kpi-unit">{unit}</div>'
            f'<div class="kpi-delta {delta_cls}">{delta}</div>'
            f'</div>')

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi("Latest Export Price", fmt_lp,
                f"{currency} · {dfs.index[-1]:%b %Y}",
                f"From {pfx}{dfs[price_col].iloc[0]:.3f}"), unsafe_allow_html=True)
c2.markdown(kpi("Period Average", fmt_ap, currency, "", color="orange"), unsafe_allow_html=True)
c3.markdown(kpi("Price Growth", f"+{growth_pct:.1f}%",
                f"{dfs.index[0]:%b %Y} → {dfs.index[-1]:%b %Y}",
                "$3.56 → $5.58 full period", "up", "green"), unsafe_allow_html=True)
c4.markdown(kpi("Avg Exchange Rate", f"{avg_fx:,.0f}", "LKR / USD",
                "108 (2008) → 296 (2024)", "dn"), unsafe_allow_html=True)
c5.markdown(kpi("Latest Volatility 12M", f"{latest_v:.1f}%", "Annualised % p.a.",
                "▼ Stabilising" if latest_v < 10 else "▲ Elevated",
                "up" if latest_v < 10 else "dn", "orange"), unsafe_allow_html=True)

# Auto insight
trend = "strong upward" if growth_pct > 30 else ("moderate upward" if growth_pct > 0 else "downward")
fx0, fx1 = dfs["Exchange Rate (LKR/USD)"].iloc[0], dfs["Exchange Rate (LKR/USD)"].iloc[-1]
st.markdown(f"""
<div class="insight-box">
  <div class="insight-label">📊 &nbsp; Auto-generated Analytical Insight</div>
  Over the <strong>{n}-month</strong> window
  (<strong>{dfs.index[0]:%b %Y} – {dfs.index[-1]:%b %Y}</strong>),
  Sri Lankan tea export prices recorded a <strong>{trend} trend of {growth_pct:+.1f}%</strong>,
  rising from <strong>${dfs['Export Price (USD/kg)'].iloc[0]:.3f}</strong> to
  <strong>${dfs['Export Price (USD/kg)'].iloc[-1]:.3f} USD/kg</strong>.
  The LKR/USD exchange rate depreciated from <strong>{fx0:.0f}</strong> to
  <strong>{fx1:.0f}</strong> (+{(fx1/fx0-1)*100:.0f}%),
  with a structural shock in April 2022 (+83% in a single month).
  Annualised 12-month volatility currently stands at <strong>{latest_v:.1f}% p.a.</strong>
  — down from the 2022 crisis peak of <strong>32.5%</strong>.
  The strongest price predictor is the Global Tea Price Index
  (<strong>r = 0.572</strong>), followed by exchange rate effects (<strong>r = 0.560</strong>).
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  § 02  TIME SERIES
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr">
  <span class="sec-num">§ 02</span>
  <span class="sec-title">Time Series Analysis</span>
  <span class="sec-sub">Zoom & pan · range selector · event annotations</span>
</div>""", unsafe_allow_html=True)

var_series = {
    "Export Price (USD/kg)":  ("Export Price (USD/kg)",   "USD/kg"),
    "Export Price (LKR/kg)":  ("Export Price (LKR/kg)",   "LKR/kg"),
    "Export Quantity (kg)":   ("Export Quantity (kg)",     "kg"),
    "Exchange Rate (LKR/USD)":("Exchange Rate (LKR/USD)", "LKR/USD"),
}
v_col, v_unit = var_series[variable]

fig_main = go.Figure()
fig_main.add_trace(go.Scatter(
    x=dfs.index, y=dfs[v_col], name=variable,
    line=dict(color=BLUE2, width=2),
    fill="tozeroy", fillcolor="rgba(43,92,168,0.07)",
    hovertemplate="<b>%{x|%b %Y}</b><br>" + variable + ": %{y:.3f}<extra></extra>",
))

for s, e, lbl, fc in [
    ("2008-09-01","2009-06-01","GFC 2008–09","rgba(164,32,32,0.08)"),
    ("2020-01-01","2020-12-01","COVID-19","rgba(201,93,26,0.08)"),
    ("2022-03-01","2022-12-01","LKR Crisis","rgba(164,32,32,0.12)"),
]:
    ts, te = pd.Timestamp(s), pd.Timestamp(e)
    if ts >= dfs.index[0] and te <= dfs.index[-1]:
        fig_main.add_vrect(x0=ts, x1=te, fillcolor=fc, layer="below",
                           line_width=0, annotation_text=lbl,
                           annotation_position="top left",
                           annotation_font=dict(size=8, color="#a42020"))

fig_main.update_layout(
    **PLOTLY_BASE,
    title=dict(text=f"<b>{variable}</b>", font=dict(size=13, color=BLUE), x=0),
    yaxis_title=v_unit, height=300, showlegend=False,
    xaxis=dict(**PLOTLY_BASE["xaxis"],
               rangeslider=dict(visible=True, thickness=0.05),
               rangeselector=dict(buttons=[
                   dict(count=2,  label="2Y",  step="year", stepmode="backward"),
                   dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                   dict(count=10, label="10Y", step="year", stepmode="backward"),
                   dict(step="all", label="All"),
               ], font=dict(size=10))),
)
st.plotly_chart(fig_main, use_container_width=True)
st.markdown("""<div style="margin-bottom:6px">
  <span class="chip chip-red">⚠ 2008 GFC</span>
  <span class="chip chip-orange">⚠ 2020 COVID</span>
  <span class="chip chip-red">⚠ 2022 LKR Crisis</span>
  <span class="chip chip-blue">↗ 2017–18 Peak</span>
  <span class="chip chip-green">↗ Post-2022 Recovery</span>
</div>""", unsafe_allow_html=True)

col_ts1, col_ts2 = st.columns(2)
with col_ts1:
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=dfs.index, y=dfs["Export Price (USD/kg)"],
                            name="Sri Lanka", line=dict(color=BLUE2, width=2),
                            hovertemplate="<b>%{x|%b %Y}</b><br>SL: $%{y:.3f}<extra></extra>"))
    f2.add_trace(go.Scatter(x=dfs.index, y=dfs["Global_Tea_Price ($)"],
                            name="Global", line=dict(color=ORANGE2, width=2, dash="dot"),
                            hovertemplate="<b>%{x|%b %Y}</b><br>Global: $%{y:.3f}<extra></extra>"))
    f2.update_layout(**PLOTLY_BASE,
                     title=dict(text="<b>SL vs Global Tea Price</b> · r = 0.572",
                                font=dict(size=12, color=BLUE), x=0),
                     height=255, yaxis_title="USD/kg")
    st.plotly_chart(f2, use_container_width=True)

with col_ts2:
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=dfs.index, y=dfs["Exchange Rate (LKR/USD)"],
                            name="LKR/USD", line=dict(color=ORANGE, width=2),
                            fill="tozeroy", fillcolor="rgba(201,93,26,0.07)",
                            hovertemplate="<b>%{x|%b %Y}</b><br>FX: LKR %{y:.1f}<extra></extra>"))
    f3.update_layout(**PLOTLY_BASE,
                     title=dict(text="<b>Exchange Rate (LKR/USD)</b> · Apr 2022: +83% shock",
                                font=dict(size=12, color=BLUE), x=0),
                     height=255, yaxis_title="LKR/USD", showlegend=False)
    st.plotly_chart(f3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  § 03  VOLATILITY
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr">
  <span class="sec-num">§ 03</span>
  <span class="sec-title">Volatility Analysis</span>
  <span class="sec-sub">Annualised rolling σ of log returns (% p.a.) · structural event bands</span>
</div>""", unsafe_allow_html=True)

fv = go.Figure()
for s, e, fc in [("2008-09-01","2009-06-01","rgba(164,32,32,0.07)"),
                  ("2011-12-01","2013-01-01","rgba(201,93,26,0.06)"),
                  ("2022-01-01","2022-12-01","rgba(164,32,32,0.12)")]:
    fv.add_vrect(x0=pd.Timestamp(s), x1=pd.Timestamp(e),
                 fillcolor=fc, layer="below", line_width=0)
fv.add_trace(go.Scatter(x=dfs.index, y=dfs["Vol_6M"], name="6-Month Vol",
                        line=dict(color=ORANGE, width=1.8),
                        hovertemplate="<b>%{x|%b %Y}</b><br>6M Vol: %{y:.2f}%<extra></extra>"))
fv.add_trace(go.Scatter(x=dfs.index, y=dfs["Vol_12M"], name="12-Month Vol",
                        line=dict(color=PURPLE, width=2.2),
                        hovertemplate="<b>%{x|%b %Y}</b><br>12M Vol: %{y:.2f}%<extra></extra>"))
fv.add_hline(y=15, line_dash="dash", line_color=RED, line_width=1,
             annotation_text="High-vol threshold (15%)", annotation_position="top right",
             annotation_font=dict(size=8, color=RED))
fv.update_layout(**PLOTLY_BASE,
                 title=dict(text="<b>6M vs 12M Rolling Volatility</b> · Annualised % p.a.",
                            font=dict(size=13, color=BLUE), x=0),
                 yaxis_title="Volatility (% p.a.)", height=300)
st.plotly_chart(fv, use_container_width=True)
st.markdown("""<div style="margin-bottom:6px">
  <span class="chip chip-red">2008–09: GFC ~24%</span>
  <span class="chip chip-orange">2012: 19% spike</span>
  <span class="chip chip-blue">2017–18: Prolonged ~9%</span>
  <span class="chip chip-red">2022: Crisis peak 32.5%</span>
  <span class="chip chip-green">2023–24: Stabilising</span>
</div>""", unsafe_allow_html=True)

cv1, cv2 = st.columns(2)
with cv1:
    rsd = dfs["Log_Return"].rolling(6).std()
    fr = go.Figure()
    fr.add_trace(go.Scatter(x=dfs.index, y=rsd, name="6M Std Dev",
                            line=dict(color=ORANGE2, width=1.8),
                            fill="tozeroy", fillcolor="rgba(232,121,58,0.09)",
                            hovertemplate="<b>%{x|%b %Y}</b><br>σ: %{y:.5f}<extra></extra>"))
    fr.update_layout(**PLOTLY_BASE,
                     title=dict(text="<b>Rolling Std Deviation</b> (Log Returns, 6M)",
                                font=dict(size=12, color=BLUE), x=0),
                     height=235, showlegend=False, yaxis_title="Std Dev")
    st.plotly_chart(fr, use_container_width=True)

with cv2:
    vrows = [
        ("Mean Vol 12M",  f"{pre_df['Vol_12M'].mean():.2f}%",  f"{post_df['Vol_12M'].mean():.2f}%"),
        ("Peak Vol 12M",  f"{pre_df['Vol_12M'].max():.2f}%",   f"{post_df['Vol_12M'].max():.2f}%"),
        ("Mean Vol 6M",   f"{pre_df['Vol_6M'].mean():.2f}%",   f"{post_df['Vol_6M'].mean():.2f}%"),
        ("Skewness",      f"{pre_df['Log_Return'].skew():.4f}",f"{post_df['Log_Return'].skew():.4f}"),
        ("Excess Kurtosis",f"{pre_df['Log_Return'].kurt():.4f}",f"{post_df['Log_Return'].kurt():.4f}"),
    ]
    tbl = ('<table class="stat-tbl"><thead>'
           '<tr><th>Metric</th><th>Pre-2020</th><th>Post-2020</th><th>Δ</th></tr>'
           '</thead><tbody>')
    for lbl, p, q in vrows:
        tbl += (f'<tr><td>{lbl}</td><td class="num">{p}</td>'
                f'<td class="num">{q}</td><td class="pos">↑ Higher</td></tr>')
    tbl += "</tbody></table>"
    st.markdown("**Volatility by Regime**")
    st.markdown(tbl, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  § 04  RELATIONSHIPS & CORRELATION
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr">
  <span class="sec-num">§ 04</span>
  <span class="sec-title">Relationship &amp; Correlation Analysis</span>
  <span class="sec-sub">OLS regression lines · interactive Pearson heatmap</span>
</div>""", unsafe_allow_html=True)

vld   = dfs.dropna(subset=["Export Price (USD/kg)", "Exchange Rate (LKR/USD)",
                             "Export Quantity (kg)", "Global_Tea_Price ($)"])
px_v  = vld["Export Price (USD/kg)"].values
fx_v  = vld["Exchange Rate (LKR/USD)"].values
qt_v  = vld["Export Quantity (kg)"].values / 1e6
gl_v  = vld["Global_Tea_Price ($)"].values

cs1, cs2, cs3 = st.columns(3)

def scatter_fig(xv, yv, xlabel, ylabel, color, title, r):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers",
                             marker=dict(color=color, size=5, opacity=0.5),
                             hovertemplate=f"{xlabel}: %{{x:.2f}}<br>{ylabel}: $%{{y:.3f}}<extra></extra>"))
    xs, ys = ols_line(xv, yv)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color=color, width=2),
                             name=f"OLS r={r:.3f}"))
    fig.update_layout(**PLOTLY_BASE,
                      title=dict(text=f"<b>{title}</b><br><sup>r = {r:.3f}</sup>",
                                 font=dict(size=11, color=BLUE), x=0),
                      height=245, showlegend=False,
                      xaxis_title=xlabel, yaxis_title=ylabel)
    return fig

cs1.plotly_chart(scatter_fig(fx_v, px_v, "Exchange Rate (LKR/USD)", "Price (USD/kg)",
                             BLUE2, "Price vs Exchange Rate", pearson_r(fx_v, px_v)),
                 use_container_width=True)
cs2.plotly_chart(scatter_fig(qt_v, px_v, "Quantity (M kg)", "Price (USD/kg)",
                             ORANGE, "Price vs Export Quantity", pearson_r(qt_v, px_v)),
                 use_container_width=True)
cs3.plotly_chart(scatter_fig(gl_v, px_v, "Global Price (USD/kg)", "SL Price (USD/kg)",
                             GREEN, "Price vs Global Tea Price", pearson_r(gl_v, px_v)),
                 use_container_width=True)

corr_cols = ["Export Price (USD/kg)", "Global_Tea_Price ($)",
             "Exchange Rate (LKR/USD)", "Export Quantity (kg)", "Log_Return"]
corr_labs = ["SL Price", "Global Price", "FX Rate", "Quantity", "Log Return"]
cmat = dfs[corr_cols].corr().values

fhm = go.Figure(go.Heatmap(
    z=cmat, x=corr_labs, y=corr_labs,
    colorscale=[[0,"#a42020"],[0.25,"#e07070"],[0.5,"#f5f5f5"],
                [0.75,"#7aabdc"],[1,"#1a3f6f"]],
    zmid=0, zmin=-1, zmax=1,
    text=[[f"{v:.3f}" for v in row] for row in cmat],
    texttemplate="%{text}", textfont=dict(size=12),
    hovertemplate="<b>%{y} vs %{x}</b><br>r = %{z:.4f}<extra></extra>",
    colorbar=dict(title="r", thickness=12, len=0.8),
))
fhm.update_layout(**PLOTLY_BASE,
                  title=dict(text="<b>Pearson Correlation Heatmap</b> — All Variables",
                             font=dict(size=13, color=BLUE), x=0),
                  height=310, hovermode="closest")
st.plotly_chart(fhm, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  § 05  DISTRIBUTION & RISK
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr">
  <span class="sec-num">§ 05</span>
  <span class="sec-title">Distribution &amp; Risk Analysis</span>
  <span class="sec-sub">Log returns · normality · tail risk · descriptive statistics</span>
</div>""", unsafe_allow_html=True)

lr_clean = dfs["Log_Return"].dropna()
mu_r, std_r = float(lr_clean.mean()), float(lr_clean.std())

cd1, cd2, cd3 = st.columns(3)
with cd1:
    bar_colors = [RED if v < -0.05 else (GREEN if v > 0.05 else BLUE2) for v in lr_clean]
    fb = go.Figure()
    fb.add_trace(go.Bar(x=dfs.index[dfs["Log_Return"].notna()], y=lr_clean,
                        marker_color=bar_colors,
                        hovertemplate="<b>%{x|%b %Y}</b><br>Return: %{y:.4f}<extra></extra>"))
    fb.add_hline(y=0, line_color=GREY, line_width=1)
    fb.update_layout(**PLOTLY_BASE,
                     title=dict(text="<b>Monthly Log Returns</b>",
                                font=dict(size=12, color=BLUE), x=0),
                     height=250, showlegend=False, yaxis_title="Log Return")
    st.plotly_chart(fb, use_container_width=True)

with cd2:
    fh = go.Figure()
    fh.add_trace(go.Histogram(x=lr_clean, nbinsx=22,
                              marker_color=BLUE2,
                              marker_line=dict(color="white", width=0.5),
                              opacity=0.8,
                              hovertemplate="Return: %{x:.4f}<br>Count: %{y}<extra></extra>",
                              name="Observed"))
    x_norm = np.linspace(float(lr_clean.min()), float(lr_clean.max()), 120)
    bin_w  = (float(lr_clean.max()) - float(lr_clean.min())) / 22
    y_norm = (np.exp(-0.5 * ((x_norm - mu_r) / std_r) ** 2)
              / (std_r * np.sqrt(2 * np.pi))) * len(lr_clean) * bin_w
    fh.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines",
                            line=dict(color=ORANGE, width=2, dash="dot"),
                            name="Normal approx."))
    fh.update_layout(**PLOTLY_BASE,
                     title=dict(text="<b>Log Returns Distribution</b>",
                                font=dict(size=12, color=BLUE), x=0),
                     height=250, xaxis_title="Log Return", yaxis_title="Frequency")
    st.plotly_chart(fh, use_container_width=True)

with cd3:
    q1 = float(lr_clean.quantile(0.25))
    q3 = float(lr_clean.quantile(0.75))
    sk = float(lr_clean.skew())
    ku = float(lr_clean.kurt())
    av = std_r * math.sqrt(12) * 100
    stat_rows = [
        ("Observations",   str(len(lr_clean)), "num"),
        ("Mean Return",    f"{mu_r*100:+.4f}%", "num"),
        ("Std Deviation",  f"{std_r*100:.4f}%", "num"),
        ("Annualised Vol", f"{av:.2f}%",         "num"),
        ("Minimum",        f"{lr_clean.min()*100:.4f}%", "neg"),
        ("Maximum",        f"{lr_clean.max()*100:.4f}%", "pos"),
        ("Skewness",       f"{sk:.4f}",  "neg" if sk < 0 else "pos"),
        ("Excess Kurtosis",f"{ku:.4f}",  "pos" if ku > 0 else "num"),
        ("Q1 (25th pct)",  f"{q1*100:.4f}%", "num"),
        ("Q3 (75th pct)",  f"{q3*100:.4f}%", "num"),
        ("IQR",            f"{(q3-q1)*100:.4f}%", "num"),
    ]
    tbl = '<table class="stat-tbl"><thead><tr><th>Statistic</th><th class="num">Value</th></tr></thead><tbody>'
    for lbl, val, cls in stat_rows:
        tbl += f'<tr><td>{lbl}</td><td class="{cls} num">{val}</td></tr>'
    tbl += "</tbody></table>"
    st.markdown("**Descriptive Statistics — Log Returns**")
    st.markdown(tbl, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  § 06  REGIME / STRUCTURAL BREAK
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""<div class="sec-hdr">
  <span class="sec-num">§ 06</span>
  <span class="sec-title">Structural Break &amp; Regime Analysis</span>
  <span class="sec-sub">Pre-2020 (n=144) vs Post-2020 (n=60) · break point Jan 2020</span>
</div>""", unsafe_allow_html=True)

creg1, creg2 = st.columns([2, 1])
with creg1:
    freg = make_subplots(specs=[[{"secondary_y": True}]])
    freg.add_vrect(x0=df.index[0], x1=pd.Timestamp("2020-01-01"),
                   fillcolor="rgba(26,63,111,0.07)", layer="below", line_width=0,
                   annotation_text="Pre-2020", annotation_position="top left",
                   annotation_font=dict(size=9, color=BLUE))
    freg.add_vrect(x0=pd.Timestamp("2020-01-01"), x1=df.index[-1],
                   fillcolor="rgba(201,93,26,0.08)", layer="below", line_width=0,
                   annotation_text="Post-2020", annotation_position="top right",
                   annotation_font=dict(size=9, color=ORANGE))
    freg.add_trace(go.Scatter(x=df.index, y=df["Export Price (USD/kg)"],
                              name="Export Price (USD/kg)",
                              line=dict(color=BLUE2, width=2),
                              hovertemplate="<b>%{x|%b %Y}</b><br>Price: $%{y:.3f}<extra></extra>"),
                   secondary_y=False)
    freg.add_trace(go.Scatter(x=df.index, y=df["Vol_12M"],
                              name="12M Volatility",
                              line=dict(color=ORANGE, width=1.5, dash="dot"),
                              hovertemplate="<b>%{x|%b %Y}</b><br>Vol: %{y:.2f}%<extra></extra>"),
                   secondary_y=True)
    pre_m  = float(pre_df["Export Price (USD/kg)"].mean())
    post_m = float(post_df["Export Price (USD/kg)"].mean())
    freg.add_hline(y=pre_m,  line_dash="dash", line_color=BLUE,   line_width=1,
                   annotation_text=f"Pre mean ${pre_m:.3f}",
                   annotation_position="right", annotation_font=dict(size=8,color=BLUE),
                   secondary_y=False)
    freg.add_hline(y=post_m, line_dash="dash", line_color=ORANGE, line_width=1,
                   annotation_text=f"Post mean ${post_m:.3f}",
                   annotation_position="right", annotation_font=dict(size=8,color=ORANGE),
                   secondary_y=False)
    freg.update_layout(**PLOTLY_BASE,
                       title=dict(text="<b>Regime Comparison</b> — Price & Volatility",
                                  font=dict(size=13, color=BLUE), x=0),
                       height=330)
    freg.update_yaxes(title_text="Export Price (USD/kg)", secondary_y=False,
                      showgrid=True, gridcolor="rgba(0,0,0,.05)")
    freg.update_yaxes(title_text="Volatility (% p.a.)", secondary_y=True, showgrid=False)
    st.plotly_chart(freg, use_container_width=True)

with creg2:
    def rpanel(title, df_r, cls, dot):
        items = {
            "Mean Price": f"${df_r['Export Price (USD/kg)'].mean():.4f}",
            "Max Price":  f"${df_r['Export Price (USD/kg)'].max():.4f}",
            "Min Price":  f"${df_r['Export Price (USD/kg)'].min():.4f}",
            "Std Dev":    f"${df_r['Export Price (USD/kg)'].std():.4f}",
            "Mean Vol":   f"{df_r['Vol_12M'].mean():.2f}%",
            "Avg Return": f"{df_r['Log_Return'].mean()*100:.4f}%",
        }
        cells = "".join(
            f'<div class="rp-stat"><div class="rp-stat-lbl">{k}</div>'
            f'<div class="rp-stat-val">{v}</div></div>'
            for k, v in items.items()
        )
        return (f'<div class="regime-panel {cls}">'
                f'<div class="rp-title"><span style="display:inline-block;width:8px;height:8px;'
                f'border-radius:50%;background:{dot};margin-right:7px"></span>{title}</div>'
                f'<div class="rp-grid">{cells}</div></div>')

    st.markdown(rpanel("Pre-2020 Regime",  pre_df,  "pre",  BLUE),   unsafe_allow_html=True)
    st.markdown(rpanel("Post-2020 Regime", post_df, "post", ORANGE), unsafe_allow_html=True)

    cp  = (post_df["Export Price (USD/kg)"].mean() /
           pre_df["Export Price (USD/kg)"].mean() - 1) * 100
    cv_ = (post_df["Vol_12M"].mean() / pre_df["Vol_12M"].mean() - 1) * 100
    csd = (post_df["Export Price (USD/kg)"].std() /
           pre_df["Export Price (USD/kg)"].std() - 1) * 100
    st.markdown(f"""<table class="stat-tbl" style="margin-top:8px">
      <thead><tr><th>Metric</th><th>Post vs Pre Δ</th></tr></thead>
      <tbody>
        <tr><td>Mean Price</td>      <td class="pos num">+{cp:.1f}%</td></tr>
        <tr><td>Mean Volatility</td> <td class="pos num">+{cv_:.1f}%</td></tr>
        <tr><td>Price Std Dev</td>   <td class="pos num">+{csd:.1f}%</td></tr>
        <tr><td>Excess Kurtosis</td> <td class="pos num">{post_df['Log_Return'].kurt():.2f} vs {pre_df['Log_Return'].kurt():.2f}</td></tr>
        <tr><td>Skewness</td>        <td class="neg num">{post_df['Log_Return'].skew():.4f} vs {pre_df['Log_Return'].skew():.4f}</td></tr>
      </tbody>
    </table>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  § 07  FORECASTING
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="sec-hdr">
  <span class="sec-num">§ 07</span>
  <span class="sec-title">Forecasting Module — ARIMA(1,0,0)</span>
  <span class="sec-sub">Yule-Walker · {fc_horizon}-month horizon · 95% confidence interval</span>
</div>""", unsafe_allow_html=True)

fc  = forecasts[fc_horizon]
his = df["Export Price (USD/kg)"].tail(48)

ffc = go.Figure()
ffc.add_trace(go.Scatter(
    x=his.index, y=his.values, name="Historical Price",
    line=dict(color=BLUE2, width=2.5),
    hovertemplate="<b>%{x|%b %Y}</b><br>Price: $%{y:.3f}<extra></extra>",
))
ffc.add_trace(go.Scatter(
    x=list(fc.index) + list(fc.index[::-1]),
    y=list(fc["upper"]) + list(fc["lower"][::-1]),
    fill="toself", fillcolor="rgba(30,122,75,0.15)",
    line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip",
))
ffc.add_trace(go.Scatter(
    x=fc.index, y=fc["mean"], name="ARIMA Forecast",
    line=dict(color=GREEN, width=2.5, dash="dash"),
    hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:.3f}<extra></extra>",
))
ffc.add_trace(go.Scatter(x=fc.index, y=fc["upper"], name="Upper 95% CI",
                         line=dict(color=GREEN, width=1, dash="dot"),
                         hovertemplate="Upper: $%{y:.3f}<extra></extra>"))
ffc.add_trace(go.Scatter(x=fc.index, y=fc["lower"], name="Lower 95% CI",
                         line=dict(color=RED, width=1, dash="dot"),
                         hovertemplate="Lower: $%{y:.3f}<extra></extra>"))
ffc.add_vline(x=his.index[-1].timestamp() * 1000,
              line_dash="dash", line_color=GREY, line_width=1.5,
              annotation_text="Forecast start", annotation_position="top",
              annotation_font=dict(size=9, color=GREY))
ffc.update_layout(**PLOTLY_BASE,
                  title=dict(
                      text=(f"<b>ARIMA(1,0,0) Price Forecast</b> · "
                            f"{fc_horizon}-Month Horizon · 95% Confidence Interval"),
                      font=dict(size=13, color=BLUE), x=0),
                  height=350, yaxis_title="Export Price (USD/kg)")
st.plotly_chart(ffc, use_container_width=True)

fc_end = fc["mean"].iloc[-1]
ci_w   = fc["upper"].iloc[-1] - fc["lower"].iloc[-1]
st.markdown(f"""
<div style="background:#f7f5f1;border:1px solid #d8d4cc;border-radius:8px;
            padding:10px 16px;margin-top:-4px;font-size:.73rem;color:#6e7180">
  <strong style="font-size:.66rem;text-transform:uppercase;letter-spacing:.5px;color:#9090a8">
    Model Diagnostics
  </strong><br>
  <span><strong>Model:</strong> ARIMA(1,0,0)</span> &nbsp;·&nbsp;
  <span><strong>Method:</strong> Yule-Walker</span> &nbsp;·&nbsp;
  <span><strong>Series:</strong> Log Returns → Price Reconstruction</span> &nbsp;·&nbsp;
  <span><strong>Last Observed:</strong> ${df['Export Price (USD/kg)'].iloc[-1]:.3f}/kg (Dec 2024)</span> &nbsp;·&nbsp;
  <span><strong>{fc_horizon}M Forecast:</strong>
    <span style="color:{GREEN};font-weight:700">${fc_end:.3f}/kg</span></span> &nbsp;·&nbsp;
  <span><strong>CI width at {fc_horizon}M:</strong> ${ci_w:.3f}</span> &nbsp;·&nbsp;
  <span><em>Note: Widening CI reflects long-horizon uncertainty</em></span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:.68rem;color:#9090a8;
            padding:12px 0;border-top:1px solid #d8d4cc;margin-top:8px">
  Sri Lankan Tea Export Price Dashboard &nbsp;·&nbsp; Dissertation Research Tool &nbsp;·&nbsp;
  Data: Tea_Export_Master_2008_2024 &amp; Global Tea Price Index &nbsp;·&nbsp;
  ARIMA(1,0,0) · Yule-Walker Estimation &nbsp;·&nbsp; Built with Streamlit &amp; Plotly
</div>""", unsafe_allow_html=True)
