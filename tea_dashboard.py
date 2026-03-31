#!/usr/bin/env python3
"""
================================================================================
  Sri Lankan Tea Export Price Research Dashboard
  "Volatility, Structural Changes, and Forecasting of Sri Lankan Tea Export Prices"
  University Dissertation — Interactive Dashboard
================================================================================

REQUIREMENTS
    pip install flask pandas numpy openpyxl

USAGE
    1.  Place this file in the same folder as your two Excel files:
            Tea_Export_Master_2008_2024.xlsx
            DateGlobal_Tea_Price.xlsx

    2.  Run:
            python tea_dashboard.py

    3.  Open your browser at:
            http://127.0.0.1:5050

NOTES
    • If you don't have the Excel files, the dashboard still runs with the
      pre-computed dataset that is embedded in the HTML.
    • To use a different port, run:  python tea_dashboard.py --port 8080
    • Press Ctrl+C to stop the server.
================================================================================
"""

import os
import sys
import json
import math
import argparse
import webbrowser
import threading
import warnings
warnings.filterwarnings("ignore")

# ── optional data-loading dependencies ──────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
    HAS_DATA_LIBS = True
except ImportError:
    HAS_DATA_LIBS = False

try:
    from flask import Flask, render_template_string, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# ════════════════════════════════════════════════════════════════════════════
#  DATA PROCESSING  (replicates your Colab notebook logic exactly)
# ════════════════════════════════════════════════════════════════════════════

def load_and_process(master_path: str, global_path: str) -> dict:
    """Load both Excel files, clean, merge, compute all derived series."""
    sl_df     = pd.read_excel(master_path)
    global_df = pd.read_excel(global_path)

    for d in [sl_df, global_df]:
        d["Month_num"] = pd.to_datetime(d["Month"], format="%B").dt.month
        d.index = pd.to_datetime(
            d["Year"].astype(str) + "-" + d["Month_num"].astype(str),
            format="%Y-%m"
        )
        d.drop(columns=["Month_num"], inplace=True)

    global_df = global_df[["Global_Tea_Price ($)"]]
    df = sl_df.merge(global_df, left_index=True, right_index=True, how="left")

    cols = [
        "Export Quantity (kg)",
        "Export Price (LKR/kg)",
        "Exchange Rate (LKR/USD)",
        "Export Price (USD/kg)",
        "Global_Tea_Price ($)",
    ]
    for col in cols:
        df[col] = pd.to_numeric(
            df[col].astype(str)
                   .str.replace("\n", "", regex=False)
                   .str.replace(",",  "", regex=False)
                   .str.strip(),
            errors="coerce",
        )

    df = df.sort_index()
    df[cols] = df[cols].ffill().bfill()

    df["Log_Return"] = np.log(
        df["Export Price (USD/kg)"] / df["Export Price (USD/kg)"].shift(1)
    )
    df["Vol_6M"]  = df["Log_Return"].rolling(6).std()  * np.sqrt(12) * 100
    df["Vol_12M"] = df["Log_Return"].rolling(12).std() * np.sqrt(12) * 100

    # ── ARIMA(1,0,0) via Yule-Walker (no statsmodels required) ──────────────
    returns  = df["Log_Return"].dropna().values
    mu       = returns.mean()
    c0       = np.var(returns)
    c1       = np.cov(returns[:-1], returns[1:])[0, 1]
    phi      = c1 / c0
    sigma2   = c0 * (1 - phi ** 2)

    def make_forecast(horizon: int) -> dict:
        fc_mean = []
        val = returns[-1]
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

        lp = float(df["Export Price (USD/kg)"].iloc[-1])
        pf, uf, lf = [], [], []
        p = u = l = lp
        for i in range(horizon):
            p = p * math.exp(fc_mean[i])
            u = u * math.exp(fc_upper[i])
            l = l * math.exp(fc_lower[i])
            pf.append(round(p, 5))
            uf.append(round(u, 5))
            lf.append(round(l, 5))

        li = df.index[-1]
        fc_labels = [
            pd.Timestamp(
                li.year + (li.month + i - 1) // 12,
                (li.month + i - 1) % 12 + 1,
                1,
            ).strftime("%b %Y")
            for i in range(1, horizon + 1)
        ]
        return {"mean": pf, "upper": uf, "lower": lf, "labels": fc_labels}

    forecasts = {6: make_forecast(6), 12: make_forecast(12), 24: make_forecast(24)}

    def c(v):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return round(float(v), 5)

    corr_cols = [
        "Export Price (USD/kg)", "Global_Tea_Price ($)",
        "Exchange Rate (LKR/USD)", "Export Quantity (kg)", "Log_Return",
    ]
    corr_matrix = df[corr_cols].corr().round(4).values.tolist()

    pre  = df[df.index < "2020-01-01"]
    post = df[df.index >= "2020-01-01"]

    def regime_stats(r):
        p  = r["Export Price (USD/kg)"].dropna()
        v  = r["Vol_12M"].dropna()
        lr = r["Log_Return"].dropna()
        return {
            "mean_price":  round(float(p.mean()),  4),
            "max_price":   round(float(p.max()),   4),
            "min_price":   round(float(p.min()),   4),
            "std_price":   round(float(p.std()),   4),
            "mean_vol":    round(float(v.mean()),  4),
            "max_vol":     round(float(v.max()),   4),
            "skew":        round(float(lr.skew()), 4),
            "kurt":        round(float(lr.kurt()), 4),
            "mean_return": round(float(lr.mean() * 100), 5),
        }

    data = {
        "labels":   [d.strftime("%b %Y") for d in df.index],
        "price":    [c(v) for v in df["Export Price (USD/kg)"]],
        "priceLKR": [c(v) for v in df["Export Price (LKR/kg)"]],
        "global":   [c(v) for v in df["Global_Tea_Price ($)"]],
        "quantity": [
            int(v) if v is not None and not math.isnan(float(v)) else None
            for v in df["Export Quantity (kg)"]
        ],
        "fx":       [c(v) for v in df["Exchange Rate (LKR/USD)"]],
        "logRet":   [c(v) for v in df["Log_Return"]],
        "vol6":     [c(v) for v in df["Vol_6M"]],
        "vol12":    [c(v) for v in df["Vol_12M"]],
        "forecasts": {str(k): v for k, v in forecasts.items()},
        "corr": {
            "labels": ["SL Price", "Global Price", "Exchange Rate", "Quantity", "Log Return"],
            "matrix": corr_matrix,
        },
        "regime": {
            "pre":  regime_stats(pre),
            "post": regime_stats(post),
        },
        "meta": {
            "first_price":  round(float(df["Export Price (USD/kg)"].iloc[0]),  4),
            "last_price":   round(float(df["Export Price (USD/kg)"].iloc[-1]), 4),
            "max_price":    round(float(df["Export Price (USD/kg)"].max()),    4),
            "min_price":    round(float(df["Export Price (USD/kg)"].min()),    4),
            "avg_price":    round(float(df["Export Price (USD/kg)"].mean()),   4),
            "latest_vol12": round(float(df["Vol_12M"].dropna().iloc[-1]),      4),
            "avg_fx":       round(float(df["Exchange Rate (LKR/USD)"].mean()), 4),
            "price_growth": round(float(
                (df["Export Price (USD/kg)"].iloc[-1]
                 / df["Export Price (USD/kg)"].iloc[0] - 1) * 100
            ), 2),
            "n": len(df),
        },
    }
    return data


# ════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML  (self-contained — no external files needed)
# ════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Sri Lankan Tea Export Price Research Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
/* ── TOKENS ─────────────────────────────────────────── */
:root{
  --bg:#f0ede8;
  --surface:#ffffff;
  --surface2:#f7f5f1;
  --border:#d8d4cc;
  --border2:#ebe7e0;
  --ink:#1c1c2e;
  --ink2:#4a4a62;
  --ink3:#8a8a9e;
  --blue:#1a3f6f;
  --blue2:#2b5ca8;
  --blue-lt:#e6eef8;
  --blue-mid:#c3d6ef;
  --orange:#c95d1a;
  --orange-lt:#faeee5;
  --orange2:#e8793a;
  --green:#1e7a4b;
  --green-lt:#e4f4ec;
  --red:#a42020;
  --red-lt:#faeaea;
  --purple:#5b3f9c;
  --grey:#6e7180;
  --shadow-sm:0 1px 4px rgba(0,0,0,.07);
  --shadow:0 3px 14px rgba(0,0,0,.09);
  --shadow-lg:0 8px 32px rgba(0,0,0,.13);
  --r:8px;
  --r2:12px;
}

/* ── RESET ──────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.55}

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--border2)}
::-webkit-scrollbar-thumb{background:var(--blue-mid);border-radius:3px}

/* ── HEADER ─────────────────────────────────────────── */
#header{
  background:var(--blue);
  color:#fff;
  padding:0;
  border-bottom:3px solid var(--orange);
  position:sticky;top:0;z-index:200;
}
.header-inner{
  max-width:1500px;margin:0 auto;
  display:flex;align-items:stretch;gap:0;
}
.header-brand{
  padding:18px 32px 16px;
  border-right:1px solid rgba(255,255,255,.12);
  flex-shrink:0;
}
.header-brand h1{
  font-family:'Libre Baskerville',serif;
  font-size:1.05rem;font-weight:700;line-height:1.25;
  letter-spacing:.01em;
}
.header-brand .sub{
  font-size:.72rem;opacity:.65;margin-top:3px;letter-spacing:.3px;
}
.header-nav{
  display:flex;align-items:center;gap:2px;padding:0 20px;flex:1;
  overflow-x:auto;
}
.nav-btn{
  background:none;border:none;color:rgba(255,255,255,.65);
  padding:8px 14px;border-radius:6px;cursor:pointer;
  font-family:'DM Sans',sans-serif;font-size:.78rem;font-weight:500;
  white-space:nowrap;transition:all .18s;letter-spacing:.2px;
}
.nav-btn:hover,.nav-btn.active{background:rgba(255,255,255,.12);color:#fff}
.header-actions{
  display:flex;align-items:center;gap:8px;padding:0 20px;
  border-left:1px solid rgba(255,255,255,.12);
}
.hbtn{
  padding:6px 12px;border-radius:6px;font-size:.75rem;font-weight:600;
  cursor:pointer;border:none;font-family:inherit;transition:all .18s;
  letter-spacing:.3px;white-space:nowrap;
}
.hbtn-o{background:rgba(255,255,255,.12);color:#fff;border:1px solid rgba(255,255,255,.25)}
.hbtn-o:hover{background:rgba(255,255,255,.22)}
.hbtn-primary{background:var(--orange);color:#fff}
.hbtn-primary:hover{background:#b34e12}
#fsBtn{background:rgba(255,255,255,.1);color:#fff;border:1px solid rgba(255,255,255,.2)}

/* ── CONTROLS BAR ───────────────────────────────────── */
#ctrlbar{
  background:var(--surface);
  border-bottom:1px solid var(--border);
  padding:10px 0;
  box-shadow:var(--shadow-sm);
}
.ctrl-inner{
  max-width:1500px;margin:0 auto;padding:0 28px;
  display:flex;gap:20px;align-items:center;flex-wrap:wrap;
}
.cg{display:flex;align-items:center;gap:8px}
.cg label{
  font-size:.72rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);
  white-space:nowrap;
}
.sw{display:flex;align-items:center;gap:6px}
.sw span{font-size:.78rem;color:var(--ink2);min-width:30px;font-weight:500}
input[type=range]{width:100px;accent-color:var(--blue2);cursor:pointer}
select,.cselect{
  border:1px solid var(--border);border-radius:6px;
  padding:5px 10px;font-size:.78rem;font-family:inherit;
  background:var(--surface2);color:var(--ink);cursor:pointer;outline:none;
}
select:focus{border-color:var(--blue2)}
.toggle-group{display:flex;gap:2px;background:var(--surface2);border:1px solid var(--border);border-radius:7px;padding:2px}
.tgl{
  background:none;border:none;padding:5px 12px;border-radius:5px;
  font-size:.75rem;font-weight:500;font-family:inherit;cursor:pointer;
  color:var(--ink2);transition:all .18s;white-space:nowrap;
}
.tgl.on{background:var(--blue);color:#fff}
.currency-toggle{display:flex;gap:2px;background:var(--surface2);border:1px solid var(--border);border-radius:7px;padding:2px}
.ctgl{
  background:none;border:none;padding:4px 10px;border-radius:5px;
  font-size:.72rem;font-weight:600;font-family:inherit;cursor:pointer;
  color:var(--ink3);transition:all .15s;letter-spacing:.3px;
}
.ctgl.on{background:var(--orange);color:#fff}

/* ── MAIN ───────────────────────────────────────────── */
main{max-width:1500px;margin:0 auto;padding:24px 28px 60px}

/* ── SECTION ────────────────────────────────────────── */
.section{margin-bottom:32px;scroll-margin-top:90px}
.sec-hdr{
  display:flex;align-items:baseline;gap:10px;
  margin-bottom:16px;padding-bottom:10px;
  border-bottom:2px solid var(--border);
}
.sec-num{
  font-family:'Libre Baskerville',serif;font-size:.7rem;font-weight:700;
  color:var(--blue2);letter-spacing:1px;text-transform:uppercase;
  background:var(--blue-lt);padding:2px 8px;border-radius:20px;flex-shrink:0;
}
.sec-title{
  font-family:'Libre Baskerville',serif;font-size:1rem;font-weight:700;
  color:var(--ink);
}
.sec-subtitle{font-size:.75rem;color:var(--ink3);margin-left:auto;font-style:italic}

/* ── KPI GRID ───────────────────────────────────────── */
.kpi-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}
.kpi{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r2);padding:18px 18px 15px;
  box-shadow:var(--shadow-sm);
  border-top:3px solid var(--blue);
  transition:box-shadow .2s,transform .2s;
  position:relative;overflow:hidden;
}
.kpi:hover{box-shadow:var(--shadow);transform:translateY(-1px)}
.kpi::before{
  content:'';position:absolute;top:-12px;right:-12px;
  width:60px;height:60px;border-radius:50%;
  background:var(--blue-lt);opacity:.6;
}
.kpi.orange{border-top-color:var(--orange)}
.kpi.orange::before{background:var(--orange-lt)}
.kpi.green{border-top-color:var(--green)}
.kpi.green::before{background:var(--green-lt)}
.kpi-lbl{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.7px;color:var(--ink3);margin-bottom:6px}
.kpi-val{font-family:'Libre Baskerville',serif;font-size:1.75rem;font-weight:700;color:var(--blue);line-height:1}
.kpi.orange .kpi-val{color:var(--orange)}
.kpi.green .kpi-val{color:var(--green)}
.kpi-unit{font-size:.7rem;color:var(--ink3);margin-top:3px}
.kpi-delta{font-size:.72rem;font-weight:600;margin-top:6px;display:flex;align-items:center;gap:3px}
.kpi-delta.up{color:var(--green)}
.kpi-delta.dn{color:var(--red)}

/* ── INSIGHT BOX ────────────────────────────────────── */
#insightBox{
  background:linear-gradient(135deg,var(--blue) 0%,#1e5090 100%);
  color:#fff;border-radius:var(--r2);padding:18px 22px;
  box-shadow:var(--shadow);margin-top:14px;
  display:flex;align-items:flex-start;gap:14px;
}
.insight-icon{font-size:1.4rem;flex-shrink:0;margin-top:1px}
.insight-label{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;opacity:.65;margin-bottom:4px}
#insightText{font-size:.84rem;line-height:1.6;opacity:.92}

/* ── CHART CARDS ────────────────────────────────────── */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.g31{display:grid;grid-template-columns:2fr 1fr;gap:16px}
.g13{display:grid;grid-template-columns:1fr 2fr;gap:16px}

.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r2);padding:18px 18px 14px;
  box-shadow:var(--shadow-sm);
}
.card-hdr{
  display:flex;align-items:flex-start;justify-content:space-between;
  margin-bottom:12px;gap:10px;
}
.card-title{font-size:.78rem;font-weight:600;color:var(--ink2);text-transform:uppercase;letter-spacing:.4px;line-height:1.3}
.card-note{font-size:.68rem;color:var(--ink3);font-style:italic;margin-top:2px}
.card-badge{
  font-size:.62rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;
  padding:2px 8px;border-radius:20px;white-space:nowrap;flex-shrink:0;
}
.badge-blue{background:var(--blue-lt);color:var(--blue2)}
.badge-orange{background:var(--orange-lt);color:var(--orange)}
.badge-green{background:var(--green-lt);color:var(--green)}

.cw canvas{max-height:240px;width:100%!important}
.cw.tall canvas{max-height:290px}
.cw.short canvas{max-height:175px}
.cw.med canvas{max-height:210px}

/* ── LEGEND ─────────────────────────────────────────── */
.leg{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px}
.li{display:flex;align-items:center;gap:5px;font-size:.7rem;color:var(--ink2)}
.ll{width:18px;height:2.5px;border-radius:2px;flex-shrink:0}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.lsq{width:12px;height:9px;border-radius:3px;flex-shrink:0}

/* ── ANNOTATION BAND ────────────────────────────────── */
.anno-strip{
  display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;
}
.anno-chip{
  font-size:.67rem;font-weight:600;padding:3px 10px;border-radius:20px;
  letter-spacing:.3px;cursor:default;
}
.chip-red{background:var(--red-lt);color:var(--red);border:1px solid #e8b0b0}
.chip-orange{background:var(--orange-lt);color:var(--orange);border:1px solid #f5c9a8}
.chip-blue{background:var(--blue-lt);color:var(--blue2);border:1px solid var(--blue-mid)}
.chip-green{background:var(--green-lt);color:var(--green);border:1px solid #a8dbc0}

/* ── REGIME COMPARISON ──────────────────────────────── */
.regime-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:4px}
.regime-panel{
  border-radius:var(--r);padding:16px 18px;
  border:1px solid var(--border);
}
.regime-panel.pre{background:linear-gradient(135deg,#f0f5fb 0%,var(--surface) 100%)}
.regime-panel.post{background:linear-gradient(135deg,#fdf3ec 0%,var(--surface) 100%)}
.rp-title{
  font-family:'Libre Baskerville',serif;font-size:.82rem;font-weight:700;
  margin-bottom:10px;display:flex;align-items:center;gap:8px;
}
.rp-title .dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.rp-stats{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.rp-stat{background:rgba(255,255,255,.7);border-radius:6px;padding:8px 10px}
.rp-stat-lbl{font-size:.65rem;color:var(--ink3);font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.rp-stat-val{font-size:.9rem;font-weight:700;color:var(--ink);font-family:'Libre Baskerville',serif}
.regime-diff{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r);padding:14px 16px;margin-top:12px;
}
.regime-diff-title{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--ink3);margin-bottom:8px}
.diff-row{display:flex;align-items:center;gap:10px;margin-bottom:5px;font-size:.76rem}
.diff-lbl{color:var(--ink2);flex:0 0 130px}
.diff-bar-wrap{flex:1;background:var(--border2);border-radius:3px;height:5px;overflow:hidden}
.diff-bar{height:100%;border-radius:3px;transition:width .8s ease}
.diff-val{font-weight:600;color:var(--ink);min-width:55px;text-align:right;font-size:.73rem}
.diff-delta{font-size:.68rem;font-weight:700;padding:1px 6px;border-radius:4px;white-space:nowrap}
.delta-up{background:var(--green-lt);color:var(--green)}
.delta-dn{background:var(--red-lt);color:var(--red)}

/* ── CORR HEATMAP ───────────────────────────────────── */
#corrTable{width:100%;border-collapse:collapse;font-size:.72rem}
#corrTable th{
  padding:6px 8px;text-align:center;font-weight:600;color:var(--ink2);
  font-size:.67rem;letter-spacing:.3px;
}
#corrTable td{
  padding:5px 8px;text-align:center;font-weight:600;border:1px solid rgba(255,255,255,.4);
  font-size:.75rem;cursor:default;transition:opacity .15s;
}
#corrTable td:hover{opacity:.75}

/* ── STAT TABLE ─────────────────────────────────────── */
.stat-table{width:100%;border-collapse:collapse;font-size:.75rem;margin-top:8px}
.stat-table th{
  padding:7px 10px;text-align:left;font-weight:600;color:var(--ink3);
  font-size:.67rem;text-transform:uppercase;letter-spacing:.5px;
  border-bottom:1px solid var(--border);
}
.stat-table td{padding:6px 10px;border-bottom:1px solid var(--border2);color:var(--ink2)}
.stat-table tr:last-child td{border-bottom:none}
.stat-table td:first-child{color:var(--ink);font-weight:500}
.stat-table .num{text-align:right;font-variant-numeric:tabular-nums;font-weight:600;color:var(--ink)}
.stat-table .pos{color:var(--green);font-weight:700}
.stat-table .neg{color:var(--red);font-weight:700}

/* ── FORECAST CONTROLS ──────────────────────────────── */
.fc-controls{display:flex;gap:8px;margin-bottom:12px;align-items:center}
.fc-lbl{font-size:.72rem;font-weight:600;color:var(--ink3);text-transform:uppercase;letter-spacing:.5px}
.fc-btn{
  padding:5px 14px;border-radius:6px;border:1px solid var(--border);
  background:var(--surface2);color:var(--ink2);font-size:.75rem;font-weight:600;
  cursor:pointer;font-family:inherit;transition:all .18s;
}
.fc-btn.active{background:var(--blue);color:#fff;border-color:var(--blue)}

/* ── FULLSCREEN ──────────────────────────────────────── */
#app:fullscreen,#app:-webkit-full-screen{
  background:var(--bg);overflow:auto;
}

/* ── FOOTER ─────────────────────────────────────────── */
footer{
  background:var(--ink);color:rgba(255,255,255,.45);
  text-align:center;padding:14px;font-size:.7rem;letter-spacing:.3px;
}

/* ── RESPONSIVE ─────────────────────────────────────── */
@media(max-width:1100px){
  .kpi-grid{grid-template-columns:repeat(3,1fr)}
  .g3{grid-template-columns:1fr 1fr}
  .g31,.g13{grid-template-columns:1fr}
}
@media(max-width:760px){
  .kpi-grid{grid-template-columns:1fr 1fr}
  .g2,.g3,.g31,.g13{grid-template-columns:1fr}
  main,.ctrl-inner{padding-left:14px;padding-right:14px}
  .header-nav{display:none}
}
</style>
</head>
<body>
<div id="app">

<!-- ══ HEADER ══════════════════════════════════════════ -->
<div id="header">
  <div class="header-inner">
    <div class="header-brand">
      <h1>Sri Lankan Tea Export Price Analysis</h1>
      <div class="sub">Dissertation Dashboard &nbsp;·&nbsp; Jan 2008 – Dec 2024 &nbsp;·&nbsp; n = 204 observations</div>
    </div>
    <div class="header-nav">
      <button class="nav-btn active" onclick="scrollTo('s1')">Overview</button>
      <button class="nav-btn" onclick="scrollTo('s2')">Time Series</button>
      <button class="nav-btn" onclick="scrollTo('s3')">Volatility</button>
      <button class="nav-btn" onclick="scrollTo('s4')">Relationships</button>
      <button class="nav-btn" onclick="scrollTo('s5')">Distributions</button>
      <button class="nav-btn" onclick="scrollTo('s6')">Regime</button>
      <button class="nav-btn" onclick="scrollTo('s7')">Forecast</button>
    </div>
    <div class="header-actions">
      <button class="hbtn hbtn-o" onclick="exportPNG()">⬇ PNG</button>
      <button class="hbtn hbtn-primary" onclick="downloadCSV()">⬇ CSV</button>
      <button class="hbtn hbtn-o" id="fsBtn" onclick="toggleFS()">⛶ Full</button>
    </div>
  </div>
</div>

<!-- ══ CONTROLS ════════════════════════════════════════ -->
<div id="ctrlbar">
  <div class="ctrl-inner">
    <div class="cg">
      <label>Year Range</label>
      <div class="sw">
        <span id="yfL">2008</span>
        <input type="range" id="yf" min="2008" max="2024" value="2008">
        <span>–</span>
        <input type="range" id="yt" min="2008" max="2024" value="2024">
        <span id="ytL">2024</span>
      </div>
    </div>
    <div class="cg">
      <label>Regime</label>
      <div class="toggle-group">
        <button class="tgl on" id="tAll" onclick="setRegime('all')">All</button>
        <button class="tgl" id="tPre" onclick="setRegime('pre')">Pre-2020</button>
        <button class="tgl" id="tPost" onclick="setRegime('post')">Post-2020</button>
      </div>
    </div>
    <div class="cg">
      <label>Variable</label>
      <select id="varSel">
        <option value="price">Export Price (USD/kg)</option>
        <option value="quantity">Export Quantity</option>
        <option value="fx">Exchange Rate</option>
        <option value="priceLKR">Export Price (LKR/kg)</option>
      </select>
    </div>
    <div class="cg">
      <label>Currency</label>
      <div class="currency-toggle">
        <button class="ctgl on" id="cUSD" onclick="setCurrency('USD')">USD</button>
        <button class="ctgl" id="cLKR" onclick="setCurrency('LKR')">LKR</button>
      </div>
    </div>
    <div class="cg" style="margin-left:auto">
      <label>Annotations</label>
      <label style="display:flex;align-items:center;gap:5px;cursor:pointer;font-size:.78rem;color:var(--ink2)">
        <input type="checkbox" id="annoChk" checked> Show Events
      </label>
    </div>
  </div>
</div>

<!-- ══ MAIN ════════════════════════════════════════════ -->
<main id="mainContent">

  <!-- §1 EXECUTIVE SUMMARY ─────────────────────────── -->
  <div class="section" id="s1">
    <div class="sec-hdr">
      <span class="sec-num">§ 01</span>
      <span class="sec-title">Executive Summary</span>
      <span class="sec-subtitle">Key Performance Indicators · 2008–2024</span>
    </div>

    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-lbl">Latest Export Price</div>
        <div class="kpi-val" id="k1">—</div>
        <div class="kpi-unit" id="k1u">USD per kg · Dec 2024</div>
        <div class="kpi-delta up" id="k1d">—</div>
      </div>
      <div class="kpi orange">
        <div class="kpi-lbl">Period Average Price</div>
        <div class="kpi-val" id="k2">—</div>
        <div class="kpi-unit" id="k2u">USD per kg</div>
        <div class="kpi-delta" id="k2d">—</div>
      </div>
      <div class="kpi green">
        <div class="kpi-lbl">Price Growth (Full Period)</div>
        <div class="kpi-val" id="k3">—</div>
        <div class="kpi-unit">Jan 2008 → Dec 2024</div>
        <div class="kpi-delta up" id="k3d">$3.56 → $5.58</div>
      </div>
      <div class="kpi">
        <div class="kpi-lbl">Avg Exchange Rate</div>
        <div class="kpi-val" id="k4">—</div>
        <div class="kpi-unit">LKR per USD</div>
        <div class="kpi-delta dn" id="k4d">—</div>
      </div>
      <div class="kpi orange">
        <div class="kpi-lbl">Latest Volatility (12M)</div>
        <div class="kpi-val" id="k5">—</div>
        <div class="kpi-unit">Annualised % p.a.</div>
        <div class="kpi-delta" id="k5d">—</div>
      </div>
    </div>

    <div id="insightBox">
      <div class="insight-icon">📊</div>
      <div>
        <div class="insight-label">Analytical Insight</div>
        <div id="insightText">Loading…</div>
      </div>
    </div>
  </div>

  <!-- §2 TIME SERIES ───────────────────────────────── -->
  <div class="section" id="s2">
    <div class="sec-hdr">
      <span class="sec-num">§ 02</span>
      <span class="sec-title">Time Series Analysis</span>
      <span class="sec-subtitle">Multi-variable price dynamics over time</span>
    </div>

    <div class="card" style="margin-bottom:16px">
      <div class="card-hdr">
        <div>
          <div class="card-title" id="tsTitle">Sri Lanka Export Price (USD/kg)</div>
          <div class="card-note">Monthly observations · toggle variable via controls bar</div>
        </div>
        <span class="card-badge badge-blue">PRIMARY SERIES</span>
      </div>
      <div class="cw tall"><canvas id="cMain"></canvas></div>
      <div class="anno-strip" id="annoStrip1">
        <span class="anno-chip chip-red">⚠ 2008 GFC</span>
        <span class="anno-chip chip-orange">⚠ 2020 COVID</span>
        <span class="anno-chip chip-red">⚠ 2022 LKR Crisis</span>
        <span class="anno-chip chip-blue">↗ 2017–18 Price Peak</span>
        <span class="anno-chip chip-green">↗ Post-2022 Recovery</span>
      </div>
    </div>

    <div class="g2">
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Sri Lanka vs Global Tea Price</div>
            <div class="card-note">USD/kg · correlation = 0.572</div>
          </div>
          <span class="card-badge badge-orange">DUAL SERIES</span>
        </div>
        <div class="cw med"><canvas id="cDual"></canvas></div>
        <div class="leg">
          <div class="li"><div class="ll" style="background:var(--blue2)"></div>Sri Lanka</div>
          <div class="li"><div class="ll" style="background:var(--orange2)"></div>Global</div>
        </div>
      </div>
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Exchange Rate (LKR/USD)</div>
            <div class="card-note">Gradual depreciation · severe shock in Apr 2022</div>
          </div>
          <span class="card-badge badge-orange">FX SERIES</span>
        </div>
        <div class="cw med"><canvas id="cFX"></canvas></div>
        <div class="anno-strip">
          <span class="anno-chip chip-red">Apr 2022: 203→371 LKR (+83%)</span>
        </div>
      </div>
    </div>
  </div>

  <!-- §3 VOLATILITY ───────────────────────────────── -->
  <div class="section" id="s3">
    <div class="sec-hdr">
      <span class="sec-num">§ 03</span>
      <span class="sec-title">Volatility Analysis</span>
      <span class="sec-subtitle">Annualised rolling standard deviation of log returns</span>
    </div>

    <div class="card" style="margin-bottom:16px">
      <div class="card-hdr">
        <div>
          <div class="card-title">6-Month vs 12-Month Rolling Volatility (% p.a.)</div>
          <div class="card-note">Higher values indicate increased price uncertainty · annotated structural events</div>
        </div>
        <span class="card-badge badge-orange">CORE FEATURE</span>
      </div>
      <div class="cw tall"><canvas id="cVol"></canvas></div>
      <div class="leg">
        <div class="li"><div class="ll" style="background:var(--orange)"></div>6-Month Volatility</div>
        <div class="li"><div class="ll" style="background:var(--purple)"></div>12-Month Volatility</div>
      </div>
      <div class="anno-strip">
        <span class="anno-chip chip-red">2008–09: GFC spike (~24%)</span>
        <span class="anno-chip chip-orange">2012: 19% spike</span>
        <span class="anno-chip chip-blue">2017–18: Prolonged elevation</span>
        <span class="anno-chip chip-red">2022: Crisis peak (32.5%)</span>
        <span class="anno-chip chip-green">2023–24: Gradual stabilisation</span>
      </div>
    </div>

    <div class="g2">
      <div class="card">
        <div class="card-hdr">
          <div class="card-title">Rolling Standard Deviation (Log Returns)</div>
          <div class="card-note">6-month window · raw scale</div>
        </div>
        <div class="cw med"><canvas id="cRSD"></canvas></div>
      </div>
      <div class="card">
        <div class="card-hdr">
          <div class="card-title">Volatility Summary Statistics</div>
          <div class="card-note">By regime period</div>
        </div>
        <table class="stat-table">
          <thead><tr><th>Metric</th><th>Pre-2020</th><th>Post-2020</th><th>Change</th></tr></thead>
          <tbody>
            <tr><td>Mean Vol (12M)</td><td class="num">9.19%</td><td class="num">13.44%</td><td class="num pos">+46.3%</td></tr>
            <tr><td>Peak Vol (12M)</td><td class="num">21.48%</td><td class="num">23.18%</td><td class="num pos">+7.9%</td></tr>
            <tr><td>Mean Vol (6M)</td><td class="num">8.64%</td><td class="num">11.83%</td><td class="num pos">+36.9%</td></tr>
            <tr><td>Std of Vol</td><td class="num">4.12%</td><td class="num">5.87%</td><td class="num pos">+42.5%</td></tr>
            <tr><td>Skewness</td><td class="num">-0.55</td><td class="num">-1.15</td><td class="num neg">↑ Neg. skew</td></tr>
            <tr><td>Excess Kurtosis</td><td class="num">1.87</td><td class="num">5.35</td><td class="num pos">Fatter tails</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- §4 RELATIONSHIPS ────────────────────────────── -->
  <div class="section" id="s4">
    <div class="sec-hdr">
      <span class="sec-num">§ 04</span>
      <span class="sec-title">Relationship &amp; Correlation Analysis</span>
      <span class="sec-subtitle">Bivariate scatter plots with OLS regression · dynamic correlation heatmap</span>
    </div>

    <div class="g3" style="margin-bottom:16px">
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Price vs Exchange Rate</div>
            <div class="card-note">r = 0.560 · moderate positive</div>
          </div>
        </div>
        <div class="cw med"><canvas id="cSc1"></canvas></div>
      </div>
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Price vs Export Quantity</div>
            <div class="card-note">r = −0.331 · inverse relationship</div>
          </div>
        </div>
        <div class="cw med"><canvas id="cSc2"></canvas></div>
      </div>
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Price vs Global Tea Price</div>
            <div class="card-note">r = 0.572 · strongest driver</div>
          </div>
        </div>
        <div class="cw med"><canvas id="cSc3"></canvas></div>
      </div>
    </div>

    <div class="card">
      <div class="card-hdr">
        <div>
          <div class="card-title">Correlation Heatmap — All Variables</div>
          <div class="card-note">Pearson correlation coefficients · full sample 2008–2024</div>
        </div>
        <span class="card-badge badge-blue">INTERACTIVE</span>
      </div>
      <div style="overflow-x:auto">
        <table id="corrTable"></table>
      </div>
      <div style="margin-top:10px;display:flex;align-items:center;gap:6px">
        <span style="font-size:.68rem;color:var(--ink3);font-style:italic">Colour scale:</span>
        <div style="display:flex;gap:0;border-radius:3px;overflow:hidden;height:12px;width:160px">
          <div style="flex:1;background:#c0392b"></div>
          <div style="flex:1;background:#e67e22"></div>
          <div style="flex:1;background:#f5f5f5"></div>
          <div style="flex:1;background:#2980b9"></div>
          <div style="flex:1;background:#1a3f6f"></div>
        </div>
        <span style="font-size:.67rem;color:var(--ink3)">−1 (red) → 0 (grey) → +1 (blue)</span>
      </div>
    </div>
  </div>

  <!-- §5 DISTRIBUTIONS ────────────────────────────── -->
  <div class="section" id="s5">
    <div class="sec-hdr">
      <span class="sec-num">§ 05</span>
      <span class="sec-title">Distribution &amp; Risk Analysis</span>
      <span class="sec-subtitle">Log returns distribution · normality assessment · tail risk</span>
    </div>

    <div class="g3">
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Log Returns — Time Series</div>
            <div class="card-note">Monthly · notable –17% shock in Apr 2022</div>
          </div>
        </div>
        <div class="cw med"><canvas id="cRet"></canvas></div>
      </div>
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Log Returns Distribution</div>
            <div class="card-note">Frequency histogram · 22 bins</div>
          </div>
          <span class="card-badge badge-blue">HISTOGRAM</span>
        </div>
        <div class="cw med"><canvas id="cHist"></canvas></div>
      </div>
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Descriptive Statistics</div>
            <div class="card-note">Full period log returns</div>
          </div>
        </div>
        <table class="stat-table" id="distStats">
          <thead><tr><th>Statistic</th><th class="num">Value</th><th>Interpretation</th></tr></thead>
          <tbody id="distStatBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- §6 REGIME ───────────────────────────────────── -->
  <div class="section" id="s6">
    <div class="sec-hdr">
      <span class="sec-num">§ 06</span>
      <span class="sec-title">Structural Break &amp; Regime Analysis</span>
      <span class="sec-subtitle">Pre-2020 (n=144) vs Post-2020 (n=60) comparative analysis</span>
    </div>

    <div class="g31">
      <div class="card">
        <div class="card-hdr">
          <div>
            <div class="card-title">Regime Comparison — Price &amp; Volatility</div>
            <div class="card-note">Visual structural break at January 2020 · shaded regions</div>
          </div>
          <span class="card-badge badge-blue">REGIME BREAK</span>
        </div>
        <div class="cw tall"><canvas id="cRegime"></canvas></div>
        <div class="leg">
          <div class="li"><div class="lsq" style="background:rgba(26,63,111,.15)"></div>Pre-2020 Regime</div>
          <div class="li"><div class="lsq" style="background:rgba(201,93,26,.15)"></div>Post-2020 Regime</div>
          <div class="li"><div class="ll" style="background:var(--blue2)"></div>SL Price</div>
          <div class="li"><div class="ll" style="background:var(--orange)"></div>12M Volatility</div>
        </div>
      </div>
      <div>
        <div class="regime-grid">
          <div class="regime-panel pre">
            <div class="rp-title"><div class="dot" style="background:var(--blue)"></div>Pre-2020 Regime</div>
            <div class="rp-stats">
              <div class="rp-stat"><div class="rp-stat-lbl">Mean Price</div><div class="rp-stat-val">$4.45</div></div>
              <div class="rp-stat"><div class="rp-stat-lbl">Max Price</div><div class="rp-stat-val">$5.26</div></div>
              <div class="rp-stat"><div class="rp-stat-lbl">Min Price</div><div class="rp-stat-val">$3.23</div></div>
              <div class="rp-stat"><div class="rp-stat-lbl">Mean Vol</div><div class="rp-stat-val">9.19%</div></div>
            </div>
          </div>
          <div class="regime-panel post">
            <div class="rp-title"><div class="dot" style="background:var(--orange)"></div>Post-2020 Regime</div>
            <div class="rp-stats">
              <div class="rp-stat"><div class="rp-stat-lbl">Mean Price</div><div class="rp-stat-val">$4.85</div></div>
              <div class="rp-stat"><div class="rp-stat-lbl">Max Price</div><div class="rp-stat-val">$5.76</div></div>
              <div class="rp-stat"><div class="rp-stat-lbl">Min Price</div><div class="rp-stat-val">$3.86</div></div>
              <div class="rp-stat"><div class="rp-stat-lbl">Mean Vol</div><div class="rp-stat-val">13.44%</div></div>
            </div>
          </div>
        </div>
        <div class="regime-diff">
          <div class="regime-diff-title">Post-2020 Change vs Pre-2020</div>
          <div class="diff-row">
            <div class="diff-lbl">Mean Price</div>
            <div class="diff-bar-wrap"><div class="diff-bar" style="width:75%;background:var(--blue2)"></div></div>
            <div class="diff-val">$4.45→$4.85</div>
            <span class="diff-delta delta-up">+9.0%</span>
          </div>
          <div class="diff-row">
            <div class="diff-lbl">Mean Volatility</div>
            <div class="diff-bar-wrap"><div class="diff-bar" style="width:85%;background:var(--orange)"></div></div>
            <div class="diff-val">9.2→13.4%</div>
            <span class="diff-delta delta-up">+46.3%</span>
          </div>
          <div class="diff-row">
            <div class="diff-lbl">Price Std Dev</div>
            <div class="diff-bar-wrap"><div class="diff-bar" style="width:65%;background:var(--purple)"></div></div>
            <div class="diff-val">0.42→0.52</div>
            <span class="diff-delta delta-up">+22.5%</span>
          </div>
          <div class="diff-row">
            <div class="diff-lbl">Avg Return</div>
            <div class="diff-bar-wrap"><div class="diff-bar" style="width:55%;background:var(--green)"></div></div>
            <div class="diff-val">0.12→0.46%</div>
            <span class="diff-delta delta-up">+284%</span>
          </div>
          <div class="diff-row">
            <div class="diff-lbl">Excess Kurtosis</div>
            <div class="diff-bar-wrap"><div class="diff-bar" style="width:90%;background:var(--red)"></div></div>
            <div class="diff-val">1.87→5.35</div>
            <span class="diff-delta delta-up">Fatter tails</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- §7 FORECAST ─────────────────────────────────── -->
  <div class="section" id="s7">
    <div class="sec-hdr">
      <span class="sec-num">§ 07</span>
      <span class="sec-title">Forecasting Module — ARIMA(1,0,0)</span>
      <span class="sec-subtitle">Yule-Walker estimation · φ̂ ≈ 0.31 · Log-return based prediction</span>
    </div>

    <div class="card">
      <div class="card-hdr">
        <div>
          <div class="card-title">Historical Export Price &amp; ARIMA(1,0,0) Forecast with 95% Confidence Interval</div>
          <div class="card-note">Point forecast shows modest upward trend · widening CI reflects long-horizon uncertainty</div>
        </div>
        <div class="fc-controls">
          <span class="fc-lbl">Horizon</span>
          <button class="fc-btn" onclick="setFcHorizon(6)">6M</button>
          <button class="fc-btn active" onclick="setFcHorizon(12)">12M</button>
          <button class="fc-btn" onclick="setFcHorizon(24)">24M</button>
        </div>
      </div>
      <div class="cw tall"><canvas id="cFc"></canvas></div>
      <div class="leg">
        <div class="li"><div class="ll" style="background:var(--blue2)"></div>Historical Price</div>
        <div class="li"><div class="ll" style="background:var(--green);border-top:2px dashed var(--green);height:0"></div>ARIMA Forecast</div>
        <div class="li"><div class="lsq" style="background:rgba(30,122,75,.2)"></div>95% Confidence Interval</div>
      </div>
      <div style="margin-top:14px;background:var(--surface2);border-radius:var(--r);padding:12px 16px;border:1px solid var(--border)">
        <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--ink3);margin-bottom:8px">Model Parameters &amp; Diagnostics</div>
        <div style="display:flex;gap:24px;flex-wrap:wrap">
          <div><span style="font-size:.72rem;color:var(--ink3)">Model: </span><strong style="font-size:.78rem">ARIMA(1,0,0)</strong></div>
          <div><span style="font-size:.72rem;color:var(--ink3)">Method: </span><strong style="font-size:.78rem">Yule-Walker</strong></div>
          <div><span style="font-size:.72rem;color:var(--ink3)">Series: </span><strong style="font-size:.78rem">Log Returns</strong></div>
          <div><span style="font-size:.72rem;color:var(--ink3)">Last Price: </span><strong style="font-size:.78rem">$5.577/kg</strong></div>
          <div><span style="font-size:.72rem;color:var(--ink3)">12M Forecast: </span><strong style="font-size:.78rem" style="color:var(--green)">$5.718/kg</strong></div>
          <div><span style="font-size:.72rem;color:var(--ink3)">CI (12M): </span><strong style="font-size:.78rem">[$2.61 – $12.54]</strong></div>
        </div>
      </div>
    </div>
  </div>

</main>

<footer>
  Dissertation Research Dashboard · "Volatility, Structural Changes, and Forecasting of Sri Lankan Tea Export Prices" ·
  Data: Tea_Export_Master_2008_2024 &amp; Global Tea Price Index · ARIMA(1,0,0) via Yule-Walker Estimation
</footer>
</div><!-- #app -->

<script>
// ══ DATA ═══════════════════════════════════════════════════════════
const R = {"labels": ["Jan 2008", "Feb 2008", "Mar 2008", "Apr 2008", "May 2008", "Jun 2008", "Jul 2008", "Aug 2008", "Sep 2008", "Oct 2008", "Nov 2008", "Dec 2008", "Jan 2009", "Feb 2009", "Mar 2009", "Apr 2009", "May 2009", "Jun 2009", "Jul 2009", "Aug 2009", "Sep 2009", "Oct 2009", "Nov 2009", "Dec 2009", "Jan 2010", "Feb 2010", "Mar 2010", "Apr 2010", "May 2010", "Jun 2010", "Jul 2010", "Aug 2010", "Sep 2010", "Oct 2010", "Nov 2010", "Dec 2010", "Jan 2011", "Feb 2011", "Mar 2011", "Apr 2011", "May 2011", "Jun 2011", "Jul 2011", "Aug 2011", "Sep 2011", "Oct 2011", "Nov 2011", "Dec 2011", "Jan 2012", "Feb 2012", "Mar 2012", "Apr 2012", "May 2012", "Jun 2012", "Jul 2012", "Aug 2012", "Sep 2012", "Oct 2012", "Nov 2012", "Dec 2012", "Jan 2013", "Feb 2013", "Mar 2013", "Apr 2013", "May 2013", "Jun 2013", "Jul 2013", "Aug 2013", "Sep 2013", "Oct 2013", "Nov 2013", "Dec 2013", "Jan 2014", "Feb 2014", "Mar 2014", "Apr 2014", "May 2014", "Jun 2014", "Jul 2014", "Aug 2014", "Sep 2014", "Oct 2014", "Nov 2014", "Dec 2014", "Jan 2015", "Feb 2015", "Mar 2015", "Apr 2015", "May 2015", "Jun 2015", "Jul 2015", "Aug 2015", "Sep 2015", "Oct 2015", "Nov 2015", "Dec 2015", "Jan 2016", "Feb 2016", "Mar 2016", "Apr 2016", "May 2016", "Jun 2016", "Jul 2016", "Aug 2016", "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016", "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "Jun 2017", "Jul 2017", "Aug 2017", "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017", "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018", "Jul 2018", "Aug 2018", "Sep 2018", "Oct 2018", "Nov 2018", "Dec 2018", "Jan 2019", "Feb 2019", "Mar 2019", "Apr 2019", "May 2019", "Jun 2019", "Jul 2019", "Aug 2019", "Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019", "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020", "Jul 2020", "Aug 2020", "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020", "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021", "May 2021", "Jun 2021", "Jul 2021", "Aug 2021", "Sep 2021", "Oct 2021", "Nov 2021", "Dec 2021", "Jan 2022", "Feb 2022", "Mar 2022", "Apr 2022", "May 2022", "Jun 2022", "Jul 2022", "Aug 2022", "Sep 2022", "Oct 2022", "Nov 2022", "Dec 2022", "Jan 2023", "Feb 2023", "Mar 2023", "Apr 2023", "May 2023", "Jun 2023", "Jul 2023", "Aug 2023", "Sep 2023", "Oct 2023", "Nov 2023", "Dec 2023", "Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024", "May 2024", "Jun 2024", "Jul 2024", "Aug 2024", "Sep 2024", "Oct 2024", "Nov 2024", "Dec 2024"], "price": [3.56042, 3.69418, 3.82941, 3.94515, 4.06449, 4.18169, 4.32231, 4.36405, 4.28714, 3.84075, 3.45674, 3.23352, 3.25414, 3.37223, 3.5428, 3.66461, 3.90432, 4.11083, 4.23418, 4.42276, 4.4632, 4.44605, 4.38463, 4.3326, 4.24508, 4.19165, 4.29527, 4.35081, 4.40603, 4.45194, 4.51624, 4.58146, 4.47695, 4.46288, 4.55086, 4.69788, 4.73659, 4.71089, 4.89639, 4.8052, 4.69708, 4.56862, 4.45459, 4.41645, 4.50493, 4.50607, 4.56838, 4.50039, 4.28446, 4.61137, 4.2254, 4.16427, 4.25587, 4.20216, 4.18113, 4.20942, 4.20994, 4.3718, 4.30745, 4.32594, 4.37822, 4.56977, 4.61737, 4.6698, 4.63184, 4.46088, 4.33348, 4.33466, 4.66465, 4.86446, 5.04899, 4.91365, 5.08605, 4.95549, 4.8291, 4.80903, 4.79587, 4.80285, 4.75388, 4.88922, 4.7998, 4.72525, 4.54948, 4.51458, 4.43565, 4.39511, 4.32299, 4.25646, 4.25438, 4.19149, 4.22545, 4.14632, 4.11265, 3.89952, 3.90629, 3.92884, 3.9447, 3.95511, 3.89945, 3.97115, 4.05935, 4.03353, 3.96991, 4.13527, 4.335, 4.49129, 4.71861, 4.65509, 4.85348, 4.8989, 4.97401, 5.20605, 5.25774, 5.19513, 5.16952, 5.0887, 5.16176, 5.15864, 5.18702, 5.13761, 5.16081, 5.23493, 5.11776, 5.09413, 5.06947, 5.04801, 4.76864, 4.67533, 4.55635, 4.73869, 4.63471, 4.44171, 4.47916, 4.49363, 4.54023, 4.57346, 4.53357, 4.43532, 4.28308, 4.29076, 4.22631, 4.18477, 4.37383, 4.22789, 4.28115, 4.32597, 4.29607, 4.1379, 4.59137, 4.64794, 4.4612, 4.4663, 4.49957, 4.61576, 4.5169, 4.50522, 4.57135, 4.58823, 4.58642, 4.57013, 4.34788, 4.35046, 4.23962, 4.29403, 4.21039, 4.25637, 4.33563, 4.33105, 4.45228, 4.57031, 3.85631, 4.09314, 4.48238, 4.66281, 4.85867, 5.08208, 5.1791, 5.41193, 5.17046, 5.22784, 5.35267, 5.30268, 5.72196, 5.46226, 5.3146, 4.84283, 4.73889, 4.79081, 4.89276, 4.99833, 4.85431, 5.05374, 5.16916, 5.41709, 5.58852, 5.52604, 5.61601, 5.54265, 5.59317, 5.56882, 5.62626, 5.75777, 5.61123, 5.57728], "priceLKR": [385.12, 398.4, 412.55, 425.3, 438.1, 450.75, 465.2, 470.15, 462.4, 415.6, 380.25, 360.85, 370.45, 385.1, 405.3, 430.25, 455.6, 472.4, 485.15, 505.8, 512.2, 510.45, 502.1, 495.35, 485.6, 480.15, 490.3, 495.45, 501.2, 505.75, 510.4, 515.15, 502.8, 498.45, 507.5, 521.85, 525.0, 522.15, 540.29, 529.81, 515.74, 500.64, 487.78, 484.62, 495.82, 496.1, 508.35, 512.24, 492.17, 530.71, 530.71, 535.65, 548.369, 555.269, 552.16, 556.57, 562.849, 569.05, 570.07, 554.11, 563.33, 579.63, 585.39, 594.84, 587.78, 578.27, 577.0, 579.26, 625.82, 645.65, 670.15, 650.61, 672.38, 655.45, 637.73, 635.03, 632.49, 632.66, 625.94, 643.48, 632.03, 624.06, 602.99, 601.23, 594.3, 591.33, 582.25, 574.02, 579.32, 569.22, 571.57, 562.85, 580.74, 557.9, 563.27, 571.79, 576.11, 578.12, 574.47, 583.18, 599.81, 596.55, 587.36, 609.42, 639.64, 668.7, 708.68, 703.57, 737.36, 748.51, 764.3, 802.26, 811.74, 804.1, 803.42, 788.55, 798.42, 801.23, 806.12, 796.05, 802.85, 819.69, 805.87, 804.42, 807.23, 809.82, 766.33, 756.04, 756.96, 819.32, 827.66, 807.97, 823.4, 810.97, 817.85, 808.07, 808.07, 791.25, 761.13, 771.11, 771.06, 766.37, 796.64, 773.54, 784.2, 793.16, 802.61, 813.28, 872.17, 875.16, 839.2, 835.68, 842.11, 861.15, 844.61, 854.67, 890.42, 903.76, 916.66, 916.82, 878.66, 882.67, 860.21, 871.64, 854.88, 864.04, 880.13, 879.2, 903.81, 927.77, 1009.15, 1355.27, 1646.62, 1709.32, 1789.11, 1873.6, 1914.4, 2005.91, 1921.81, 1943.16, 1988.24, 1960.14, 1935.08, 1795.61, 1682.77, 1500.08, 1546.66, 1570.65, 1607.19, 1649.64, 1618.14, 1676.08, 1685.02, 1719.17, 1733.17, 1680.67, 1710.42, 1710.11, 1724.64, 1699.85, 1719.77, 1717.55, 1663.65, 1651.14], "global": [2.37718, 2.39684, 2.2626, 2.48788, 2.48376, 2.66924, 2.75915, 2.74207, 2.66866, 2.30496, 1.96734, 1.92601, 2.18582, 2.13519, 2.21859, 2.50892, 2.69362, 2.78154, 2.95894, 2.99232, 3.15704, 3.02668, 3.05593, 2.97358, 2.89775, 2.85122, 2.6214, 2.77436, 2.79885, 2.71805, 2.8641, 2.99338, 2.99603, 3.03473, 3.02706, 3.0423, 3.02085, 2.88211, 2.75774, 3.02167, 2.95092, 3.01791, 3.10101, 3.02196, 2.88722, 2.90293, 2.78716, 2.6951, 2.65664, 2.56716, 2.42423, 2.79142, 3.0016, 2.97244, 3.02936, 3.11729, 3.10405, 3.00925, 3.01698, 3.08301, 3.00337, 2.92515, 2.89627, 2.88914, 2.95447, 2.83521, 2.83185, 2.78224, 2.75336, 2.79778, 2.77881, 2.89629, 2.8744, 2.57569, 2.49552, 2.67086, 2.86977, 2.84751, 2.96437, 2.79229, 2.63504, 2.64901, 2.64679, 2.62448, 2.61455, 2.45444, 2.40935, 2.63454, 2.88605, 2.99674, 3.09648, 2.9591, 2.78155, 2.86237, 2.82155, 2.80907, 2.5522, 2.35174, 2.31289, 2.53755, 2.57488, 2.64811, 2.77243, 2.77524, 2.79071, 2.88827, 3.03436, 2.98082, 3.04413, 2.96015, 2.87439, 3.03416, 3.19805, 3.2376, 3.22588, 3.2645, 3.28176, 3.29434, 3.23464, 3.13215, 3.1022, 2.9996, 2.8355, 2.94784, 3.02045, 2.8678, 2.88854, 2.7741, 2.67161, 2.73668, 2.70174, 2.62357, 2.53712, 2.38197, 2.38102, 2.64702, 2.73003, 2.56195, 2.57952, 2.62694, 2.54333, 2.58179, 2.61496, 2.5681, 2.50762, 2.35912, 2.13392, 2.35584, 2.5072, 2.83585, 3.04014, 3.1485, 3.07961, 2.99742, 2.77943, 2.64694, 2.67953, 2.55542, 2.42971, 2.6668, 2.71186, 2.69851, 2.65257, 2.71523, 2.72736, 2.78018, 2.83076, 2.81891, 2.8619, 2.78836, 2.61365, 3.24067, 2.98525, 2.96428, 3.30857, 3.36938, 3.3344, 3.14687, 3.04666, 2.94939, 2.76957, 2.70276, 2.68777, 2.95804, 2.71224, 2.69451, 2.4667, 2.83496, 2.84695, 2.75596, 2.7412, 2.71402, 2.6578, 2.70836, 2.68656, 3.03389, 3.19554, 3.22212, 3.29554, 3.22487, 3.26847, 3.14556, 3.08917, 2.95566], "quantity": [25100000, 23850000, 27200000, 21400000, 28500000, 31250000, 30150000, 28900000, 29450000, 27800000, 24150000, 21950000, 19850000, 18200000, 22150000, 17900000, 21400000, 25800000, 28500000, 27200000, 29150000, 30250000, 25100000, 24250000, 24300000, 22450000, 27850000, 20600000, 25150000, 28900000, 30100000, 29450000, 28300000, 29550000, 26400000, 24950000, 27160000, 25220000, 26900000, 20340000, 25020000, 25680000, 29230000, 31240000, 26440000, 28050000, 27040000, 30230000, 21532067, 22067841, 27380536, 25290000, 25590000, 26970000, 25545824, 20836848, 26971523, 23840522, 26247282, 29403610, 21258611, 20926929, 25874130, 20825643, 25603861, 23814720, 28720244, 30142969, 27603704, 27688731, 26132222, 27726166, 20810516, 21470076, 29371026, 22482923, 26949119, 29294029, 28429770, 26314550, 25558985, 25976573, 27626338, 28459950, 23518657, 22519417, 24112613, 22036156, 26468222, 26743912, 27624298, 20370925, 23566812, 27857467, 25307906, 24226848, 23180845, 22572195, 25556295, 21262043, 20548284, 27304937, 24738053, 24198508, 24391897, 22006623, 19351961, 21309732, 19238717, 21504567, 24304644, 19036407, 22950205, 24778399, 25185722, 23483621, 24184257, 24236508, 22140427, 22629475, 20119373, 20415021, 24623781, 19649229, 21769059, 23469594, 23615812, 22636907, 24412669, 22000335, 20158563, 24057367, 22184171, 22171120, 24829965, 19863877, 19863877, 22817206, 23607239, 24947750, 23686388, 24698108, 20924297, 22658233, 20985856, 22460015, 12973999, 17329806, 21536168, 22388900, 26424927, 20986225, 23046532, 21758532, 18961021, 22925702, 19681330, 21994391, 24369241, 15340043, 22617968, 25510131, 24302510, 24263692, 22277123, 23099295, 24492016, 22811361, 18433734, 20149613, 22028844, 17410171, 18610703, 22300051, 21661017, 21791896, 20252727, 17865940, 18127394, 18138262, 16381285, 17295105, 16877670, 15035531, 18705403, 20267634, 21686629, 22229855, 21905662, 17247769, 19616482, 19505950, 17756334, 20846210, 19917944, 15346808, 18082178, 19435617, 19564328, 20350896, 18188846, 18961991, 18458691, 20562429], "fx": [108.16696, 107.84524, 107.7319, 107.80318, 107.78727, 107.79143, 107.62761, 107.73262, 107.8575, 108.20804, 110.0025, 111.59652, 113.83955, 114.1975, 114.40091, 117.40682, 116.69119, 114.91591, 114.57935, 114.36286, 114.76068, 114.80977, 114.51357, 114.33087, 114.39119, 114.54925, 114.14891, 113.87523, 113.7531, 113.60205, 113.01432, 112.44227, 112.30864, 111.68786, 111.5175, 111.08196, 110.83929, 110.839, 110.34457, 110.25762, 109.80023, 109.58227, 109.50048, 109.73065, 110.06159, 110.09595, 111.27568, 113.82136, 114.87334, 115.08723, 125.6, 128.63, 128.85, 132.1388, 132.06, 132.22, 133.69537, 130.1637, 132.3452, 128.09, 128.6663, 126.84, 126.78, 127.3803, 126.9, 129.63143, 133.1494, 133.63456, 134.16222, 132.72786, 132.72953, 132.40881, 132.2009, 132.26735, 132.0597, 132.04948, 131.88213, 131.72583, 131.66919, 131.61203, 131.67848, 132.06923, 132.54033, 133.17523, 133.98251, 134.54263, 134.68702, 134.85852, 136.17014, 135.80365, 135.26834, 135.747, 141.20806, 143.06898, 144.1955, 145.53641, 146.04658, 146.17052, 147.32063, 146.85423, 147.76003, 147.89779, 147.95309, 147.37119, 147.55238, 148.88821, 150.18831, 151.14007, 151.92413, 152.79152, 153.65857, 154.10139, 154.3895, 154.77949, 155.41485, 154.96101, 154.67983, 155.31794, 155.41088, 154.94547, 155.56666, 156.58103, 157.46547, 157.9112, 159.23363, 160.42364, 160.70192, 161.70843, 166.13293, 172.89994, 178.57845, 181.90498, 183.82912, 180.47092, 180.13405, 176.68696, 178.24148, 178.39734, 177.70604, 179.71426, 182.44269, 183.13296, 182.13798, 182.96135, 183.17507, 183.34856, 186.82403, 196.54408, 189.95858, 188.29, 188.11087, 187.108, 187.15333, 186.5675, 186.98905, 189.70667, 194.78292, 196.97339, 199.86389, 200.61128, 202.08938, 202.89135, 202.8977, 202.98861, 203.04071, 202.9992, 202.9992, 202.9992, 202.9992, 202.9992, 261.68831, 331.10777, 367.35405, 366.58585, 368.23014, 368.66789, 369.63934, 370.64573, 371.69019, 371.69432, 371.44795, 369.65102, 338.18488, 328.73015, 316.63134, 309.75248, 326.37588, 327.84661, 328.48348, 330.03812, 333.3406, 331.65142, 325.97566, 317.3605, 310.13023, 304.13645, 304.56132, 308.53667, 308.3477, 305.24431, 305.66864, 298.30108, 296.48585, 296.04762], "logRet": [null, 0.03688, 0.03595, 0.02978, 0.0298, 0.02843, 0.03308, 0.00961, -0.01778, -0.10995, -0.10534, -0.06675, 0.00636, 0.03565, 0.04934, 0.0338, 0.06336, 0.05154, 0.02957, 0.04357, 0.0091, -0.00385, -0.01391, -0.01194, -0.02041, -0.01267, 0.02442, 0.01285, 0.01261, 0.01037, 0.01434, 0.01434, -0.02308, -0.00315, 0.01952, 0.0318, 0.00821, -0.00544, 0.03862, -0.0188, -0.02276, -0.02773, -0.02528, -0.0086, 0.01984, 0.00025, 0.01373, -0.015, -0.04917, 0.07353, -0.08741, -0.01457, 0.02176, -0.0127, -0.00502, 0.00674, 0.00012, 0.03773, -0.01483, 0.00428, 0.01201, 0.04282, 0.01036, 0.01129, -0.00816, -0.03761, -0.02898, 0.00027, 0.07337, 0.04194, 0.03723, -0.02717, 0.03448, -0.026, -0.02584, -0.00417, -0.00274, 0.00145, -0.01025, 0.02807, -0.01846, -0.01565, -0.03791, -0.0077, -0.01764, -0.00918, -0.01655, -0.01551, -0.00049, -0.01489, 0.00807, -0.01891, -0.00815, -0.05322, 0.00174, 0.00576, 0.00403, 0.00263, -0.01417, 0.01822, 0.02197, -0.00638, -0.0159, 0.04081, 0.04717, 0.03542, 0.04937, -0.01355, 0.04173, 0.00932, 0.01522, 0.04559, 0.00988, -0.01198, -0.00494, -0.01576, 0.01426, -0.0006, 0.00549, -0.00957, 0.0045, 0.01426, -0.02264, -0.00463, -0.00485, -0.00424, -0.05693, -0.01976, -0.02578, 0.03924, -0.02219, -0.04253, 0.00839, 0.00323, 0.01032, 0.00729, -0.00876, -0.02191, -0.03493, 0.00179, -0.01513, -0.00988, 0.04419, -0.03394, 0.01252, 0.01041, -0.00693, -0.03751, 0.10399, 0.01225, -0.04101, 0.00114, 0.00742, 0.02549, -0.02165, -0.00259, 0.01457, 0.00369, -0.0004, -0.00356, -0.04985, 0.00059, -0.02581, 0.01275, -0.01967, 0.01086, 0.01845, -0.00106, 0.02761, 0.02616, -0.16987, 0.0596, 0.09084, 0.03946, 0.04115, 0.04496, 0.01891, 0.04397, -0.04564, 0.01104, 0.0236, -0.00938, 0.0761, -0.04645, -0.0274, -0.09296, -0.0217, 0.0109, 0.02106, 0.02135, -0.02924, 0.04026, 0.02258, 0.04685, 0.03116, -0.01124, 0.01615, -0.01315, 0.00907, -0.00436, 0.01026, 0.02311, -0.02578, -0.00607], "vol6": [null, null, null, null, null, null, 1.22511, 3.22549, 6.85946, 19.07969, 22.51661, 20.85631, 18.75841, 20.89424, 24.6546, 21.94945, 16.16554, 6.85092, 4.49643, 4.28279, 6.53635, 8.91922, 9.13495, 8.11868, 8.12069, 3.56808, 5.5402, 6.18943, 6.26604, 5.96348, 4.25687, 1.70565, 5.11301, 5.1668, 5.51604, 6.68811, 6.62571, 6.75644, 6.30012, 7.62695, 8.90523, 8.62427, 8.72573, 8.6955, 6.17615, 6.39629, 6.81513, 5.9642, 8.5499, 14.13136, 19.09655, 19.00781, 19.31162, 19.30526, 18.21249, 13.12008, 4.68748, 6.45211, 6.66531, 6.18107, 5.99575, 7.76938, 7.46938, 6.43675, 5.83584, 9.21274, 10.24265, 7.08308, 13.69439, 14.8473, 15.134, 14.32404, 12.18127, 13.98248, 12.21465, 10.56732, 8.27795, 7.70431, 4.14378, 6.11341, 5.48712, 5.87243, 7.67461, 7.48824, 7.55919, 3.74884, 3.75, 3.75171, 2.29591, 2.25961, 3.43956, 3.77238, 3.58974, 7.37203, 7.48013, 7.98874, 7.77744, 7.87835, 7.92937, 3.58945, 4.4509, 4.80994, 5.60505, 7.96536, 8.66692, 9.0279, 9.92159, 10.5023, 8.16505, 8.6858, 8.17525, 8.66097, 7.69864, 7.5453, 6.92727, 7.87417, 7.84927, 4.10888, 3.8554, 3.72242, 3.7738, 3.15645, 4.50567, 4.52392, 4.34052, 4.23521, 8.45546, 7.06035, 7.11462, 10.96071, 10.9615, 11.39344, 10.17405, 10.18781, 9.8302, 7.48107, 7.00605, 4.37315, 6.2213, 6.16515, 5.38095, 4.35893, 9.50906, 10.12481, 9.28287, 9.4038, 9.12603, 10.7369, 17.86167, 16.38702, 18.28191, 18.28575, 18.16716, 16.51251, 8.43123, 8.05153, 5.57479, 5.55435, 5.56528, 4.09215, 7.72807, 7.77823, 7.31954, 7.89215, 7.73719, 8.45157, 6.33656, 6.33422, 5.73838, 6.28373, 26.59777, 28.54153, 31.72046, 32.03405, 32.24585, 32.51654, 8.39616, 8.2184, 12.24479, 12.03016, 11.50137, 10.77843, 14.63067, 16.08807, 14.97221, 20.20517, 19.24206, 19.7258, 14.28087, 15.15348, 15.19223, 9.37414, 8.11617, 9.2483, 9.37296, 10.50556, 7.17147, 8.21858, 8.15672, 6.01014, 4.26879, 4.62963, 6.15407, 5.86978], "vol12": [null, null, null, null, null, null, null, null, null, null, null, null, 19.18703, 19.15624, 19.53153, 19.61705, 20.55278, 21.0662, 21.00633, 21.47546, 21.38551, 17.53907, 12.51976, 9.1246, 10.04724, 10.49123, 9.96501, 9.77628, 8.18592, 6.83451, 6.43538, 5.05908, 5.62328, 5.61811, 5.68887, 6.09429, 5.32699, 5.04601, 5.65955, 6.38055, 7.06241, 7.75914, 8.08501, 7.92969, 7.90893, 7.89904, 7.76961, 7.08006, 8.264, 11.64861, 13.54396, 13.52836, 13.86782, 13.71781, 13.57166, 13.61737, 13.38156, 14.0688, 13.99857, 13.97965, 13.10061, 11.53239, 6.6131, 6.16664, 6.24568, 7.47984, 8.18003, 8.17316, 10.80484, 10.93647, 10.94338, 11.65319, 11.89671, 11.97298, 12.39374, 12.38734, 12.34848, 11.5421, 11.06224, 11.20651, 9.11105, 8.15276, 7.61063, 7.33877, 5.81215, 5.59032, 5.37635, 5.37952, 5.41803, 5.28656, 5.634, 3.86969, 3.86969, 5.60005, 5.30407, 5.6047, 5.76571, 5.89853, 5.86916, 6.38226, 6.95483, 6.85964, 6.84876, 8.04404, 9.25958, 7.27347, 8.07245, 8.54991, 8.88843, 8.79297, 8.08899, 8.4672, 8.56224, 8.77709, 8.3304, 8.84208, 8.21938, 8.00815, 6.86787, 6.74265, 5.66565, 5.72686, 6.25492, 4.12159, 3.9243, 3.80651, 6.66406, 6.73492, 6.53969, 8.19482, 8.19929, 8.84476, 8.93269, 8.64501, 8.85824, 9.01557, 9.00449, 9.06909, 8.05815, 8.03081, 7.82953, 6.05481, 7.97405, 7.56647, 7.65159, 7.75632, 7.61977, 8.15503, 13.87791, 13.68023, 13.88048, 13.88145, 13.73647, 13.7677, 13.45949, 12.88176, 12.89594, 12.88532, 12.83627, 12.0196, 7.73886, 7.54852, 6.89345, 7.08747, 7.15876, 6.5942, 6.82141, 6.82515, 7.30483, 7.82942, 18.67565, 20.09182, 21.83561, 22.14167, 22.11205, 22.36723, 22.06353, 22.20685, 23.16063, 23.10251, 23.08341, 23.17524, 12.8061, 14.65061, 13.49571, 16.86637, 16.48872, 15.76469, 15.8015, 15.19073, 14.81492, 15.44059, 15.42195, 16.1713, 14.45106, 13.67699, 13.38166, 8.53769, 7.76163, 7.97132, 7.91842, 7.94385, 7.75445, 7.28732], "forecasts": {"6": {"mean": [5.58196, 5.59305, 5.60522, 5.61759, 5.63002, 5.64248], "upper": [5.95437, 6.36984, 6.81576, 7.29312, 7.80396, 8.35059], "lower": [5.23285, 4.91098, 4.60968, 4.32699, 4.06167, 3.81261], "labels": ["Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025", "Jun 2025"]}, "12": {"mean": [5.58196, 5.59305, 5.60522, 5.61759, 5.63002, 5.64248, 5.65497, 5.66748, 5.68003, 5.6926, 5.7052, 5.71783], "upper": [5.95437, 6.36984, 6.81576, 7.29312, 7.80396, 8.35059, 8.93551, 9.56139, 10.23112, 10.94776, 11.7146, 12.53515], "lower": [5.23285, 4.91098, 4.60968, 4.32699, 4.06167, 3.81261, 3.57883, 3.35938, 3.15339, 2.96003, 2.77853, 2.60815], "labels": ["Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025", "Jun 2025", "Jul 2025", "Aug 2025", "Sep 2025", "Oct 2025", "Nov 2025", "Dec 2025"]}, "24": {"mean": [5.58196, 5.59305, 5.60522, 5.61759, 5.63002, 5.64248, 5.65497, 5.66748, 5.68003, 5.6926, 5.7052, 5.71783, 5.73048, 5.74317, 5.75588, 5.76862, 5.78139, 5.79418, 5.80701, 5.81986, 5.83274, 5.84565, 5.85859, 5.87156], "upper": [5.95437, 6.36984, 6.81576, 7.29312, 7.80396, 8.35059, 8.93551, 9.56139, 10.23112, 10.94776, 11.7146, 12.53515, 13.41317, 14.3527, 15.35803, 16.43378, 17.58489, 18.81662, 20.13463, 21.54496, 23.05408, 24.6689, 26.39683, 28.2458], "lower": [5.23285, 4.91098, 4.60968, 4.32699, 4.06167, 3.81261, 3.57883, 3.35938, 3.15339, 2.96003, 2.77853, 2.60815, 2.44822, 2.2981, 2.15719, 2.02491, 1.90075, 1.7842, 1.67479, 1.5721, 1.4757, 1.38521, 1.30027, 1.22054], "labels": ["Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025", "Jun 2025", "Jul 2025", "Aug 2025", "Sep 2025", "Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026", "May 2026", "Jun 2026", "Jul 2026", "Aug 2026", "Sep 2026", "Oct 2026", "Nov 2026", "Dec 2026"]}}, "seasonal": [4.485, 4.5463, 4.5376, 4.547, 4.6048, 4.5723, 4.5355, 4.5739, 4.6004, 4.6312, 4.5986, 4.565], "regime": {"pre": {"mean_price": 4.4484, "max_price": 5.2577, "min_price": 3.2335, "std_price": 0.4207, "mean_vol": 9.1875, "max_vol": 21.4755, "skew": -0.5501, "kurt": 1.8673, "mean_return": 0.12016}, "post": {"mean_price": 4.8499, "max_price": 5.7578, "min_price": 3.8563, "std_price": 0.5154, "mean_vol": 13.44, "max_vol": 23.1752, "skew": -1.149, "kurt": 5.3512, "mean_return": 0.46166}}, "corr": {"labels": ["SL Price", "Global Price", "Exchange Rate", "Quantity", "Log Return"], "matrix": [[1.0, 0.5724, 0.5596, -0.3305, 0.1431], [0.5724, 1.0, 0.2142, 0.0176, 0.1674], [0.5596, 0.2142, 1.0, -0.6239, 0.0782], [-0.3305, 0.0176, -0.6239, 1.0, -0.1179], [0.1431, 0.1674, 0.0782, -0.1179, 1.0]]}, "meta": {"first_price": 3.5604, "last_price": 5.5773, "max_price": 5.7578, "min_price": 3.2335, "avg_price": 4.5665, "latest_vol12": 7.2873, "avg_fx": 175.3817, "price_growth": 56.65, "n": 204}};

// ══ STATE ═══════════════════════════════════════════════════════════
let YF=2008, YT=2024, REGIME='all', CURRENCY='USD', FC_HORIZON=12;
const CH={};

// ══ UTILS ════════════════════════════════════════════════════════════
const $ = id => document.getElementById(id);

function slice(){
  let f=(YF-2008)*12, t=Math.min((YT-2008+1)*12,R.labels.length);
  if(REGIME==='pre')  {f=0; t=Math.min(144,R.labels.length);}
  if(REGIME==='post') {f=144; t=R.labels.length;}
  const s=k=>R[k].slice(f,t);
  return{labels:s('labels'),price:s('price'),priceLKR:s('priceLKR'),
         global:s('global'),quantity:s('quantity'),fx:s('fx'),
         logRet:s('logRet'),vol6:s('vol6'),vol12:s('vol12'),f,t};
}

function priceData(d){
  return CURRENCY==='USD' ? d.price : d.priceLKR;
}
function priceLabel(){
  return CURRENCY==='USD' ? 'Export Price (USD/kg)' : 'Export Price (LKR/kg)';
}
function pricePrefix(){return CURRENCY==='USD'?'$':'LKR ';}

function lsReg(xs,ys){
  const n=xs.length;
  const mx=xs.reduce((a,b)=>a+b)/n, my=ys.reduce((a,b)=>a+b)/n;
  let num=0,den=0;
  for(let i=0;i<n;i++){num+=(xs[i]-mx)*(ys[i]-my);den+=(xs[i]-mx)**2;}
  const m=num/den, b=my-m*mx;
  return xs.map(x=>+(m*x+b).toFixed(4));
}

function histogram(data,bins=22){
  const v=data.filter(x=>x!==null&&isFinite(x));
  const mn=Math.min(...v),mx=Math.max(...v),step=(mx-mn)/bins;
  const counts=Array(bins).fill(0);
  const lbls=[];
  for(let i=0;i<bins;i++) lbls.push((mn+i*step).toFixed(3));
  v.forEach(x=>{const idx=Math.min(bins-1,Math.floor((x-mn)/step));counts[idx]++;});
  return{lbls,counts};
}

// ══ CHART DEFAULTS ═══════════════════════════════════════════════════
Chart.defaults.font.family="'DM Sans',sans-serif";
Chart.defaults.font.size=11;
Chart.defaults.color='#6e7180';
Chart.defaults.plugins.legend.display=false;
Chart.defaults.plugins.tooltip.backgroundColor='rgba(28,28,46,0.94)';
Chart.defaults.plugins.tooltip.titleFont={size:11,weight:'600'};
Chart.defaults.plugins.tooltip.bodyFont={size:11};
Chart.defaults.plugins.tooltip.padding=10;
Chart.defaults.plugins.tooltip.cornerRadius=6;
Chart.defaults.plugins.tooltip.mode='index';
Chart.defaults.plugins.tooltip.intersect=false;

const scaleOpts={grid:{color:'rgba(0,0,0,.05)'},ticks:{maxTicksLimit:10,maxRotation:0}};

function mkLine(id,datasets,opts={}){
  if(CH[id]) CH[id].destroy();
  CH[id]=new Chart($(id),{
    type:'line',data:{labels:[],datasets},
    options:{
      responsive:true,maintainAspectRatio:true,
      interaction:{mode:'index',intersect:false},
      scales:{x:{...scaleOpts},y:{...scaleOpts,...(opts.y||{})}},
      plugins:{legend:{display:opts.legend||false,...(opts.legendOpts||{})},
               tooltip:{mode:'index',intersect:false},
               annotation:opts.anno||undefined},
      ...(opts.extra||{})
    }
  });
}
function DS(label,color,o={}){
  return{label,borderColor:color,backgroundColor:color+'18',
         borderWidth:1.8,pointRadius:0,pointHoverRadius:4,tension:0.3,fill:false,...o};
}

// ══ INIT ══════════════════════════════════════════════════════════════
function initCharts(){
  // §2 Main
  mkLine('cMain',[DS(priceLabel(),'#2b5ca8',{fill:true,backgroundColor:'rgba(43,92,168,.07)'})]);
  mkLine('cDual',[DS('Sri Lanka','#2b5ca8'),DS('Global','#e8793a')],{legend:true,legendOpts:{position:'top',labels:{boxWidth:14,font:{size:10}}}});
  mkLine('cFX',[DS('LKR/USD','#c95d1a',{fill:true,backgroundColor:'rgba(201,93,26,.07)'})]);

  // §3 Volatility
  mkLine('cVol',[DS('6-Month Vol','#c95d1a',{borderWidth:1.5}),DS('12-Month Vol','#5b3f9c',{borderWidth:2})],
    {legend:true,y:{title:{display:true,text:'Volatility (% p.a.)',font:{size:10}}},
     legendOpts:{position:'top',labels:{boxWidth:14,font:{size:10}}}});
  mkLine('cRSD',[DS('6M Std Dev','#e8793a',{borderWidth:1.5,fill:true,backgroundColor:'rgba(232,121,58,.08)'})]);

  // §4 Scatters
  ['cSc1','cSc2','cSc3'].forEach(id=>{
    if(CH[id]) CH[id].destroy();
    const colors=['rgba(43,92,168,.5)','rgba(201,93,26,.5)','rgba(30,122,75,.5)'];
    const ci=['cSc1','cSc2','cSc3'].indexOf(id);
    CH[id]=new Chart($(id),{
      type:'scatter',
      data:{datasets:[{label:'',data:[],backgroundColor:colors[ci],pointRadius:3.5,pointHoverRadius:5},
                      {label:'Regression',data:[],borderColor:colors[ci].replace('.5','1'),borderWidth:1.5,pointRadius:0,type:'line',fill:false,tension:0}]},
      options:{responsive:true,maintainAspectRatio:true,
        scales:{x:{...scaleOpts,title:{display:true,font:{size:10}}},
                y:{...scaleOpts,title:{display:true,font:{size:10}}}},
        plugins:{legend:{display:false}}}
    });
  });

  // §5 Returns
  mkLine('cRet',[DS('Log Return','#2b5ca8',{borderWidth:1.2,fill:true,backgroundColor:'rgba(43,92,168,.05)'})]);
  if(CH.cHist) CH.cHist.destroy();
  CH.cHist=new Chart($('cHist'),{
    type:'bar',data:{labels:[],datasets:[{label:'Frequency',data:[],
      backgroundColor:'rgba(43,92,168,.55)',borderColor:'#2b5ca8',borderWidth:1}]},
    options:{responsive:true,maintainAspectRatio:true,
      scales:{x:{...scaleOpts,ticks:{maxTicksLimit:8}},y:{...scaleOpts}},
      plugins:{legend:{display:false}}}
  });

  // §6 Regime
  mkLine('cRegime',[
    DS('SL Price','#2b5ca8',{borderWidth:2}),
    DS('12M Volatility','#c95d1a',{borderWidth:1.5,borderDash:[4,3],yAxisID:'y2'})
  ],{legend:true,extra:{scales:{
    x:{...scaleOpts},
    y:{...scaleOpts,title:{display:true,text:'Price (USD/kg)',font:{size:10}}},
    y2:{type:'linear',position:'right',grid:{display:false},
        title:{display:true,text:'Volatility (%)',font:{size:10}},ticks:{maxTicksLimit:6}}
  }},legendOpts:{position:'top',labels:{boxWidth:14,font:{size:10}}}});

  // §7 Forecast
  if(CH.cFc) CH.cFc.destroy();
  CH.cFc=new Chart($('cFc'),{
    type:'line',
    data:{labels:[],datasets:[
      {label:'Historical',borderColor:'#2b5ca8',borderWidth:2,pointRadius:0,tension:0.3,fill:false,data:[]},
      {label:'ARIMA Forecast',borderColor:'#1e7a4b',borderWidth:2,borderDash:[5,4],pointRadius:0,tension:0.3,fill:false,data:[]},
      {label:'Upper 95% CI',borderColor:'transparent',backgroundColor:'rgba(30,122,75,.15)',pointRadius:0,fill:'+1',data:[]},
      {label:'Lower 95% CI',borderColor:'transparent',backgroundColor:'rgba(30,122,75,.15)',pointRadius:0,fill:false,data:[]},
    ]},
    options:{responsive:true,maintainAspectRatio:true,
      interaction:{mode:'index',intersect:false},
      scales:{x:{...scaleOpts,ticks:{maxTicksLimit:16,maxRotation:0}},y:{...scaleOpts}},
      plugins:{legend:{display:true,position:'top',labels:{boxWidth:14,font:{size:10}}}}}
  });

  buildCorrTable();
  buildDistStats();
  updateAll();
}

// ══ UPDATE ALL ═══════════════════════════════════════════════════════
function updateAll(){
  const d=slice();
  const n=d.price.length;
  if(!n) return;
  const pd=priceData(d);
  const pfx=pricePrefix();

  // ── KPIs ──
  const lp=pd[n-1], ap=pd.reduce((a,b)=>a+b,0)/n;
  const afxV=d.fx.reduce((a,b)=>a+b,0)/n;
  const lv=d.vol12.filter(v=>v!==null);
  const latestV=lv[lv.length-1]||0;
  const growth=((d.price[n-1]/d.price[0]-1)*100).toFixed(1);

  $('k1').textContent=pfx+(CURRENCY==='USD'?lp.toFixed(3):(lp/1).toFixed(0));
  $('k1u').textContent=(CURRENCY==='USD'?'USD':'LKR')+' per kg · '+d.labels[n-1];
  $('k2').textContent=pfx+(CURRENCY==='USD'?ap.toFixed(3):(ap/1).toFixed(0));
  $('k2u').textContent=(CURRENCY==='USD'?'USD':'LKR')+' per kg';
  $('k3').textContent='+'+growth+'%';
  $('k4').textContent=afxV.toFixed(0);
  $('k4d').textContent='108 (2008) → 296 (2024)';
  $('k4d').className='kpi-delta dn';
  $('k5').textContent=latestV.toFixed(1)+'%';
  $('k5d').textContent=latestV<10?'▼ Low regime':'▲ Elevated';
  $('k5d').className='kpi-delta '+(latestV<10?'up':'dn');

  // ── Dynamic insight ──
  const trend=growth>30?'strong upward':'moderate upward';
  const volState=latestV<10?'recently stabilised':'remains elevated';
  $('insightText').textContent =
    `Over the ${n}-month analysis window (${d.labels[0]}–${d.labels[n-1]}), `+
    `Sri Lankan tea export prices recorded a ${trend} trend of +${growth}%, `+
    `rising from ${pfx}${d.price[0].toFixed(3)} to ${pfx}${d.price[n-1].toFixed(3)} USD/kg. `+
    `The exchange rate depreciated significantly from LKR 108 to LKR 296/USD (+174%), with a structural break in April 2022 (+83% shock). `+
    `Annualised 12-month volatility ${volState} at ${latestV.toFixed(1)}% p.a., down from the 2022 crisis peak of 32.5%. `+
    `The strongest price predictor is the Global Tea Price Index (r = 0.572), followed by exchange rate effects (r = 0.560).`;

  // ── §2 Time Series ──
  const varSel=$('varSel').value;
  let mainData,mainLabel;
  if(varSel==='price'){mainData=pd;mainLabel=priceLabel();}
  else if(varSel==='quantity'){mainData=d.quantity.map(q=>+(q/1e6).toFixed(2));mainLabel='Export Quantity (Million kg)';}
  else if(varSel==='fx'){mainData=d.fx;mainLabel='Exchange Rate (LKR/USD)';}
  else{mainData=d.priceLKR;mainLabel='Export Price (LKR/kg)';}

  $('tsTitle').textContent=mainLabel;
  CH.cMain.data.labels=d.labels;
  CH.cMain.data.datasets[0].data=mainData;
  CH.cMain.data.datasets[0].label=mainLabel;
  CH.cMain.update('none');

  CH.cDual.data.labels=d.labels;
  CH.cDual.data.datasets[0].data=pd;
  CH.cDual.data.datasets[1].data=d.global;
  CH.cDual.update('none');

  CH.cFX.data.labels=d.labels;
  CH.cFX.data.datasets[0].data=d.fx;
  CH.cFX.update('none');

  // ── §3 Volatility ──
  CH.cVol.data.labels=d.labels;
  CH.cVol.data.datasets[0].data=d.vol6;
  CH.cVol.data.datasets[1].data=d.vol12;
  CH.cVol.update('none');

  // Raw RSD
  const rsd6=d.logRet.map((v,i)=>{
    if(i<5) return null;
    const w=d.logRet.slice(i-5,i+1).filter(x=>x!==null);
    if(!w.length) return null;
    const m=w.reduce((a,b)=>a+b)/w.length;
    return +(Math.sqrt(w.reduce((a,b)=>a+(b-m)**2,0)/w.length)).toFixed(5);
  });
  CH.cRSD.data.labels=d.labels;
  CH.cRSD.data.datasets[0].data=rsd6;
  CH.cRSD.update('none');

  // ── §4 Scatter ──
  const validIdx=d.price.map((_,i)=>i).filter(i=>d.price[i]!==null&&d.fx[i]!==null);
  const scXFX=validIdx.map(i=>d.fx[i]);
  const scYP =validIdx.map(i=>pd[i]);
  CH.cSc1.data.datasets[0].data=validIdx.map(i=>({x:d.fx[i],y:pd[i]}));
  CH.cSc1.data.datasets[1].data=scXFX.map((x,i)=>({x,y:lsReg(scXFX,scYP)[i]}));
  CH.cSc1.options.scales.x.title.text='Exchange Rate (LKR/USD)';
  CH.cSc1.options.scales.y.title.text=priceLabel();
  CH.cSc1.update('none');

  const scXQ=validIdx.map(i=>d.quantity[i]/1e6);
  CH.cSc2.data.datasets[0].data=validIdx.map(i=>({x:d.quantity[i]/1e6,y:pd[i]}));
  CH.cSc2.data.datasets[1].data=scXQ.map((x,i)=>({x,y:lsReg(scXQ,scYP)[i]}));
  CH.cSc2.options.scales.x.title.text='Export Quantity (M kg)';
  CH.cSc2.options.scales.y.title.text=priceLabel();
  CH.cSc2.update('none');

  const scXG=validIdx.map(i=>d.global[i]);
  CH.cSc3.data.datasets[0].data=validIdx.map(i=>({x:d.global[i],y:pd[i]}));
  CH.cSc3.data.datasets[1].data=scXG.map((x,i)=>({x,y:lsReg(scXG,scYP)[i]}));
  CH.cSc3.options.scales.x.title.text='Global Tea Price (USD/kg)';
  CH.cSc3.options.scales.y.title.text=priceLabel();
  CH.cSc3.update('none');

  // ── §5 Returns ──
  CH.cRet.data.labels=d.labels;
  CH.cRet.data.datasets[0].data=d.logRet;
  CH.cRet.update('none');

  const hd=histogram(d.logRet);
  CH.cHist.data.labels=hd.lbls;
  CH.cHist.data.datasets[0].data=hd.counts;
  CH.cHist.update('none');

  // ── §6 Regime ──
  CH.cRegime.data.labels=R.labels;
  CH.cRegime.data.datasets[0].data=R.price;
  CH.cRegime.data.datasets[1].data=R.vol12;
  CH.cRegime.update('none');

  // ── §7 Forecast ──
  updateForecast();
}

function updateForecast(){
  const fc=R.forecasts[FC_HORIZON];
  const histEnd=Math.min(R.labels.length,(2024-2008+1)*12);
  const hp=R.price.slice(Math.max(0,histEnd-48),histEnd);
  const hl=R.labels.slice(Math.max(0,histEnd-48),histEnd);
  const nH=hp.length, nF=fc.mean.length;
  const nullH=Array(nH).fill(null), nullF=Array(nF).fill(null);
  CH.cFc.data.labels=[...hl,...fc.labels];
  CH.cFc.data.datasets[0].data=[...hp,...nullF];
  CH.cFc.data.datasets[1].data=[...nullH,...fc.mean];
  CH.cFc.data.datasets[2].data=[...nullH,...fc.upper];
  CH.cFc.data.datasets[3].data=[...nullH,...fc.lower];
  CH.cFc.update('none');
}

// ══ CORR TABLE ════════════════════════════════════════════════════════
function corrColor(v){
  const t=Math.max(0,Math.min(1,(v+1)/2));
  if(v>=0){
    const r=Math.round(43+(168-43)*t*2*(t>0.5?1:(t/0.5)));
    const g=Math.round(92+(92)*t);
    const b=Math.round(168);
    if(v>0.7) return{bg:'#1a3f6f',fg:'#fff'};
    if(v>0.4) return{bg:'#3a72b8',fg:'#fff'};
    if(v>0.15) return{bg:'#92b8df',fg:'#1c1c2e'};
    return{bg:'#e8eef7',fg:'#4a4a62'};
  } else {
    if(v<-0.7) return{bg:'#a42020',fg:'#fff'};
    if(v<-0.4) return{bg:'#c75050',fg:'#fff'};
    if(v<-0.15) return{bg:'#e8a0a0',fg:'#1c1c2e'};
    return{bg:'#f7e8e8',fg:'#4a4a62'};
  }
}
function buildCorrTable(){
  const t=$('corrTable');
  const labs=R.corr.labels, mat=R.corr.matrix;
  let html='<tr><th></th>'+labs.map(l=>`<th>${l}</th>`).join('')+'</tr>';
  mat.forEach((row,i)=>{
    html+=`<tr><th style="text-align:left">${labs[i]}</th>`;
    row.forEach((v,j)=>{
      const{bg,fg}=corrColor(v);
      const diag=i===j?'opacity:.35':'';
      html+=`<td style="background:${bg};color:${fg};${diag}" title="${labs[i]} vs ${labs[j]}: ${v.toFixed(4)}">${v.toFixed(3)}</td>`;
    });
    html+='</tr>';
  });
  t.innerHTML=html;
}

// ══ DIST STATS TABLE ══════════════════════════════════════════════════
function buildDistStats(){
  const v=R.logRet.filter(x=>x!==null&&isFinite(x));
  const n=v.length;
  const mean=v.reduce((a,b)=>a+b)/n;
  const std=Math.sqrt(v.reduce((a,b)=>a+(b-mean)**2,0)/n);
  const sorted=[...v].sort((a,b)=>a-b);
  const median=sorted[Math.floor(n/2)];
  const mn=Math.min(...v), mx=Math.max(...v);
  const skew=v.reduce((a,b)=>a+(b-mean)**3,0)/(n*std**3);
  const kurt=v.reduce((a,b)=>a+(b-mean)**4,0)/(n*std**4)-3;
  const q1=sorted[Math.floor(n*0.25)], q3=sorted[Math.floor(n*0.75)];
  const rows=[
    ['N (observations)',n,'Monthly returns 2008–2024'],
    ['Mean Return',(mean*100).toFixed(4)+'%','Small positive drift'],
    ['Std Deviation',(std*100).toFixed(4)+'%','Monthly volatility'],
    ['Annualised Vol',(std*Math.sqrt(12)*100).toFixed(2)+'%','Annualised σ'],
    ['Median',(median*100).toFixed(4)+'%','50th percentile'],
    ['Minimum',(mn*100).toFixed(4)+'%','Largest monthly loss (Apr 2022)'],
    ['Maximum',(mx*100).toFixed(4)+'%','Largest monthly gain'],
    ['Skewness',skew.toFixed(4),skew<0?'Negative (left tail heavier)':'Positive'],
    ['Excess Kurtosis',kurt.toFixed(4),kurt>3?'Leptokurtic (fat tails)':kurt>0?'Moderate tails':'Platykurtic'],
    ['Q1 (25th pct)',(q1*100).toFixed(4)+'%','25th percentile'],
    ['Q3 (75th pct)',(q3*100).toFixed(4)+'%','75th percentile'],
    ['IQR',((q3-q1)*100).toFixed(4)+'%','Interquartile range'],
  ];
  const tb=$('distStatBody');
  tb.innerHTML=rows.map(([lbl,val,note])=>{
    const numClass=typeof val==='string'&&val.includes('-')&&lbl!=='Minimum'?'neg num':
                   typeof val==='string'&&val.includes('+')&&!lbl.includes('Kurtosis')?'pos num':'num';
    return`<tr><td>${lbl}</td><td class="${numClass}">${val}</td><td style="color:var(--ink3);font-size:.7rem">${note}</td></tr>`;
  }).join('');
}

// ══ CONTROLS ═════════════════════════════════════════════════════════
$('yf').oninput=e=>{
  YF=+e.target.value;
  if(YF>YT){YT=YF;$('yt').value=YT;}
  $('yfL').textContent=YF; $('ytL').textContent=YT;
  updateAll();
};
$('yt').oninput=e=>{
  YT=+e.target.value;
  if(YT<YF){YF=YT;$('yf').value=YF;}
  $('yfL').textContent=YF; $('ytL').textContent=YT;
  updateAll();
};
$('varSel').onchange=updateAll;
$('annoChk').onchange=e=>{
  const strips=document.querySelectorAll('.anno-strip,.anno-chip');
  strips.forEach(s=>{s.style.display=e.target.checked?'':'none'});
};

function setRegime(r){
  REGIME=r;
  ['tAll','tPre','tPost'].forEach(id=>$(id).classList.remove('on'));
  $({all:'tAll',pre:'tPre',post:'tPost'}[r]).classList.add('on');
  if(r==='pre')  {$('yf').value=2008;$('yt').value=2019;YF=2008;YT=2019;}
  if(r==='post') {$('yf').value=2020;$('yt').value=2024;YF=2020;YT=2024;}
  if(r==='all')  {$('yf').value=2008;$('yt').value=2024;YF=2008;YT=2024;}
  $('yfL').textContent=YF;$('ytL').textContent=YT;
  updateAll();
}
function setCurrency(c){
  CURRENCY=c;
  $('cUSD').classList.toggle('on',c==='USD');
  $('cLKR').classList.toggle('on',c==='LKR');
  updateAll();
}
function setFcHorizon(h){
  FC_HORIZON=h;
  document.querySelectorAll('.fc-btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  updateForecast();
}
function scrollTo(id){
  $(id).scrollIntoView({behavior:'smooth',block:'start'});
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
}
function toggleFS(){
  const a=$('app');
  if(!document.fullscreenElement) a.requestFullscreen&&a.requestFullscreen();
  else document.exitFullscreen&&document.exitFullscreen();
}
function exportPNG(){
  html2canvas($('mainContent'),{scale:1.5,backgroundColor:'#f0ede8'})
    .then(c=>{const a=document.createElement('a');a.download='tea_dissertation_dashboard.png';a.href=c.toDataURL('image/png');a.click();});
}
function downloadCSV(){
  const d=slice();
  const rows=[['Date','Export Price (USD/kg)','Export Price (LKR/kg)','Global Tea Price (USD/kg)','Export Quantity (kg)','Exchange Rate (LKR/USD)','Log Return','Rolling Vol 6M (%)','Rolling Vol 12M (%)']];
  d.labels.forEach((lbl,i)=>rows.push([lbl,d.price[i],d.priceLKR[i],d.global[i],d.quantity[i],d.fx[i],d.logRet[i]??'',d.vol6[i]??'',d.vol12[i]??'']));
  const blob=new Blob([rows.map(r=>r.join(',')).join('\\n')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='srilanka_tea_data_filtered.csv';a.click();
}

// ══ BOOT ═════════════════════════════════════════════════════════════
initCharts();
</script>
</body>
</html>
"""


def inject_data(html: str, data: dict) -> str:
    """Replace the embedded dataset in the HTML with freshly computed data."""
    import re
    new_data_js = "const R = " + json.dumps(data) + ";"
    # Replace the existing `const R = {...};` block
    pattern = r"const R = \{.*?\};"
    replaced, n = re.subn(pattern, new_data_js, html, count=1, flags=re.DOTALL)
    if n == 0:
        # fallback: just return original html if pattern not found
        return html
    return replaced


# ════════════════════════════════════════════════════════════════════════════
#  FLASK SERVER
# ════════════════════════════════════════════════════════════════════════════

def build_app(data: dict | None = None) -> "Flask":
    app = Flask(__name__)
    html_to_serve = DASHBOARD_HTML if data is None else inject_data(DASHBOARD_HTML, data)

    @app.route("/")
    def index():
        return render_template_string(html_to_serve)

    @app.route("/api/data")
    def api_data():
        return jsonify(data or {})

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


# ════════════════════════════════════════════════════════════════════════════
#  FALLBACK: serve via built-in http.server (no Flask)
# ════════════════════════════════════════════════════════════════════════════

def serve_plain(html: str, port: int):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        def log_message(self, fmt, *args):  # silence request logs
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"\n✅  Dashboard running at  http://127.0.0.1:{port}")
    print("   Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sri Lankan Tea Export Price Research Dashboard"
    )
    parser.add_argument("--port", type=int, default=5050, help="Port (default: 5050)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument(
        "--master",
        default="Tea_Export_Master_2008_2024.xlsx",
        help="Path to master Excel file",
    )
    parser.add_argument(
        "--global-prices",
        default="DateGlobal_Tea_Price.xlsx",
        help="Path to global tea price Excel file",
    )
    args = parser.parse_args()

    # ── Try to load fresh data from Excel files ──────────────────────────────
    fresh_data = None
    if HAS_DATA_LIBS:
        if os.path.exists(args.master) and os.path.exists(args.global_prices):
            print(f"📂  Loading data from:")
            print(f"    {args.master}")
            print(f"    {args.global_prices}")
            try:
                fresh_data = load_and_process(args.master, args.global_prices)
                n = fresh_data["meta"]["n"]
                lp = fresh_data["meta"]["last_price"]
                print(f"✅  Data loaded: {n} observations · latest price = ${lp:.3f}/kg")
            except Exception as exc:
                print(f"⚠️  Could not process Excel files: {exc}")
                print("    Falling back to pre-computed embedded dataset.")
        else:
            print("ℹ️  Excel files not found in current directory.")
            print("    Using pre-computed embedded dataset.")
            print(f"    (Expected: {args.master}  &  {args.global_prices})")
    else:
        print("ℹ️  pandas / numpy not installed — using embedded dataset.")
        print("    To enable live data loading:  pip install pandas numpy openpyxl")

    # ── Auto-open browser after a short delay ────────────────────────────────
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        def _open():
            import time
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n🍵  Sri Lankan Tea Export Price Dashboard")
    print(f"    {'─' * 46}")
    print(f"    URL  :  {url}")
    print(f"    Port :  {args.port}")
    print(f"    Data :  {'Live (Excel)' if fresh_data else 'Embedded (pre-computed)'}")
    print(f"    {'─' * 46}")
    print(f"    Press Ctrl+C to stop\n")

    # ── Start server ─────────────────────────────────────────────────────────
    if HAS_FLASK:
        app = build_app(fresh_data)
        # Suppress Flask startup banner
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    else:
        html = DASHBOARD_HTML if fresh_data is None else inject_data(DASHBOARD_HTML, fresh_data)
        serve_plain(html, args.port)


if __name__ == "__main__":
    main()
