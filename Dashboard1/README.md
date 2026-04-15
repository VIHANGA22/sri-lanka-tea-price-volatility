# Sri Lanka Tea Export Price Volatility Dashboard

**Decision-Support Analytics Tool · 2008–2024**

A production-grade Streamlit dashboard presenting time series analysis and 12-month price forecasting for Sri Lankan tea exports. Built for business professionals, policy makers, and market analysts.

---

## Overview

This dashboard visualises the findings of a research project on Sri Lankan tea export price volatility (2008–2024), combining:

- Historical price and market data analysis
- Rolling volatility modelling (6M & 12M windows)
- ARIMA(1,0,0) forecasting on log returns
- Exchange rate impact analysis
- Decision-oriented insights for stakeholders

---

## Features

| Section | Description |
|---|---|
| Executive Overview | KPI cards: latest price, exchange rate, global price, forecast direction |
| Price & Market Trends | Sri Lanka vs global benchmark price (2008–2024) |
| Exchange Rate Impact | LKR/USD trend + scatter correlation with export price |
| Volatility & Risk | 6M/12M rolling volatility with spike detection |
| 12-Month Forecast | ARIMA price forecast with 95% confidence interval (log-return converted to price scale) |
| Model Performance | MAE, RMSE, MAPE on 80/20 train-test split |
| Decision Insights | 4 concise research-based insights for decision makers |

---

## Project Structure

```
.
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── Cleaned_Merged_Tea_Dataset.csv  # Dataset (place in same directory)
```

---

## Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place your dataset file in the same directory as `app.py` and ensure it is named:

```
Cleaned_Merged_Tea_Dataset.csv
```

If your file is `.xlsx`, convert it first:

```python
import pandas as pd
df = pd.read_excel("Cleaned_Merged_Tea_Dataset.xlsx")
df.to_csv("Cleaned_Merged_Tea_Dataset.csv")
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Data Requirements

The dataset must contain the following columns:

| Column | Description |
|---|---|
| `Export Quantity (kg)` | Monthly export volume |
| `Export Price (LKR/kg)` | Price in Sri Lankan Rupees |
| `Exchange Rate (LKR/USD)` | Monthly average exchange rate |
| `Export Price (USD/kg)` | USD-denominated export price |
| `Global_Tea_Price ($)` | Global benchmark price |
| `Log_Return` | Log returns of export price |

The index must be a datetime (monthly frequency, `MS` or `M`).

---

## Model Notes

- **Model:** ARIMA(1,0,0) fitted on log returns of export price
- **Validation:** 80/20 train-test split
- **Forecast horizon:** 12 months
- **Confidence interval:** 95%, back-transformed to price scale via `P(t) × exp(return_forecast)`
- **MAPE caveat:** MAPE is computed excluding near-zero return observations; MAE and RMSE are the primary performance indicators

---

## Design

The dashboard uses a minimal corporate design system:

- **Primary:** Navy `#0D2137` / Blue `#2E6DA4`
- **Background:** Light grey `#F4F6F8` / White
- **Forecast:** Red `#C0392B`
- **Confidence band:** Soft blue `rgba(46,109,164,0.15)`
- **Typography:** IBM Plex Sans + IBM Plex Mono

---

## Deployment (Streamlit Cloud)

1. Push project files to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository and set `app.py` as the entry point
4. Add dataset to repository or configure secrets for remote data access

---

*For analytical and research purposes only. Not intended as financial or investment advice.*
