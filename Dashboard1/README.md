# 🍃 Sri Lanka Tea Export Price Volatility Dashboard
### A Decision-Support System for Exporters, Policy Makers & Analysts (2008–2024)

---

## 📋 Project Description

This is a fully interactive, production-ready **Streamlit dashboard** built to analyse and forecast Sri Lanka's tea export price volatility from **2008 to 2024**. It was developed as part of an academic research project combining time series analysis, ARIMA modelling, and data visualisation into a single decision-support tool.

The dashboard is designed for **both technical and non-technical users** — all insights are explained in plain language, and all charts include interactive hover tooltips and zoom support.

---

## 🎯 Target Users

| User | How they benefit |
|---|---|
| **Tea Exporters** | Price outlook, hedging signals, forward contract timing |
| **Policy Makers** | Exchange rate impact analysis, volatility risk monitoring |
| **Business Analysts** | Historical trends, ARIMA forecast data, model validation |
| **Researchers** | Full data explorer, downloadable filtered CSVs |

---

## ✨ Dashboard Features

### 1. 📊 Overview — KPI Cards
- Latest Export Price (USD/kg)
- Current Exchange Rate (LKR/USD)
- Global Tea Benchmark Price
- Year-over-Year % Price Change
- Executive summary insight box

### 2. 📈 Price Analysis
- Interactive line chart comparing Sri Lanka export price vs global tea price
- Filter by year range and monthly/yearly aggregation
- Divergence annotations during crisis periods

### 3. 💱 Exchange Rate Analysis
- LKR/USD trend chart with shaded fill
- Scatter plot: Exchange Rate vs Export Price with trend line
- Correlation coefficient and policy recommendation insight

### 4. 📦 Export Quantity Analysis
- Monthly/yearly export volume bar chart
- Scatter plot: Quantity vs Price with trend line
- Supply-demand dynamics insight box

### 5. ⚡ Volatility Analysis
- 6-month and 12-month rolling annualised volatility
- Automatic red-shaded **high-risk period** detection
- Peak volatility annotation with date label

### 6. 🔮 Price Forecast (ARIMA)
- Configurable ARIMA(p, d, q) order from sidebar
- 6–24 month forecast horizon (adjustable)
- 90% Confidence Interval shown as pink band
- Month-by-month forecast table (expandable)
- Trend interpretation: Increasing / Decreasing / Stable

### 7. 🧪 Model Performance
- MAE, RMSE, MAPE displayed as metric cards
- Explanation of MAPE inflation near-zero issue
- Residual plot and histogram (expandable)

### 8. 🎯 Decision Support Panel
- Actionable insight cards for:
  - Tea Exporters
  - Policy Makers
  - Supply vs Demand
  - Global Market Position

### 9. 🗃️ Raw Data Explorer
- Full filtered data table
- One-click CSV download

---

## 🚀 How to Run — Step-by-Step

### Step 1 — Clone or Download the Project

```bash
# If using Git
git clone https://github.com/your-username/tea-export-dashboard.git
cd tea-export-dashboard

# Or simply download and unzip the project folder
cd sri_lanka_tea_dashboard
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Python 3.9+ recommended
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at:
```
http://localhost:8501
```

---

## 📂 Required CSV Format (if using your own data)

Upload your CSV via the sidebar. The file must contain these columns:

| Column | Description | Example |
|---|---|---|
| `Year` | Calendar year | 2020 |
| `Month` | Month number (1–12) | 6 |
| `Export_Qty_kg` | Export volume in kg | 285000000 |
| `Export_Price_LKR` | Price in LKR/kg | 2100.50 |
| `Exchange_Rate` | LKR per 1 USD | 203.5 |
| `Export_Price_USD` | Price in USD/kg | 5.8420 |
| `Global_Tea_Price` | World tea price USD/kg | 5.2100 |

**Optional pre-computed columns** (will be auto-calculated if missing):

| Column | Formula |
|---|---|
| `Log_Return` | `ln(Price_t / Price_{t-1})` |
| `Rolling_Vol_6m` | 6-month rolling std of log returns × √12 |
| `Rolling_Vol_12m` | 12-month rolling std of log returns × √12 |

> If no file is uploaded, the dashboard uses **realistic synthetic data** (2008–2024) that mirrors the research dataset's statistical properties.

---

## ⚙️ Sidebar Controls

| Control | Description |
|---|---|
| **Upload CSV** | Load your own dataset |
| **Year Range Slider** | Filter all charts to a date window |
| **Aggregation** | Switch between Monthly and Yearly views |
| **p, d, q** | Set ARIMA model order |
| **Forecast Horizon** | Choose 6–24 months ahead |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32 | Dashboard framework |
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `matplotlib` | ≥ 3.7 | Chart rendering |
| `seaborn` | ≥ 0.12 | Statistical plots |
| `statsmodels` | ≥ 0.14 | ARIMA modelling |
| `scikit-learn` | ≥ 1.3 | Validation metrics (MAE, RMSE) |
| `openpyxl` | ≥ 3.1 | Excel export support |

---

## 🧠 Methodology Notes

### ARIMA Model
- The default ARIMA(1,1,1) is suited for the tea price series which exhibits a unit root (I(1) process)
- The model is trained on 85% of the historical data; the remaining 15% is used for out-of-sample validation
- ARIMA order can be adjusted via the sidebar — use ADF test results as guidance

### Volatility
- Log returns: `r_t = ln(P_t / P_{t-1})`
- Annualised volatility: `σ_annual = σ_monthly × √12`
- High-risk threshold: 30% above the long-run average volatility

### MAPE Caveat
MAPE (Mean Absolute Percentage Error) is computed on price levels. When applied to log-returns (which can be very close to zero), the denominator becomes tiny and MAPE values appear artificially large. **Always prioritise MAE and RMSE** for model evaluation.

---

## 📸 Dashboard Sections At a Glance

```
┌─────────────────────────────────────────────────┐
│  🍃 Sri Lanka Tea Export Price Volatility        │
│     Decision-Support Dashboard 2008–2024         │
├──────────┬──────────┬──────────┬─────────────────┤
│ Price    │ Exchange │ Global   │ YoY Change      │
│ USD/kg   │ Rate     │ Price    │ ± %             │
├──────────┴──────────┴──────────┴─────────────────┤
│  📈 Price Analysis (SL vs Global)                │
│  💱 Exchange Rate Trend + Scatter                 │
│  📦 Quantity Trend + Scatter                     │
│  ⚡ 6m & 12m Rolling Volatility                  │
│  🔮 ARIMA Forecast + Confidence Intervals         │
│  🧪 Model Performance: MAE, RMSE, MAPE           │
│  🎯 Decision Support: 4-Panel Insights           │
└─────────────────────────────────────────────────┘
```

---

## 👥 Authors & Acknowledgements

- **Research Domain:** Sri Lanka Tea Export Economics
- **Analysis Period:** January 2008 – December 2024
- **Tools:** Python, Streamlit, statsmodels, pandas, matplotlib
- **Academic Context:** Submitted as part of a postgraduate research project on commodity price volatility

---

## 📄 Licence

This project is released for academic and research use. For commercial deployment, please contact the author.

---

*Dashboard version 1.0.0 — April 2024*
