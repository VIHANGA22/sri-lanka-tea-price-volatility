# 🍃 Sri Lanka Tea Price Analysis Dashboard

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Put your Excel file in the same folder as `tea_dashboard.py`.  
Accepted filenames (auto-detected):
- `Cleaned_Merged_Tea_Dataset.xlsx`
- `Cleaned_Merged_Tea_Dataset__1_.xlsx`

Or simply upload via the **sidebar uploader** at runtime.

### 3. Launch the dashboard
```bash
streamlit run tea_dashboard.py
```

The browser opens automatically at `http://localhost:8501`

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| 📈 Price Trends | SL export price (USD & LKR) with annotated peaks |
| 🌍 Global Comparison | SL vs Global price overlay + correlation scatter |
| 💱 Exchange Rate & Quantity | FX trend, quantity trend, scatter relationships |
| 🔮 ARIMA Forecast | Forecast + shaded 95% CI + forecast table |
| 📊 Statistical Analysis | Log returns, rolling volatility, descriptive stats |

## Sidebar Controls
- **Upload Dataset** — drag & drop your .xlsx
- **Year Range** — slider to filter years
- **Months** — optional monthly filter
- **ARIMA p, d, q** — tune model order live
- **Forecast horizon** — 6 to 24 months

## Column Requirements
Your Excel file must contain:
```
Year | Month | Export Quantity (kg) | Export Price (LKR/kg) |
Exchange Rate (LKR/USD) | Export Price (USD/kg) |
Global_Tea_Price ($) | Log_Return
```
