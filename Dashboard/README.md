# Tea Export Price — Interactive Findings App

A small Streamlit app that presents the findings from the two analysis notebooks
(`Tea_Price_Exploration.ipynb` and `Tea_price_corrected.ipynb`) for Sri Lankan
monthly tea export prices, 2008–2024.

## What's inside

Five sections (sidebar):

1. **Overview** – headline numbers, the 12-month forecast, and the key findings.
2. **Exploratory Analysis** – price over time, distribution & volatility, seasonality,
   and drivers / correlations.
3. **Stationarity & Diagnostics** – ADF test, log returns, rolling volatility,
   seasonal decomposition, and SARIMAX residual diagnostics.
4. **Forecasting Models** – SARIMAX, XGBoost and Prophet hybrids (each forecasts the
   rupee price as `exp(USD price + exchange rate)`), with component decomposition and CIs.
5. **Model Comparison** – MAE / RMSE / MAPE / R² table, charts, and the three forecasts overlaid.

The model orders are the AIC-selected ones from the notebook's grid search
(USD: SARIMAX(1,0,0) with global price as exog; FX: ARIMA(2,0,1); global: ARIMA(1,1,2)).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep these two CSVs in the same folder as `app.py`:
`Tea_Export_Master_2008_2024.csv` and `DateGlobal_Tea_Price.csv` (already included).

The first load fits the models (a few seconds) and then caches them, so navigation is instant.

## Notes

- **Prophet is optional.** If `prophet` isn't installed the app still runs — the
  Prophet model is simply skipped with a notice. To enable it, uncomment `prophet`
  in `requirements.txt` and reinstall.
- All charts are interactive (zoom, hover, toggle series via the legend).
