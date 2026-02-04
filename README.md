# Sri Lankan Tea Export Price Volatility Analysis

## Project Overview
This project analyses the behaviour of Sri Lankan tea export prices over time, focusing on price volatility and structural changes. The study uses monthly data from January 2008 to December 2024, with prices expressed in USD per kilogram for international comparability.

## Data
- Tea export price and quantity (monthly): Sri Lanka Tea Board (official monthly statistics).
- Exchange rate (LKR/USD): Central Bank of Sri Lanka (monthly averages), with supplementary historical exchange-rate data used only where official records were incomplete.
- Processed variable: Export Price (USD/kg) computed using monthly average exchange rates.

## Work Completed (IPD Evidence)
- Data consolidation into a master dataset (raw Excel in `data/raw/`)
- Data cleaning and validation (processed Excel in `data/processed/`)
- Exploratory analysis and reproducible workflow (notebook in `notebooks/`)
- Outputs:
  - Price time series plot: `outputs/figures/price_timeseries.png`
  - Log returns plot: `outputs/figures/log_returns.png`
  - Summary statistics table: `outputs/tables/summary_statistics.csv`

## Repository Structure
- `notebooks/` : Jupyter notebooks (code)
- `data/raw/` : raw master dataset (Excel)
- `data/processed/` : cleaned dataset used for analysis
- `outputs/figures/` : generated plots
- `outputs/tables/` : generated tables
- `docs/appendix/` : supporting materials for IPD/dissertation

## How to Run
Install requirements:
```bash
pip install -r requirements.txt
