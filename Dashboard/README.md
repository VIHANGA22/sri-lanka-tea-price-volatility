# 🍃 Sri Lankan Tea Export Price — Findings Dashboard

Interactive Streamlit app summarising two notebooks:
- `Tea_Price_Exploration.ipynb` — exploratory data analysis
- `Tea_price_forecasting.ipynb` — stationarity checks + SARIMAX / XGBoost / Prophet hybrid forecasts

## 1. Data files

Included in this package, already validated against the app's loading logic:

- `Tea_Export_Master_2008_2024.csv`
- `DateGlobal_Tea_Price.csv`

204 months, Jan 2008 – Dec 2024, no missing values after merge. Keep both files
in the same folder as `app.py` — the app reads them from the project root at startup.

## 2. Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Model fitting (SARIMAX + XGBoost) is
cached with `@st.cache_resource` / `@st.cache_data`, so only the first load is slow.

## 3. Deploy on Streamlit Community Cloud via GitHub

1. **Push this folder to a new GitHub repo:**
   ```bash
   cd tea-dashboard
   git init
   git add .
   git commit -m "Tea export price findings dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
   Make sure the two CSVs above are committed too — Streamlit Cloud only sees
   what's in the repo.

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New app"** → pick your repo/branch.
   - Set **Main file path** to `app.py`.
   - Click **Deploy**.

   `requirements.txt` and `.streamlit/config.toml` in this folder are picked up
   automatically — no extra configuration needed.

3. **First build** takes a couple of minutes (installs `statsmodels`, `xgboost`,
   etc.). Subsequent pushes to `main` redeploy automatically.

## 4. Optional: enabling the Prophet model

The "Prophet (hybrid)" option on the Forecasting page is optional — the app
detects at runtime whether `prophet` is installed and shows a friendly message
if it isn't, rather than crashing. It's left out of `requirements.txt` by
default because it needs a compiled Stan backend and noticeably slows down
Streamlit Cloud's build. To include it, add this line to `requirements.txt`:

```
prophet>=1.1
```

## Project structure

```
tea-dashboard/
├── app.py                              # Streamlit app (5 pages: Overview, EDA,
│                                        #   Stationarity & Diagnostics, Forecasting
│                                        #   Models, Model Comparison)
├── requirements.txt
├── .streamlit/config.toml              # tea/maroon theme
├── .gitignore
├── Tea_Export_Master_2008_2024.csv
└── DateGlobal_Tea_Price.csv
```
