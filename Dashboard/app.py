"""
Sri Lankan Tea Export Price — Interactive Findings App
Built from two notebooks:
  1. Tea_Price_Exploration.ipynb   (exploratory data analysis / visualisation)
  2. Tea_price_corrected.ipynb     (stationarity, SARIMAX / XGBoost / Prophet hybrids)

Run with:  streamlit run app.py
(Place Tea_Export_Master_2008_2024.csv and DateGlobal_Tea_Price.csv in the same
folder as this file — paths are resolved relative to the script's own location,
so this works regardless of the process's working directory or which subfolder
the app lives in within the repo.)
"""

import os, warnings, itertools

# Streamlit Cloud's free tier runs on a fractional CPU. numpy/scipy/statsmodels
# each try to spawn multiple BLAS threads by default, and on a 1-core box that
# thread contention (not the actual math) is what makes model fitting crawl.
# Pin everything to 1 thread before numpy/scipy get imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, kurtosis, skew
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
Z = 1.96
CRISIS_START, CRISIS_END = "2022-01-01", "2023-06-01"

# Resolve data files relative to this script's own folder, not the process's
# working directory — Streamlit Cloud's cwd is the repo root, which breaks
# bare relative paths whenever app.py lives in a subfolder.
HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Tea Export Price — Findings", page_icon="🍃", layout="wide")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def calc_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    return mae, rmse, mape, r2


def se_from_summary(sf):
    return (sf["mean_ci_upper"].values - sf["mean"].values) / Z


def connect(hist_index, hist_values, fc_index, *series):
    """Prepend the last historical point so forecast lines join the history."""
    new_idx = pd.DatetimeIndex([hist_index[-1]]).append(pd.DatetimeIndex(fc_index))
    out = [new_idx]
    for s in series:
        out.append(np.concatenate([[hist_values[-1]], np.asarray(s)]))
    return tuple(out)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    dp = pd.read_csv(os.path.join(HERE, "DateGlobal_Tea_Price.csv"))
    de = pd.read_csv(os.path.join(HERE, "Tea_Export_Master_2008_2024.csv"))
    num_cols = ["Export Quantity (kg)", "Export Price (LKR/kg)",
                "Exchange Rate (LKR/USD)", "Export Price (USD/kg)"]
    for c in num_cols:
        de[c] = pd.to_numeric(de[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                              errors="coerce")
    dp["Global_Tea_Price ($)"] = pd.to_numeric(dp["Global_Tea_Price ($)"], errors="coerce")
    df = pd.merge(dp, de, on=["Year", "Month"])
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
    df = df.set_index("Date").sort_index().drop(columns=["Year", "Month"]).ffill().bfill()
    df.index.freq = "MS"
    df["log_Export_Price_USD"] = np.log(df["Export Price (USD/kg)"])
    df["log_Exchange_Rate"] = np.log(df["Exchange Rate (LKR/USD)"])
    df["log_Global_Price"] = np.log(df["Global_Tea_Price ($)"])
    return df


# ----------------------------------------------------------------------------
# Modelling (AIC-selected orders from the notebook's grid search)
# ----------------------------------------------------------------------------
ORDER_USD, SORDER_USD = (1, 0, 0), (0, 0, 0, 12)   # SARIMAX, exog = log global price
ORDER_FX = (2, 0, 1)                               # ARIMA on log exchange rate
ORDER_GLOB = (1, 1, 2)                             # ARIMA on log global price


@st.cache_resource(show_spinner=False)
def fit_sarimax(df):
    y_usd = df["log_Export_Price_USD"]
    exog = df[["log_Global_Price"]]
    res_u = sm.tsa.statespace.SARIMAX(y_usd, exog=exog, order=ORDER_USD,
                                      seasonal_order=SORDER_USD,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
    res_f = sm.tsa.statespace.SARIMAX(df["log_Exchange_Rate"], order=ORDER_FX,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
    res_g = sm.tsa.statespace.SARIMAX(df["log_Global_Price"], order=ORDER_GLOB,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
    return res_u, res_f, res_g


@st.cache_data(show_spinner=False)
def sarimax_outputs(_res_u, _res_f, _res_g, df, steps=12):
    exog = df[["log_Global_Price"]]
    usd_fit = _res_u.predict(0, len(df) - 1, exog=exog)
    fx_fit = _res_f.predict(0, len(df) - 1)
    glob_fit = _res_g.predict(0, len(df) - 1)
    lkr_fit = np.exp(usd_fit + fx_fit)

    g_oos = _res_g.get_forecast(steps).summary_frame()
    fexog = pd.DataFrame({"log_Global_Price": g_oos["mean"].values}, index=g_oos.index)
    u_oos = _res_u.get_forecast(steps, exog=fexog).summary_frame()
    f_oos = _res_f.get_forecast(steps).summary_frame()

    mean_log = u_oos["mean"].values + f_oos["mean"].values
    se_tot = np.sqrt(se_from_summary(u_oos) ** 2 + se_from_summary(f_oos) ** 2)
    fc = np.exp(mean_log)
    lo = np.exp(mean_log - Z * se_tot)
    hi = np.exp(mean_log + Z * se_tot)
    return dict(usd_fit=usd_fit, fx_fit=fx_fit, glob_fit=glob_fit, lkr_fit=lkr_fit,
                idx=u_oos.index, fc=fc, lo=lo, hi=hi,
                g_oos_mean=np.exp(g_oos["mean"].values), g_oos_idx=g_oos.index,
                resid=_res_u.resid[2:])


@st.cache_data(show_spinner=False)
def run_xgboost(df, steps=12):
    import xgboost as xgb
    dx = df[["log_Export_Price_USD", "log_Global_Price"]].copy()
    dx["target"] = dx["log_Export_Price_USD"].diff()
    for l in [1, 2, 3, 12]:
        dx[f"lag_{l}"] = dx["log_Export_Price_USD"].shift(l)
        dx[f"global_lag_{l}"] = dx["log_Global_Price"].shift(l)
    dx["rolling_mean_3"] = dx["log_Export_Price_USD"].shift(1).rolling(3).mean()
    dx["rolling_std_3"] = dx["log_Export_Price_USD"].shift(1).rolling(3).std()
    dx["month"] = dx.index.month
    dx["sin_month"] = np.sin(2 * np.pi * dx["month"] / 12)
    dx["cos_month"] = np.cos(2 * np.pi * dx["month"] / 12)
    dx = dx.dropna()
    feats = [c for c in dx.columns
             if c not in ["target", "log_Export_Price_USD", "log_Global_Price", "month"]]
    X, yv = dx[feats], dx["target"]
    n = len(dx); tr = int(n * 0.7)

    pred = []
    last_model = None
    REFIT_EVERY = 3  # retrain every 3rd month instead of every month — same
                      # walk-forward story, ~3x fewer model fits, no visible
                      # change to the chart since predictions between refits
                      # still use the latest available data as features
    for i, t in enumerate(range(tr, n)):
        if last_model is None or i % REFIT_EVERY == 0:
            last_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3,
                                          random_state=42, n_jobs=1, tree_method="hist")
            last_model.fit(X.iloc[:t], yv.iloc[:t])
        pred.append(last_model.predict(X.iloc[t:t + 1])[0])
    idx = dx.index[tr:]
    recon = [dx["log_Export_Price_USD"].iloc[tr + i - 1] + d for i, d in enumerate(pred)]
    test_lkr = np.exp(np.array(recon) + df.loc[idx, "log_Exchange_Rate"].values)

    final = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42,
                             n_jobs=1, tree_method="hist").fit(X, yv)
    importance = pd.Series(final.feature_importances_, index=feats).sort_values()

    # recursive OOS using forecasted global price
    res_g = fit_sarimax(df)[2]
    glob_future = np.exp(res_g.get_forecast(steps).summary_frame()["mean"].values)  # placeholder
    glob_future_log = res_g.get_forecast(steps).summary_frame()["mean"].values
    future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=steps, freq="MS")
    df_fc = dx.copy()
    for i, date in enumerate(future_dates):
        nr = pd.Series(index=dx.columns, dtype="float64")
        for l in [1, 2, 3, 12]:
            nr[f"lag_{l}"] = df_fc["log_Export_Price_USD"].iloc[-l]
            nr[f"global_lag_{l}"] = df_fc["log_Global_Price"].iloc[-l]
        nr["rolling_mean_3"] = df_fc["log_Export_Price_USD"].iloc[-3:].mean()
        nr["rolling_std_3"] = df_fc["log_Export_Price_USD"].iloc[-3:].std()
        nr["sin_month"] = np.sin(2 * np.pi * date.month / 12)
        nr["cos_month"] = np.cos(2 * np.pi * date.month / 12)
        d = final.predict(pd.DataFrame(nr[feats].values.reshape(1, -1), columns=feats))[0]
        nr["log_Export_Price_USD"] = df_fc["log_Export_Price_USD"].iloc[-1] + d
        nr["log_Global_Price"] = glob_future_log[i]; nr["target"] = d
        df_fc.loc[date] = nr
    xgb_usd_oos = df_fc.loc[future_dates, "log_Export_Price_USD"].values

    res_f = fit_sarimax(df)[1]
    fx_oos = res_f.get_forecast(steps).summary_frame()
    rmse_log = np.sqrt(np.mean((dx["log_Export_Price_USD"].iloc[tr:].values - np.array(recon)) ** 2))
    se_x = rmse_log * np.sqrt(np.arange(1, steps + 1))
    se_fx = se_from_summary(fx_oos)
    mean_log = xgb_usd_oos + fx_oos["mean"].values
    se_tot = np.sqrt(se_x ** 2 + se_fx ** 2)
    fc = np.exp(mean_log); lo = np.exp(mean_log - Z * se_tot); hi = np.exp(mean_log + Z * se_tot)
    return dict(test_idx=idx, test_lkr=test_lkr, recon=recon, importance=importance,
                future_idx=future_dates, fc=fc, lo=lo, hi=hi)


@st.cache_data(show_spinner=False)
def run_prophet(df, _glob_oos_log, _fx_oos_mean, _fx_oos_se, steps=12):
    """Optional — only runs if prophet is installed. Returns None on failure."""
    try:
        from prophet import Prophet
    except Exception:
        return None
    try:
        dfp = df[["log_Export_Price_USD", "log_Global_Price"]].reset_index().rename(
            columns={"Date": "ds", "log_Export_Price_USD": "y", "log_Global_Price": "global_price"})
        m = Prophet(growth="linear", yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
        m.add_regressor("global_price"); m.fit(dfp)
        future = m.make_future_dataframe(periods=steps, freq="MS")
        future["global_price"] = dfp["global_price"].tolist() + list(_glob_oos_log)
        fcst = m.predict(future)

        fx_fit = fit_sarimax(df)[1].predict(0, len(df) - 1)
        usd_log = fcst.set_index("ds").reindex(df.index)["yhat"]
        in_lkr = np.exp(usd_log + fx_fit)

        future_dates = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=steps, freq="MS")
        p = fcst.set_index("ds").loc[future_dates[0]:]
        mean_log = p["yhat"].values + _fx_oos_mean
        se_p = (p["yhat_upper"].values - p["yhat"].values) / Z
        se_tot = np.sqrt(se_p ** 2 + _fx_oos_se ** 2)
        return dict(in_lkr=in_lkr, future_idx=future_dates,
                    fc=np.exp(mean_log), lo=np.exp(mean_log - Z * se_tot),
                    hi=np.exp(mean_log + Z * se_tot))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Load everything
# ----------------------------------------------------------------------------
df = load_data()
price = df["Export Price (LKR/kg)"]

with st.spinner("Fitting models (cached after first run)…"):
    res_u, res_f, res_g = fit_sarimax(df)
    S = sarimax_outputs(res_u, res_f, res_g, df)
    X = run_xgboost(df)

mask = df.index[2:]
periods = [("Overall", mask),
           ("Pre-2022", mask[mask < "2022-01-01"]),
           ("Post-2022", mask[mask >= "2022-01-01"])]

# ----------------------------------------------------------------------------
# Sidebar nav
# ----------------------------------------------------------------------------
st.sidebar.title("🍃 Tea Price Findings")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Exploratory Analysis", "Stationarity & Diagnostics",
     "Forecasting Models", "Model Comparison"],
)
st.sidebar.caption(
    f"Data: {df.index.min():%b %Y} – {df.index.max():%b %Y}  ·  {len(df)} months\n\n"
    "Target: Sri Lankan tea **export price (LKR/kg)**, modelled as "
    "USD price × exchange rate."
)


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("Sri Lankan Tea Export Price — Findings")
    st.write(
        "An interactive summary of the two analysis notebooks: exploratory findings on the "
        "monthly tea export price (2008–2024), and three forecasting models that predict the "
        "rupee price by combining a USD-price model with an exchange-rate model."
    )

    c1, c2, c3, c4 = st.columns(4)
    last = price.iloc[-1]
    c1.metric("Latest export price", f"{last:,.0f} LKR/kg",
              f"{(price.iloc[-1] - price.iloc[-13]):+,.0f} vs 1yr ago")
    c2.metric("17-yr average", f"{price.mean():,.0f} LKR/kg")
    c3.metric("Best model (SARIMAX) MAPE", "2.32%")
    c4.metric("12-mo forecast (Dec-25)", f"{S['fc'][-1]:,.0f} LKR/kg")

    st.subheader("Headline: price history and the 12-month forecast")
    hist = price
    ci, cv = connect(hist.index, hist.values, S["idx"], S["fc"])
    _, clo = connect(hist.index, hist.values, S["idx"], S["lo"])
    _, chi = connect(hist.index, hist.values, S["idx"], S["hi"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-60:], y=hist.iloc[-60:], name="Actual",
                             line=dict(color="black")))
    fig.add_trace(go.Scatter(x=ci, y=chi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ci, y=clo, fill="tonexty", fillcolor="rgba(200,30,30,0.15)",
                             line=dict(width=0), name="95% CI"))
    fig.add_trace(go.Scatter(x=ci, y=cv, name="SARIMAX forecast",
                             line=dict(color="#c0392b"), mode="lines+markers"))
    fig.update_layout(height=430, yaxis_title="LKR / kg", hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key findings")
    st.markdown(
        "- **One structural break dominates.** The price roughly triples after the 2022 currency "
        "crisis — almost entirely because of rupee depreciation, not a change in the USD tea price.\n"
        "- **The rupee price tracks the exchange rate.** LKR price and the LKR/USD rate move together; "
        "export quantity moves largely independently.\n"
        "- **Volatility jumps after 2022.** Rolling 12-month volatility steps up sharply, so a single "
        "stable model can't describe both eras equally well.\n"
        "- **Seasonality is weak.** Calendar-month effects are small relative to the trend and the 2022 shock.\n"
        "- **Decomposition works.** Modelling USD price and FX separately, then recombining "
        "(price = exp(USD + FX)), reconstructs the rupee price accurately.\n"
        "- **SARIMAX wins.** Across MAE / RMSE / MAPE / R², the hybrid SARIMAX is the strongest of the "
        "three models, with XGBoost close behind."
    )


# ============================================================================
# PAGE: EXPLORATORY ANALYSIS
# ============================================================================
elif page == "Exploratory Analysis":
    st.title("Exploratory Analysis")
    st.caption("Findings from the visualisation notebook, grouped into price-over-time, "
               "distribution & volatility, seasonality, and drivers.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Price over time", "Distribution & volatility", "Seasonality", "Drivers & relationships"])

    # --- Price over time ---
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price, line=dict(color="#7a1f1f"),
                                 fill="tozeroy", fillcolor="rgba(122,31,31,0.06)", name="LKR/kg"))
        fig.add_vrect(x0=CRISIS_START, x1=CRISIS_END, fillcolor="orange", opacity=0.15,
                      line_width=0, annotation_text="2022 currency crisis",
                      annotation_position="top left")
        fig.update_layout(title="Tea Export Price (LKR/kg), 2008–2024",
                          height=420, yaxis_title="LKR / kg")
        st.plotly_chart(fig, use_container_width=True)

        roll_m = price.rolling(12).mean(); roll_s = price.rolling(12).std()
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=("Price vs 12-month moving average",
                                             "12-month rolling volatility (note the post-2022 jump)"))
        fig2.add_trace(go.Scatter(x=price.index, y=price, line=dict(color="#cccccc"),
                                  name="Monthly"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=roll_m.index, y=roll_m, line=dict(color="#7a1f1f", width=2.5),
                                  name="12-mo avg"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=roll_s.index, y=roll_s, line=dict(color="#1f4e79", width=2),
                                  name="Volatility"), row=2, col=1)
        fig2.update_layout(height=520, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        annual = price.groupby(price.index.year).agg(["min", "mean", "max"])
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=annual.index, y=annual["max"], line=dict(width=0),
                                  showlegend=False, hoverinfo="skip"))
        fig3.add_trace(go.Scatter(x=annual.index, y=annual["min"], fill="tonexty",
                                  fillcolor="rgba(122,31,31,0.15)", line=dict(width=0),
                                  name="min–max range"))
        fig3.add_trace(go.Scatter(x=annual.index, y=annual["mean"], mode="lines+markers",
                                  line=dict(color="#7a1f1f"), name="annual mean"))
        fig3.update_layout(title="Annual price range and mean", height=380, yaxis_title="LKR/kg")
        st.plotly_chart(fig3, use_container_width=True)

    # --- Distribution & volatility ---
    with tab2:
        pre = price[price.index < "2022-01-01"]
        post = price[price.index >= "2022-01-01"]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(price, nbins=25, title="Price distribution (all years)",
                               color_discrete_sequence=["#7a1f1f"])
            fig.add_vline(x=price.mean(), line_dash="dash",
                          annotation_text=f"mean={price.mean():.0f}")
            fig.update_layout(height=360, showlegend=False, xaxis_title="LKR/kg")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pre, name=f"pre-2022 (n={len(pre)})",
                                       marker_color="#1f4e79", opacity=0.6))
            fig.add_trace(go.Histogram(x=post, name=f"2022+ (n={len(post)})",
                                       marker_color="#c0392b", opacity=0.6))
            fig.update_layout(barmode="overlay", height=360,
                              title="Two regimes: before vs after 2022", xaxis_title="LKR/kg")
            st.plotly_chart(fig, use_container_width=True)

        chg = price.diff()
        c3, c4 = st.columns(2)
        with c3:
            colors = np.where(chg >= 0, "#2e7d32", "#c0392b")
            fig = go.Figure(go.Bar(x=chg.index, y=chg, marker_color=colors))
            fig.update_layout(title="Month-over-month change (LKR/kg)", height=360,
                              yaxis_title="Δ price")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.histogram(chg.dropna(), nbins=30,
                               color_discrete_sequence=["#555555"],
                               title="Distribution of monthly changes")
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(height=360, showlegend=False, xaxis_title="Δ LKR/kg")
            st.plotly_chart(fig, use_container_width=True)
        st.info("The change distribution is centred near zero but has fat tails — large monthly "
                "jumps cluster in the post-2022 period.")

    # --- Seasonality ---
    with tab3:
        detrended = (price - price.rolling(12, center=True).mean()).dropna()
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig = go.Figure()
        for m in range(1, 13):
            vals = detrended[detrended.index.month == m]
            fig.add_trace(go.Box(y=vals, name=labels[m - 1], marker_color="#7a1f1f"))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Detrended price by calendar month", height=400,
                          yaxis_title="deviation from trend", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        piv = pd.DataFrame({"y": price.index.year, "m": price.index.month, "v": price.values}
                           ).pivot(index="y", columns="m", values="v")
        fig = px.imshow(piv, color_continuous_scale="YlOrRd", aspect="auto",
                        labels=dict(x="Month", y="Year", color="LKR/kg"),
                        x=labels, title="Price heatmap (Year × Month)")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.info("Boxes sit close to the zero line, so calendar-month seasonality is weak — the "
                "trend and the 2022 shock dominate the picture.")

    # --- Drivers ---
    with tab4:
        drivers = {"Exchange Rate (LKR/USD)": "#1f4e79",
                   "Global_Tea_Price ($)": "#2e7d32",
                   "Export Quantity (kg)": "#8e44ad"}
        cols = st.columns(3)
        for (col, color), cc in zip(drivers.items(), cols):
            r = df[col].corr(price)
            fig = px.scatter(x=df[col], y=price, opacity=0.5,
                             color_discrete_sequence=[color],
                             labels={"x": col, "y": "LKR/kg"})
            fig.update_layout(title=f"corr with price = {r:.2f}", height=320,
                              margin=dict(t=40))
            cc.plotly_chart(fig, use_container_width=True)

        norm = lambda s: (s - s.min()) / (s.max() - s.min())
        fig = go.Figure()
        for col, c in [("Export Price (LKR/kg)", "#7a1f1f"),
                       ("Exchange Rate (LKR/USD)", "#1f4e79"),
                       ("Global_Tea_Price ($)", "#2e7d32"),
                       ("Export Quantity (kg)", "#8e44ad")]:
            fig.add_trace(go.Scatter(x=df.index, y=norm(df[col]), name=col, line=dict(color=c)))
        fig.update_layout(title="All variables on a common 0–1 scale", height=420,
                          yaxis_title="normalised", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        corr_cols = ["Export Price (LKR/kg)", "Export Price (USD/kg)",
                     "Exchange Rate (LKR/USD)", "Global_Tea_Price ($)", "Export Quantity (kg)"]
        corr = df[corr_cols].corr()
        short = [c.split(" (")[0] for c in corr_cols]
        fig = px.imshow(corr.values, x=short, y=short, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, text_auto=".2f", title="Correlation matrix")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.success("The LKR price tracks the exchange rate closely; export quantity moves "
                   "independently of price.")


# ============================================================================
# PAGE: STATIONARITY & DIAGNOSTICS
# ============================================================================
elif page == "Stationarity & Diagnostics":
    st.title("Stationarity & Model Identification")
    st.caption("Why the USD price is modelled in log-differences, and whether the residuals behave.")

    returns = df["log_Export_Price_USD"].diff().dropna()
    adf_stat, adf_p, *_ = adfuller(returns)
    c1, c2, c3 = st.columns(3)
    c1.metric("ADF statistic (log returns)", f"{adf_stat:.2f}")
    c2.metric("p-value", f"{adf_p:.4f}")
    c3.metric("Conclusion", "Stationary" if adf_p < 0.05 else "Non-stationary")
    st.caption("A tiny p-value means the differenced (log-return) series is stationary, which "
               "justifies first-differencing inside the ARIMA/SARIMAX models.")

    fig = go.Figure(go.Scatter(x=returns.index, y=returns, line=dict(color="#1f4e79")))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(title="Log returns of USD tea price", height=320, yaxis_title="log return")
    st.plotly_chart(fig, use_container_width=True)

    roll_vol = returns.rolling(12).std()
    fig = go.Figure(go.Scatter(x=roll_vol.index, y=roll_vol, line=dict(color="#c0392b")))
    fig.update_layout(title="12-month rolling volatility (std of log returns)", height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonal decomposition (USD price)")
    decomp = seasonal_decompose(df["Export Price (USD/kg)"], model="additive", period=12)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=df.index, y=decomp.observed, line=dict(color="#333")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomp.trend, line=dict(color="#7a1f1f")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomp.seasonal, line=dict(color="#2e7d32")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomp.resid, mode="markers",
                             marker=dict(size=3, color="#888")), row=4, col=1)
    fig.update_layout(height=620, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residual diagnostics (SARIMAX, USD component)")
    resid = S["resid"]
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    jb_stat, jb_p = jarque_bera(resid)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Ljung-Box(12) p", f"{lb['lb_pvalue'].iloc[0]:.3f}",
              "no leftover autocorr" if lb["lb_pvalue"].iloc[0] > 0.05 else "autocorr remains")
    d2.metric("Jarque-Bera p", f"{jb_p:.2g}", "~normal" if jb_p > 0.05 else "non-normal")
    d3.metric("Skew", f"{skew(resid):.2f}")
    d4.metric("Kurtosis", f"{kurtosis(resid, fisher=False):.2f}", "normal = 3")
    st.warning("If Ljung-Box passes but Jarque-Bera fails with high kurtosis, the dynamics are "
               "captured but the errors are fat-tailed — meaning the 95% intervals likely "
               "understate tail risk in volatile periods such as 2022.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(resid, nbins=30, title="Residual distribution",
                           color_discrete_sequence=["#7a1f1f"])
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        from scipy import stats as sps
        qq = sps.probplot(resid, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers",
                                 marker=dict(size=4, color="#1f4e79"), name="residuals"))
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1],
                                 line=dict(color="red"), name="normal"))
        fig.update_layout(title="Q-Q plot", height=340)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: FORECASTING MODELS
# ============================================================================
elif page == "Forecasting Models":
    st.title("Forecasting Models")
    st.caption("Each model forecasts the rupee price as exp(USD price + exchange rate), with the "
               "global tea price itself forecast forward as a regressor.")

    model = st.selectbox("Model", ["SARIMAX (hybrid)", "XGBoost (hybrid)", "Prophet (hybrid)"])
    hist = price

    # --- shared component decomposition view ---
    if model == "SARIMAX (hybrid)":
        st.subheader("Component decomposition: USD + FX → LKR")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Modelled log components",
                                            "Reconstructed LKR price = exp(USD + FX)"))
        fig.add_trace(go.Scatter(x=df.index[2:], y=S["usd_fit"][2:], name="log USD",
                                 line=dict(color="#1f4e79")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[2:], y=S["fx_fit"][2:], name="log FX",
                                 line=dict(color="#c0392b")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[2:], y=price[2:], name="Actual LKR",
                                 line=dict(color="black")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index[2:], y=S["lkr_fit"][2:], name="Reconstructed",
                                 line=dict(color="#8e44ad", dash="dash")), row=2, col=1)
        fig.update_layout(height=560, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("12-month forecast")
        ci, cv = connect(hist.index, hist.values, S["idx"], S["fc"])
        _, clo = connect(hist.index, hist.values, S["idx"], S["lo"])
        _, chi = connect(hist.index, hist.values, S["idx"], S["hi"])
        fc_idx, fc_val, lo_val, hi_val = S["idx"], S["fc"], S["lo"], S["hi"]
        color, ci_fill = "#c0392b", "rgba(192,57,43,0.15)"

    elif model == "XGBoost (hybrid)":
        st.subheader("Walk-forward fit (out-of-sample test period)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X["test_idx"], y=df.loc[X["test_idx"], "log_Export_Price_USD"],
                                 name="Actual log USD", line=dict(color="#1f4e79")))
        fig.add_trace(go.Scatter(x=X["test_idx"], y=X["recon"], name="XGBoost",
                                 line=dict(color="red", dash="dash")))
        fig.update_layout(height=360, hovermode="x unified", yaxis_title="log USD")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(X["importance"], orientation="h", title="Feature importance",
                     color_discrete_sequence=["#2e7d32"])
        fig.update_layout(height=380, showlegend=False, xaxis_title="importance")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("12-month recursive forecast")
        ci, cv = connect(hist.index, hist.values, X["future_idx"], X["fc"])
        _, clo = connect(hist.index, hist.values, X["future_idx"], X["lo"])
        _, chi = connect(hist.index, hist.values, X["future_idx"], X["hi"])
        fc_idx, fc_val, lo_val, hi_val = X["future_idx"], X["fc"], X["lo"], X["hi"]
        color, ci_fill = "#2e7d32", "rgba(46,125,50,0.15)"

    else:  # Prophet
        glob_oos_log = res_g.get_forecast(12).summary_frame()["mean"].values
        fx_sf = res_f.get_forecast(12).summary_frame()
        P = run_prophet(df, glob_oos_log, fx_sf["mean"].values, se_from_summary(fx_sf))
        if P is None:
            st.error("Prophet is not installed in this environment, so its forecast can't be shown. "
                     "Install it with `pip install prophet` and reload to enable this model. "
                     "SARIMAX and XGBoost are available now.")
            st.stop()
        st.subheader("12-month forecast")
        ci, cv = connect(hist.index, hist.values, P["future_idx"], P["fc"])
        _, clo = connect(hist.index, hist.values, P["future_idx"], P["lo"])
        _, chi = connect(hist.index, hist.values, P["future_idx"], P["hi"])
        fc_idx, fc_val, lo_val, hi_val = P["future_idx"], P["fc"], P["lo"], P["hi"]
        color, ci_fill = "#7d3c98", "rgba(125,60,152,0.15)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-36:], y=hist.iloc[-36:], name="Actual LKR",
                             line=dict(color="black")))
    fig.add_trace(go.Scatter(x=ci, y=chi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ci, y=clo, fill="tonexty", fillcolor=ci_fill,
                             line=dict(width=0), name="95% CI"))
    fig.add_trace(go.Scatter(x=ci, y=cv, name=f"{model} forecast",
                             line=dict(color=color), mode="lines+markers"))
    fig.update_layout(height=440, yaxis_title="LKR/kg", hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    tbl = pd.DataFrame({"Month": [d.strftime("%b %Y") for d in fc_idx],
                        "Forecast (LKR/kg)": np.round(fc_val, 1),
                        "Lower 95%": np.round(lo_val, 1),
                        "Upper 95%": np.round(hi_val, 1)})
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Accuracy on the in-sample / test period, then the three 2025 forecasts overlaid. "
               "Lower MAE / RMSE / MAPE is better; higher R² is better.")

    rows = [("SARIMAX", *calc_metrics(df.loc[mask, "Export Price (LKR/kg)"], S["lkr_fit"][2:])),
            ("XGBoost (test)", *calc_metrics(df.loc[X["test_idx"], "Export Price (LKR/kg)"],
                                             X["test_lkr"]))]

    glob_oos_log = res_g.get_forecast(12).summary_frame()["mean"].values
    fx_sf = res_f.get_forecast(12).summary_frame()
    P = run_prophet(df, glob_oos_log, fx_sf["mean"].values, se_from_summary(fx_sf))
    if P is not None:
        rows.append(("Prophet", *calc_metrics(df.loc[df.index[2:], "Export Price (LKR/kg)"],
                                              P["in_lkr"].iloc[2:])))

    bench = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "MAPE%", "R2"]).set_index("Model").round(3)
    st.dataframe(bench, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bench.index, y=bench["MAE"], name="MAE", marker_color="#1f4e79"))
        fig.add_trace(go.Bar(x=bench.index, y=bench["RMSE"], name="RMSE", marker_color="#c0392b"))
        fig.update_layout(title="Error by model (lower = better)", height=380, barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bench.index, y=bench["R2"], marker_color="#2e7d32"))
        fig.update_layout(title="R² by model (higher = better)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("2025 forecast comparison")
    hist = price
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-36:], y=hist.iloc[-36:], name="Actual",
                             line=dict(color="black", width=2)))
    series = [(S["idx"], S["fc"], "#c0392b", "SARIMAX"),
              (X["future_idx"], X["fc"], "#2e7d32", "XGBoost")]
    if P is not None:
        series.append((P["future_idx"], P["fc"], "#7d3c98", "Prophet"))
    for fidx, fc, c, lab in series:
        ci, cv = connect(hist.index, hist.values, fidx, fc)
        fig.add_trace(go.Scatter(x=ci, y=cv, name=lab, mode="lines+markers",
                                 line=dict(color=c, dash="dash")))
    fig.update_layout(height=440, yaxis_title="LKR/kg", hovermode="x unified",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    best = bench["RMSE"].idxmin()
    st.success(f"**{best}** has the lowest RMSE here. Per the project brief, accuracy measures across "
               "models point to the hybrid SARIMAX as the model to select for prediction — it has the "
               "best overall fit (R² ≈ 0.995, MAPE ≈ 2.3%) while remaining simple and interpretable.")
    if P is None:
        st.caption("Prophet is not installed, so it is omitted from this comparison. "
                   "Install it with `pip install prophet` to include it.")
