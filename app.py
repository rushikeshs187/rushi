# app.py
# Streamlit EDA + ML comparison dashboard for financial markets
# Models: RandomForest (RF), Support Vector Machine (SVM), Artificial Neural Network (MLP)
# Data: auto-fetched from Yahoo via yfinance (no file uploads needed)

import os
import math
import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ---------------------------
# UI CONFIG
# ---------------------------
st.set_page_config(
    page_title="IBR Finance — EDA & ML Model Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS polish
st.markdown("""
<style>
.small { font-size:0.85rem; color:#666; }
.hr { border-top: 1px solid #eee; margin: 0.5rem 0 1rem 0; }
.metric-good { color:#0a8; font-weight:600; }
.metric-bad { color:#c33; font-weight:600; }
blockquote { border-left: 0.25rem solid #eee; padding: 0.25rem 0 0.25rem 0.75rem; color:#555; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DEFAULT UNIVERSES (developed + emerging) — small curated sets
# (Avoids scraping constituents; extend as desired)
# ---------------------------
UNIVERSES = {
    "US (Developed)": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"],
    "UK (Developed)": ["HSBA.L", "AZN.L", "BP.L", "ULVR.L", "GSK.L", "RIO.L"],
    "India (Emerging)": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    "Brazil (Emerging)": ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "B3SA3.SA"],
    "South Africa (Emerging)": ["NPN.JO", "AGL.JO", "BHG.JO", "SOL.JO"],
    "Crypto (Global)": ["BTC-USD", "ETH-USD"]
}

# Helpful intervals from Yahoo
INTERVALS = ["1d", "1wk", "1mo"]
PERIODS = ["1y", "2y", "5y", "10y", "max"]

# ---------------------------
# UTILS
# ---------------------------
def _safe_name(x):
    if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
        return ", ".join(map(str, x))
    return str(x)

def compute_indicators(df):
    """
    Input: long-form DataFrame with columns: ['Symbol', 'Date', 'Open','High','Low','Close','Adj Close','Volume']
    Returns df with added columns: 'Return','LogRet','SMA_20','SMA_50','EMA_12','EMA_26','MACD','MACD_Signal','RSI_14',
    'BB_M','BB_U','BB_L','Volatility_20'
    """
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    out = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").copy()
        px = g["Adj Close"].astype(float)

        # Returns
        g["Return"] = px.pct_change()
        g["LogRet"] = np.log(px).diff()

        # SMAs
        g["SMA_20"] = px.rolling(20).mean()
        g["SMA_50"] = px.rolling(50).mean()

        # EMAs
        g["EMA_12"] = px.ewm(span=12, adjust=False).mean()
        g["EMA_26"] = px.ewm(span=26, adjust=False).mean()

        # MACD
        g["MACD"] = g["EMA_12"] - g["EMA_26"]
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        g["BB_M"] = px.rolling(20).mean()
        bb_std = px.rolling(20).std()
        g["BB_U"] = g["BB_M"] + 2 * bb_std
        g["BB_L"] = g["BB_M"] - 2 * bb_std

        # RSI
        g["RSI_14"] = rsi(px, 14)

        # Rolling volatility (20d)
        g["Volatility_20"] = g["Return"].rolling(20).std() * np.sqrt(252)

        out.append(g)

    out = pd.concat(out, axis=0)
    return out

def prepare_ml_frame(df, horizon=1):
    """
    Build supervised dataset for next-day direction classification.
    Features: technicals + lagged returns
    Target: 1 if next LogRet > 0 else 0 (per symbol)
    """
    feats = [
        "Return", "LogRet",
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal",
        "RSI_14",
        "BB_M", "BB_U", "BB_L",
        "Volatility_20",
        # Add a few lags for returns (auto-correlations)
        "LagRet_1", "LagRet_2", "LagRet_5"
    ]
    frames = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").copy()
        g["LagRet_1"] = g["LogRet"].shift(1)
        g["LagRet_2"] = g["LogRet"].shift(2)
        g["LagRet_5"] = g["LogRet"].shift(5)
        g["Target"] = (g["LogRet"].shift(-horizon) > 0).astype(int)
        frames.append(g)
    X = pd.concat(frames, axis=0)
    X = X.dropna(subset=feats + ["Target"]).copy()
    return X, feats

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    rets = pd.Series(returns).dropna()
    if rets.empty:
        return np.nan
    excess = rets - (risk_free / periods_per_year)
    mu = excess.mean() * periods_per_year
    sigma = excess.std(ddof=0) * np.sqrt(periods_per_year)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return mu / sigma

def max_drawdown(cum_curve):
    s = pd.Series(cum_curve).fillna(method="ffill")
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    return dd.min()

# ---------------------------
# DATA FETCH
# ---------------------------
@st.cache_data(show_spinner=True)
def fetch_prices(symbols, period="5y", interval="1d"):
    """
    Return a tidy (long) DataFrame with columns:
    ['Symbol','Date','Open','High','Low','Close','Adj Close','Volume']
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    data = yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="ticker"
    )

    frames = []
    # When single symbol, yfinance returns columns leveled differently, handle both cases.
    if isinstance(data.columns, pd.MultiIndex):
        # multi-symbol
        for sym in symbols:
            if sym not in data.columns.levels[0] and sym not in data.columns.get_level_values(0).unique():
                continue
            g = data[sym].copy()
            g = g.reset_index().rename(columns={"index":"Date"})
            g.insert(0, "Symbol", sym)
            frames.append(g)
    else:
        # single symbol case
        g = data.reset_index().rename(columns={"index":"Date"})
        g.insert(0, "Symbol", symbols[0])
        frames.append(g)

    if not frames:
        return pd.DataFrame(columns=["Symbol","Date","Open","High","Low","Close","Adj Close","Volume"])

    df = pd.concat(frames, axis=0, ignore_index=True)
    # Normalize column names
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Ensure expected columns exist
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce types
    num_cols = ["Open","High","Low","Close","Adj Close","Volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure datetime
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # De-duplicate on (Symbol, Date)
    df = df.dropna(subset=["Date"]).sort_values(["Symbol","Date"])
    df = df[~df[["Date","Symbol"]].duplicated(keep="last")]
    return df

# ---------------------------
# SIDEBAR — controls
# ---------------------------
st.sidebar.title("Controls")

market = st.sidebar.selectbox("Market", list(UNIVERSES.keys()), index=0)
symbols_default = UNIVERSES[market]
symbols = st.sidebar.multiselect(
    "Symbols", options=symbols_default, default=symbols_default
)

period = st.sidebar.selectbox("History Window", PERIODS, index=2)  # default 5y
interval = st.sidebar.selectbox("Sampling Interval", INTERVALS, index=0)

eda_symbol = st.sidebar.selectbox(
    "Primary symbol for deep EDA", options=symbols if symbols else symbols_default, index=0
)

# ML params
st.sidebar.markdown("---")
st.sidebar.subheader("ML Settings")
horizon = st.sidebar.slider("Prediction horizon (days ahead)", 1, 5, 1, 1)
test_size_years = st.sidebar.slider("Test span (years)", 1, 5, 2, 1)
n_splits = st.sidebar.slider("CV splits (rolling)", 2, 5, 3, 1)

# Model hyperparams (kept modest for speed)
rf_trees = st.sidebar.slider("RF: n_estimators", 50, 400, 200, 50)
svm_c = st.sidebar.selectbox("SVM: C", [0.1, 0.5, 1.0, 2.0, 5.0], index=2)
mlp_hidden = st.sidebar.selectbox("ANN (MLP): hidden units", [32, 64, 128], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("IBR Objectives: EDA first • Compare RF, SVM, ANN • Cover developed & emerging markets • Sharpe/Drawdown & basic explainability")

# ---------------------------
# HEADER
# ---------------------------
st.title("IBR Finance — Exploratory Analysis & ML Model Comparison")
st.markdown(
    "A research‑grade dashboard aligned to your objectives: EDA → feature building → **RF vs SVM vs ANN** next‑day direction, "
    "with backtests and risk‑adjusted metrics. No uploads required — data fetched automatically from Yahoo Finance."
)

# ---------------------------
# LOAD DATA
# ---------------------------
if not symbols:
    st.warning("Please pick at least one symbol.")
    st.stop()

with st.spinner("Fetching market data..."):
    raw = fetch_prices(symbols, period=period, interval=interval)

if raw.empty:
    st.error("No data was returned for the selected inputs. Try a different market/period/interval.")
    st.stop()

# EDA-ready
raw = raw.sort_values(["Symbol","Date"]).reset_index(drop=True)

# ---------------------------
# EDA SECTION
# ---------------------------
st.markdown("## 1) Exploratory Data Analysis (EDA)")

# High-level coverage
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Symbols", f"{raw['Symbol'].nunique()}")
with c2:
    st.metric("Obs", f"{len(raw):,}")
with c3:
    st.metric("Date Range", f"{raw['Date'].min().date()} → {raw['Date'].max().date()}")
with c4:
    st.metric("Interval", interval)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# Picked symbol deep dive
sym_df = raw[raw["Symbol"] == eda_symbol].copy()
sym_df = sym_df.dropna(subset=["Adj Close"])
sym_df["Return"] = sym_df["Adj Close"].pct_change()
sym_df["LogRet"] = np.log(sym_df["Adj Close"]).diff()

# Price chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=sym_df["Date"], y=sym_df["Adj Close"], mode="lines", name=f"{eda_symbol} Adj Close"))
fig.update_layout(height=360, title=f"Price — {eda_symbol}", xaxis_title=None, yaxis_title="Adj Close")
st.plotly_chart(fig, use_container_width=True)

# Missingness & summary
colA, colB = st.columns([2, 1])
with colA:
    miss = sym_df[["Open","High","Low","Close","Adj Close","Volume"]].isna().mean().sort_values(ascending=False)
    miss_fig = px.bar(miss, title="Missingness by column (ratio)", labels={"value":"missing ratio","index":"column"})
    miss_fig.update_layout(height=300)
    st.plotly_chart(miss_fig, use_container_width=True)
with colB:
    desc = sym_df[["Adj Close","Return","LogRet","Volume"]].describe().T
    st.dataframe(desc, use_container_width=True)

# Rolling stats & distribution
col1, col2 = st.columns(2)
with col1:
    roll = sym_df.set_index("Date")["Return"].rolling(20).std() * np.sqrt(252)
    rfig = go.Figure()
    rfig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Ann. Vol (20d)"))
    rfig.update_layout(height=300, title="Rolling Annualized Volatility (20d)")
    st.plotly_chart(rfig, use_container_width=True)
with col2:
    hist = px.histogram(sym_df, x="Return", nbins=60, title="Return Distribution")
    hist.update_layout(height=300)
    st.plotly_chart(hist, use_container_width=True)

# Correlations (with technicals)
with st.spinner("Computing technical indicators..."):
    enriched = compute_indicators(raw)

corr_cols = ["Return","LogRet","SMA_20","SMA_50","EMA_12","EMA_26","MACD","MACD_Signal","RSI_14","Volatility_20"]
corr_df = enriched[enriched["Symbol"] == eda_symbol][corr_cols].dropna()
if len(corr_df) > 0:
    corr = corr_df.corr(numeric_only=True)
    cfig = px.imshow(corr, text_auto=True, aspect="auto", title=f"Correlation Heatmap — {eda_symbol}")
    cfig.update_layout(height=500)
    st.plotly_chart(cfig, use_container_width=True)
else:
    st.info("Not enough data for correlation heatmap yet (wait for indicators to warm up).")

# Multi-symbol snapshot (last value table)
latest = (
    enriched.sort_values("Date")
    .groupby("Symbol", as_index=False)
    .apply(lambda g: g.iloc[-1][["Symbol","Date","Adj Close","RSI_14","Volatility_20"]])
    .reset_index(drop=True)
)
st.markdown("#### Latest snapshot")
st.dataframe(latest.style.format({"Adj Close":"{:,.2f}", "RSI_14":"{:,.1f}", "Volatility_20":"{:,.2f}"}), use_container_width=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------------------------
# ML SECTION
# ---------------------------
st.markdown("## 2) Model Comparison — RF vs SVM vs ANN (MLP)")

# Build supervised ML frame across all selected symbols
ml_df, features = prepare_ml_frame(enriched, horizon=horizon)

# Split train/test chronologically (last 'test_size_years' years as test)
if ml_df.empty:
    st.error("Insufficient data after feature engineering. Try increasing period or using daily interval.")
    st.stop()

# Determine cutoff date for test
max_date = ml_df["Date"].max()
cutoff = max_date - pd.DateOffset(years=test_size_years)
train = ml_df[ml_df["Date"] < cutoff].copy()
test = ml_df[ml_df["Date"] >= cutoff].copy()

if train["Target"].nunique() < 2 or test["Target"].nunique() < 2:
    st.warning("Target class imbalance or insufficient variety in train/test. Consider a longer period.")
    st.stop()

X_train = train[features].values
y_train = train["Target"].values
X_test = test[features].values
y_test = test["Target"].values

# Pipelines (scale where needed)
pipelines = {
    "RandomForest": Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "SVM (RBF)": Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(C=svm_c, kernel="rbf", gamma="scale", probability=True, random_state=42))
    ]),
    "ANN (MLP)": Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(mlp_hidden,),
                              activation="relu",
                              solver="adam",
                              alpha=1e-4,
                              learning_rate_init=1e-3,
                              max_iter=300,
                              random_state=42))
    ])
}

# Time-series CV for in-sample sanity (optional, compact)
tscv = TimeSeriesSplit(n_splits=n_splits)

with st.spinner("Training and evaluating models..."):
    rows = []
    proba_dict = {}
    preds_dict = {}

    for name, pipe in pipelines.items():
        # Fit on all training data (clean approach; CV score for info)
        # Quick CV accuracy
        cv_scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            p = pipe.fit(X_tr, y_tr)
            y_va_hat = p.predict(X_va)
            acc = accuracy_score(y_va, y_va_hat)
            cv_scores.append(acc)

        # Final fit on full train and test
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)
        # Some classifiers expose predict_proba; SVM with probability=True has it
        if hasattr(pipe[-1], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # Fallback to decision_function if available, map to [0,1] via logistic
            if hasattr(pipe[-1], "decision_function"):
                dfunc = pipe.decision_function(X_test)
                y_proba = 1 / (1 + np.exp(-dfunc))
            else:
                # Last resort: 0/1
                y_proba = y_hat.astype(float)

        proba_dict[name] = y_proba
        preds_dict[name] = y_hat

        # Metrics
        acc = accuracy_score(y_test, y_hat)
        prec = precision_score(y_test, y_hat, zero_division=0)
        rec = recall_score(y_test, y_hat, zero_division=0)
        f1 = f1_score(y_test, y_hat, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = np.nan

        rows.append({
            "Model": name,
            "CV Acc (mean)": np.mean(cv_scores),
            "Test Acc": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": auc
        })

    metrics_df = pd.DataFrame(rows).sort_values("Test Acc", ascending=False)

st.markdown("#### Classification metrics (Out-of-sample)")
st.dataframe(
    metrics_df.style.format({
        "CV Acc (mean)":"{:.3f}",
        "Test Acc":"{:.3f}",
        "Precision":"{:.3f}",
        "Recall":"{:.3f}",
        "F1":"{:.3f}",
        "ROC-AUC":"{:.3f}",
    }),
    use_container_width=True
)

# Backtest: convert predictions to positions (+1 for long if proba>0.5) per symbol, aggregate equally
bt = test[["Date","Symbol","LogRet"]].copy()
bt = bt.reset_index(drop=True)

bt_curves = {}
risk_rows = []

for name, proba in proba_dict.items():
    # signal per row aligned with test
    sig = (proba > 0.5).astype(int) * 1.0  # long or flat
    # Position applied to next period's return (avoid lookahead by shifting signal)
    pos = pd.Series(sig).shift(1).fillna(0.0).values

    strat_ret = pos * bt["LogRet"].values
    # Aggregate across symbols by equal weight on each date
    df_tmp = bt.copy()
    df_tmp["StratRet"] = strat_ret
    # daily (or weekly/monthly) aggregation by date: equally-weighted across symbols
    eq = df_tmp.groupby("Date")["StratRet"].mean()
    cum = (1 + eq).cumprod()

    bt_curves[name] = cum

    sh = sharpe_ratio(eq.values)
    mdd = max_drawdown(cum.values)
    risk_rows.append({"Model": name, "Sharpe": sh, "Max Drawdown": mdd})

risk_df = pd.DataFrame(risk_rows).sort_values("Sharpe", ascending=False)

colX, colY = st.columns([2,1])
with colX:
    fig_bt = go.Figure()
    for name, curve in bt_curves.items():
        fig_bt.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=name))
    fig_bt.update_layout(height=380, title="Backtest — Equity Curves (equal-weight, long/flat based on model signal)",
                         yaxis_title="Growth of $1")
    st.plotly_chart(fig_bt, use_container_width=True)

with colY:
    st.markdown("#### Risk‑adjusted performance")
    st.dataframe(
        risk_df.style.format({"Sharpe":"{:.2f}", "Max Drawdown":"{:.1%}"}),
        use_container_width=True
    )

# Confusion matrices
st.markdown("#### Confusion matrices (Out-of-sample)")
cm_cols = st.columns(3)
for (name, yhat), container in zip(preds_dict.items(), cm_cols):
    cm = confusion_matrix(y_test, yhat)
    z = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
    with container:
        fig_cm = px.imshow(z, text_auto=True, aspect="auto", title=name)
        fig_cm.update_layout(height=300, margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

# Feature importance (RF only)
if "RandomForest" in pipelines:
    rf = pipelines["RandomForest"].fit(X_train, y_train)
    try:
        importances = rf[-1].feature_importances_
        fi = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
        st.markdown("#### RF — Feature importance (Gini)")
        st.dataframe(fi, use_container_width=True)
        fi_fig = px.bar(fi.head(12), x="Importance", y="Feature", orientation="h", title="Top features (RF)")
        fi_fig.update_layout(height=400)
        st.plotly_chart(fi_fig, use_container_width=True)
    except Exception:
        st.info("Feature importance not available.")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ---------------------------
# NOTES / ALIGNMENT
# ---------------------------
st.markdown("## 3) Notes & Alignment to Objectives")
st.markdown("""
- **EDA first**: coverage, missingness, distributions, rolling volatility, correlations, and key technicals are surfaced before modeling.  
- **Model comparison**: clean, like‑for‑like setup for **RF vs SVM vs ANN (MLP)** on next‑day direction across selected markets.  
- **Risk & Backtest**: simple, transparent long/flat strategy from predicted direction → equity curves, **Sharpe** & **Max Drawdown**.  
- **No uploads**: Data pulled via Yahoo Finance (`yfinance`).  
- **Emerging vs Developed**: quick toggles for India/Brazil/South Africa vs US/UK (extend the ticker lists as needed).  
- **Method guards**: time‑series split, scaling only within pipelines (prevents leakage), horizon shift, and post‑fit evaluation on strictly out‑of‑sample data.
""")

st.markdown("> Tip: to broaden the study, add more tickers to `UNIVERSES`, or extend features (e.g., macro factors, sector dummies).")
