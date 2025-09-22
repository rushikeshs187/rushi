# app.py — IBR Finance: Sectioned EDA → Features → Modeling → Backtest → Insights
# Clean, research-first UX with tabs, expanders, and downloads.
# (Includes a small safety net that installs missing libs when run locally.)

# ---------------------------
# BOOTSTRAP MISSING PACKAGES
# ---------------------------
def _ensure(pkgs):
    import importlib, sys, subprocess
    for name, pip_name in pkgs:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

_ensure([
    ("streamlit", "streamlit>=1.38"),
    ("pandas", "pandas>=1.5"),
    ("numpy", "numpy>=1.23"),
    ("yfinance", "yfinance>=0.2.40"),
    ("plotly", "plotly>=5.24"),
    ("sklearn", "scikit-learn>=1.3"),
])

# ---------------------------
# IMPORTS
# ---------------------------
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
# PAGE CONFIG & THEME POLISH
# ---------------------------
st.set_page_config(
    page_title="IBR Finance — Markets EDA & ML Model Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root{
  --card-bg:#161a22;         /* dark card */
  --card-border:#2a2f3a;     /* subtle border */
  --card-text:#e8eaed;       /* main text */
  --muted:#9aa0a6;           /* labels */
  --radius:14px;
}
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
hr.hr { border:0; height:1px; background:#2a2f3a; margin:.6rem 0 1rem; }

.kpi-card{
  background:var(--card-bg);
  border:1px solid var(--card-border);
  border-radius:var(--radius);
  padding:1rem 1.2rem;
}
.kpi-card .label{ color:var(--muted); font-size:.9rem; letter-spacing:.01em; }
.kpi-card h3{ color:var(--card-text); margin:.25rem 0 0; font-weight:700; }

.caption, .small { color: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# UNIVERSES & PRESETS
# ---------------------------
UNIVERSES = {
    "US (Developed)": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"],
    "UK (Developed)": ["HSBA.L", "AZN.L", "BP.L", "ULVR.L", "GSK.L", "RIO.L"],
    "India (Emerging)": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    "Brazil (Emerging)": ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "B3SA3.SA"],
    "South Africa (Emerging)": ["NPN.JO", "AGL.JO", "BHG.JO", "SOL.JO"],
    "Crypto (Global)": ["BTC-USD", "ETH-USD"],
}
INTERVALS = ["1d", "1wk", "1mo"]
PERIODS = ["1y", "2y", "5y", "10y", "max"]

PRESETS = {
    "Developed study": ["US (Developed)", ["AAPL","MSFT","NVDA","AMZN","GOOGL","META"]],
    "Emerging study": ["India (Emerging)", ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"]],
}

# ---------------------------
# UTILITIES
# ---------------------------
def compute_indicators(df):
    """Add common technical indicators by symbol."""
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    out = []
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").copy()
        pxv = g["Adj Close"].astype(float)
        g["Return"] = pxv.pct_change()
        g["LogRet"] = np.log(pxv).diff()
        g["SMA_20"] = pxv.rolling(20).mean()
        g["SMA_50"] = pxv.rolling(50).mean()
        g["EMA_12"] = pxv.ewm(span=12, adjust=False).mean()
        g["EMA_26"] = pxv.ewm(span=26, adjust=False).mean()
        g["MACD"] = g["EMA_12"] - g["EMA_26"]
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["BB_M"] = pxv.rolling(20).mean()
        bb_std = pxv.rolling(20).std()
        g["BB_U"] = g["BB_M"] + 2 * bb_std
        g["BB_L"] = g["BB_M"] - 2 * bb_std
        g["RSI_14"] = rsi(pxv, 14)
        g["Volatility_20"] = g["Return"].rolling(20).std() * np.sqrt(252)
        out.append(g)
    return pd.concat(out, axis=0) if out else pd.DataFrame()

def prepare_ml_frame(df, horizon=1):
    """Supervised frame for next-day direction. Target=1 if next LogRet>0."""
    feats = [
        "Return","LogRet","SMA_20","SMA_50","EMA_12","EMA_26",
        "MACD","MACD_Signal","RSI_14","BB_M","BB_U","BB_L","Volatility_20",
        "LagRet_1","LagRet_2","LagRet_5",
    ]
    if df.empty: return pd.DataFrame(), feats
    frames = []
    for _, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date").copy()
        g["LagRet_1"] = g["LogRet"].shift(1)
        g["LagRet_2"] = g["LogRet"].shift(2)
        g["LagRet_5"] = g["LogRet"].shift(5)
        g["Target"] = (g["LogRet"].shift(-horizon) > 0).astype(int)
        frames.append(g)
    X = pd.concat(frames, axis=0) if frames else pd.DataFrame()
    if X.empty: return X, feats
    X = X.dropna(subset=feats + ["Target"]).copy()
    return X, feats

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    rets = pd.Series(returns).dropna()
    if rets.empty: return np.nan
    excess = rets - (risk_free / periods_per_year)
    mu = excess.mean() * periods_per_year
    sigma = excess.std(ddof=0) * np.sqrt(periods_per_year)
    if sigma == 0 or np.isnan(sigma): return np.nan
    return mu / sigma

def max_drawdown(cum_curve):
    s = pd.Series(cum_curve).replace([np.inf, -np.inf], np.nan).ffill()
    if s.empty: return np.nan
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    return float(dd.min())

@st.cache_data(show_spinner=True)
def fetch_prices(symbols, period="5y", interval="1d"):
    """Tidy long frame: ['Symbol','Date','Open','High','Low','Close','Adj Close','Volume']"""
    if isinstance(symbols, str):
        symbols = [symbols]
    data = yf.download(
        tickers=" ".join(symbols),
        period=period, interval=interval,
        auto_adjust=False, progress=False, threads=True, group_by="ticker"
    )
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["Symbol","Date","Open","High","Low","Close","Adj Close","Volume"])
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in data.columns.get_level_values(0):
                g = data[sym].reset_index()
                if "Date" not in g.columns: g.rename(columns={"index":"Date"}, inplace=True)
                g.insert(0, "Symbol", sym)
                frames.append(g)
    else:
        g = data.reset_index()
        if "Date" not in g.columns: g.rename(columns={"index":"Date"}, inplace=True)
        g.insert(0, "Symbol", symbols[0])
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["Symbol","Date","Open","High","Low","Close","Adj Close","Volume"])
    df = pd.concat(frames, axis=0, ignore_index=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col not in df.columns: df[col] = np.nan
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Symbol","Date"])
    df = df[~df[["Date","Symbol"]].duplicated(keep="last")]
    return df.reset_index(drop=True)

def _download_csv_button(df, label, filename):
    if df is None or df.empty: return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.title("Study Controls")

preset = st.sidebar.selectbox("Preset", ["— None —"] + list(PRESETS.keys()), index=0)
if preset != "— None —":
    p_market, p_symbols = PRESETS[preset]
else:
    p_market, p_symbols = list(UNIVERSES.keys())[0], UNIVERSES[list(UNIVERSES.keys())[0]]

market = st.sidebar.selectbox("Market", list(UNIVERSES.keys()), index=list(UNIVERSES.keys()).index(p_market))
symbols_default = UNIVERSES[market]
symbols = st.sidebar.multiselect("Symbols", options=symbols_default,
                                 default=(p_symbols if preset != "— None —" else symbols_default))
period = st.sidebar.selectbox("History Window", PERIODS, index=2)   # 5y
interval = st.sidebar.selectbox("Sampling Interval", INTERVALS, index=0)  # 1d
eda_symbol = st.sidebar.selectbox("Primary symbol (deep-dive)", options=symbols if symbols else symbols_default, index=0)

with st.sidebar.expander("Advanced Modeling", expanded=False):
    horizon = st.slider("Prediction horizon (days ahead)", 1, 5, 1, 1)
    test_size_years = st.slider("Test span (years)", 1, 5, 2, 1)
    n_splits = st.slider("CV splits (rolling)", 2, 5, 3, 1)
    rf_trees = st.slider("RF: n_estimators", 50, 400, 200, 50)
    svm_c = st.selectbox("SVM: C", [0.1, 0.5, 1.0, 2.0, 5.0], index=2)
    mlp_hidden = st.selectbox("ANN (MLP): hidden units", [32, 64, 128], index=1)

# ---------------------------
# HEADER / HERO
# ---------------------------
st.subheader("IBR Finance — Markets EDA & ML Model Comparison")
st.caption("Section-first dashboard aligned to research objectives: **Exploration → Features → Models → Backtest → Insights**")

# ---------------------------
# DATA LOAD
# ---------------------------
if not symbols:
    st.warning("Pick at least one symbol to proceed.")
    st.stop()

with st.spinner("Fetching market data..."):
    raw = fetch_prices(symbols, period=period, interval=interval)

if raw.empty:
    st.error("No data returned for selected inputs. Try another market/period/interval.")
    st.stop()

raw = raw.sort_values(["Symbol","Date"]).reset_index(drop=True)

# Pre-compute enriched & (default horizon=1 for EDA/Features visuals)
enriched = compute_indicators(raw) if not raw.empty else pd.DataFrame()

# ---------------------------
# TABS: Overview | EDA | Features | Modeling | Backtest & Risk | Insights
# ---------------------------
tab_overview, tab_eda, tab_feat, tab_model, tab_bt, tab_insights = st.tabs(
    ["Overview", "EDA", "Features", "Modeling", "Backtest & Risk", "Insights"]
)

# ============ OVERVIEW ============
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='label'>Symbols</div><h3>{raw['Symbol'].nunique()}</h3></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='label'>Observations</div><h3>{len(raw):,}</h3></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='label'>Date Range</div><h3>{raw['Date'].min().date()} → {raw['Date'].max().date()}</h3></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><div class='label'>Interval</div><h3>{interval}</h3></div>", unsafe_allow_html=True)

    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

    # Primary price with simple overlays
    sym_df = raw[raw["Symbol"] == eda_symbol].dropna(subset=["Adj Close"]).copy()
    sym_df["SMA_20"] = sym_df["Adj Close"].rolling(20).mean()
    sym_df["SMA_50"] = sym_df["Adj Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sym_df["Date"], y=sym_df["Adj Close"], mode="lines", name=f"{eda_symbol} Adj Close"))
    fig.add_trace(go.Scatter(x=sym_df["Date"], y=sym_df["SMA_20"], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=sym_df["Date"], y=sym_df["SMA_50"], mode="lines", name="SMA 50"))
    fig.update_layout(height=380, title=f"{eda_symbol} — Price & SMAs", xaxis_title=None, yaxis_title="Adj Close")
    st.plotly_chart(fig, use_container_width=True)

    colL, colR = st.columns([3,1])
    with colL:
        st.caption("Download raw dataset")
        _download_csv_button(raw, "⬇️ Download prices (CSV)", f"prices_{market.replace(' ','_')}.csv")
    with colR:
        st.write("")

# ============ EDA ============
with tab_eda:
    st.markdown("### Distribution & Volatility")
    sym_df = raw[raw["Symbol"] == eda_symbol].dropna(subset=["Adj Close"]).copy()
    sym_df["Return"] = sym_df["Adj Close"].pct_change()
    sym_df["LogRet"] = np.log(sym_df["Adj Close"]).diff()

    col1, col2 = st.columns(2)
    with col1:
        hist = px.histogram(sym_df, x="Return", nbins=60, title=f"{eda_symbol} — Daily Return Distribution")
        hist.update_layout(height=320)
        st.plotly_chart(hist, use_container_width=True)
    with col2:
        roll = sym_df.set_index("Date")["Return"].rolling(20).std() * np.sqrt(252)
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Ann. Vol (20d)"))
        rfig.update_layout(height=320, title=f"{eda_symbol} — Rolling Annualized Volatility (20d)")
        st.plotly_chart(rfig, use_container_width=True)

    with st.expander("Summary stats (adjusted close & returns)"):
        desc = sym_df[["Adj Close","Return","LogRet","Volume"]].describe().T
        st.dataframe(desc, use_container_width=True)

# ============ FEATURES ============
with tab_feat:
    st.markdown("### Technical Indicators & Correlations")
    if enriched.empty:
        st.info("Insufficient data to compute indicators.")
    else:
        latest = (
            enriched.sort_values("Date")
            .groupby("Symbol", as_index=False)
            .apply(lambda g: g.iloc[-1][["Symbol","Date","Adj Close","RSI_14","Volatility_20"]])
            .reset_index(drop=True)
        )
        st.markdown("**Latest snapshot**")
        st.dataframe(latest.style.format({"Adj Close":"{:,.2f}", "RSI_14":"{:,.1f}", "Volatility_20":"{:,.2f}"}),
                     use_container_width=True)

        corr_cols = ["Return","LogRet","SMA_20","SMA_50","EMA_12","EMA_26","MACD","MACD_Signal","RSI_14","Volatility_20"]
        corr_df = enriched[enriched["Symbol"] == eda_symbol][corr_cols].dropna()
        if len(corr_df) > 30:
            corr = corr_df.corr(numeric_only=True)
            cfig = px.imshow(corr, text_auto=True, aspect="auto", title=f"{eda_symbol} — Feature Correlation")
            cfig.update_layout(height=520)
            st.plotly_chart(cfig, use_container_width=True)
        else:
            st.info("Not enough rows yet for a stable correlation heatmap (need >30).")

        _download_csv_button(enriched, "⬇️ Download enriched (CSV)", "enriched_features.csv")

# ============ MODELING ============
with tab_model:
    st.markdown("### Next-day Direction — RF vs SVM vs ANN (MLP)")
    ml_df, features = prepare_ml_frame(enriched, horizon=horizon) if not enriched.empty else (pd.DataFrame(), [])
    if ml_df.empty:
        st.warning("Not enough data after feature engineering. Try a longer period or daily interval.")
        st.stop()

    max_date = ml_df["Date"].max()
    cutoff = max_date - pd.DateOffset(years=test_size_years)
    train = ml_df[ml_df["Date"] < cutoff].copy()
    test = ml_df[ml_df["Date"] >= cutoff].copy()
    if train["Target"].nunique() < 2 or test["Target"].nunique() < 2:
        st.warning("Target class imbalance or insufficient variety in train/test. Increase the history window.")
        st.stop()

    X_train = train[features].values; y_train = train["Target"].values
    X_test  = test[features].values;  y_test  = test["Target"].values

    pipelines = {
        "RandomForest": Pipeline([("rf", RandomForestClassifier(
            n_estimators=rf_trees, max_depth=None, random_state=42, n_jobs=-1))]),
        "SVM (RBF)": Pipeline([("sc", StandardScaler()), ("svm", SVC(C=svm_c, kernel="rbf",
            gamma="scale", probability=True, random_state=42))]),
        "ANN (MLP)": Pipeline([("sc", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(mlp_hidden,), activation="relu",
                                  solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                                  max_iter=300, random_state=42))]),
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)

    with st.spinner("Training models..."):
        rows, proba_dict, preds_dict = [], {}, {}
        for name, pipe in pipelines.items():
            # quick in-sample CV for sanity
            cv_scores = []
            for tr_idx, va_idx in tscv.split(X_train):
                p = pipe.fit(X_train[tr_idx], y_train[tr_idx])
                cv_scores.append(accuracy_score(y_train[va_idx], p.predict(X_train[va_idx])))

            pipe.fit(X_train, y_train)
            y_hat = pipe.predict(X_test)
            if hasattr(pipe[-1], "predict_proba"):
                y_proba = pipe.predict_proba(X_test)[:, 1]
            elif hasattr(pipe[-1], "decision_function"):
                dfunc = pipe.decision_function(X_test); y_proba = 1/(1+np.exp(-dfunc))
            else:
                y_proba = y_hat.astype(float)

            proba_dict[name] = y_proba
            preds_dict[name] = y_hat

            acc = accuracy_score(y_test, y_hat)
            prec = precision_score(y_test, y_hat, zero_division=0)
            rec = recall_score(y_test, y_hat, zero_division=0)
            f1 = f1_score(y_test, y_hat, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = np.nan

            rows.append({"Model": name, "CV Acc (mean)": float(np.mean(cv_scores)),
                         "Test Acc": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc})

        metrics_df = pd.DataFrame(rows).sort_values("Test Acc", ascending=False)

    st.markdown("**Out-of-sample metrics**")
    st.dataframe(metrics_df.style.format({
        "CV Acc (mean)":"{:.3f}","Test Acc":"{:.3f}","Precision":"{:.3f}",
        "Recall":"{:.3f}","F1":"{:.3f}","ROC-AUC":"{:.3f}"}), use_container_width=True)
    _download_csv_button(metrics_df, "⬇️ Download metrics (CSV)", "model_metrics.csv")

    st.markdown("**Confusion matrices**")
    cm_cols = st.columns(3)
    for (name, yhat), container in zip(list(preds_dict.items())[:3], cm_cols):
        cm = confusion_matrix(y_test, yhat)
        z = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
        with container:
            fig_cm = px.imshow(z, text_auto=True, aspect="auto", title=name)
            fig_cm.update_layout(height=300, margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

    # Stash for backtest tab
    st.session_state["test_df"] = test[["Date","Symbol","LogRet"]].reset_index(drop=True)
    st.session_state["probas"] = proba_dict

# ============ BACKTEST & RISK ============
with tab_bt:
    st.markdown("### Simple long/flat backtest & risk")
    bt = st.session_state.get("test_df")
    probas = st.session_state.get("probas")

    if bt is None or probas is None:
        st.info("Run the Modeling tab first.")
    else:
        curves, risk_rows = {}, []
        for name, proba in probas.items():
            if len(proba) != len(bt):  # very defensive alignment fallback
                proba = np.resize(proba, len(bt))
            sig = (proba > 0.5).astype(int).astype(float)
            pos = pd.Series(sig).shift(1).fillna(0.0).values  # avoid lookahead
            df_tmp = bt.copy()
            df_tmp["StratRet"] = pos * df_tmp["LogRet"].values
            eq = df_tmp.groupby("Date")["StratRet"].mean()
            cum = (1 + eq).cumprod()
            curves[name] = cum
            risk_rows.append({"Model": name, "Sharpe": sharpe_ratio(eq.values), "Max Drawdown": max_drawdown(cum.values)})

        fig_bt = go.Figure()
        for name, curve in curves.items():
            fig_bt.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="lines", name=name))
        fig_bt.update_layout(height=380, title="Equity Curves (equal-weight, long/flat by model signal)",
                             yaxis_title="Growth of $1")
        st.plotly_chart(fig_bt, use_container_width=True)

        risk_df = pd.DataFrame(risk_rows).sort_values("Sharpe", ascending=False)
        st.dataframe(risk_df.style.format({"Sharpe":"{:.2f}", "Max Drawdown":"{:.1%}"}), use_container_width=True)

        # Downloads
        curves_df = pd.concat([v.rename(k) for k, v in curves.items()], axis=1).reset_index().rename(columns={"index":"Date"})
        _download_csv_button(curves_df, "⬇️ Download equity curves (CSV)", "equity_curves.csv")
        _download_csv_button(risk_df, "⬇️ Download risk table (CSV)", "risk_metrics.csv")

# ============ INSIGHTS ============
with tab_insights:
    st.markdown("### Alignment to IBR Objectives (Concise)")
    st.markdown("""
- **Methodology flow**: EDA → Feature engineering → Modeling → Backtest mirrors a research workflow.
- **Comparability**: RF, SVM, and ANN are trained on identical features/targets with rolling CV.
- **Interpretability**: feature correlations + RF-style importances (add expander if needed) to justify signals.
- **Performance**: backtest translates classification into **risk-adjusted PnL** (Sharpe, Max Drawdown).
- **Scope**: toggles for Developed vs Emerging universes; extend the `UNIVERSES` dict to scale the study.
""")

    st.markdown("### What to present")
    st.markdown("""
1) **EDA**: volatility regimes & distribution shape.  
2) **Features**: which indicators correlate with next-day moves.  
3) **Model results**: OOS metrics (accuracy, F1, AUC) + stability across CV.  
4) **Backtest**: growth of $1 & drawdowns; emphasize robustness.  
5) **Limitations**: Yahoo sampling, class imbalance, no trading costs.  
6) **Next steps**: macro factors, sector dummies, thresholds, cost/slippage, hyper-tuning.
""")
