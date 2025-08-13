# app.py
import os
import io
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ML
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional: ANN (safe toggle below)
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TF_OK = True
except Exception:
    TF_OK = False


# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="ML in Finance — Mid Review", layout="wide")

st.markdown(
    """
    <style>
    .small {font-size:0.9rem;color:#555}
    .ok   {color:#0b8a42;font-weight:600}
    .warn {color:#b36b00;font-weight:600}
    .bad  {color:#c72d2d;font-weight:600}
    </style>
    """,
    unsafe_allow_html=True
)

DATA_FILES = {
    "S&P 500":     {"feat": "sp500_full_features.csv",   "base": "sp500_full_data.csv"},
    "FTSE 100":    {"feat": "ftse100_full_features.csv", "base": "ftse100_full_data.csv"},
    "NIFTY 50":    {"feat": "nifty50_full_features.csv", "base": "nifty50_full_data.csv"},
    "BOVESPA":     {"feat": "bovespa_full_features.csv", "base": "bovespa_full_data.csv"},
}

DEFAULT_FEATURES = [
    # If your *_full_features have these already, great.
    # If not, we'll compute (Return_1D, SMA_20, SMA_50, Volatility_20)
    "Return_1D", "SMA_20", "SMA_50", "Volatility_20"
]

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_market_df(market_key: str) -> pd.DataFrame:
    """Load features first; else base data. Ensure canonical columns & features."""
    files = DATA_FILES[market_key]
    df = None
    for f in [files["feat"], files["base"]]:
        if os.path.exists(f):
            df = pd.read_csv(f)
            break
    if df is None:
        return pd.DataFrame()  # not found

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Flexible date col
    date_col = None
    for c in ["Date", "date", "DATE", "Price_Ticker"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # Try to infer if first col looks like date
        if df.columns[0].lower().startswith("date"):
            date_col = df.columns[0]
        else:
            # give up
            return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].sort_values(by=date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "Date"})

    # Basic required columns
    must_have = ["Date", "Ticker", "Close"]
    for need in must_have:
        if need not in df.columns:
            # try fallback names
            if need == "Ticker":
                # sometimes uppercase
                if "TICKER" in df.columns:
                    df = df.rename(columns={"TICKER": "Ticker"})
            if need == "Close":
                for alt in ["Adj Close", "Close_AAPL", "close"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "Close"})
                        break

    # Drop junk rows
    df = df[df["Close"].astype(str).str.isnumeric() | df["Close"].apply(lambda x: isinstance(x, (int, float)))]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df[df["Close"].notna()]

    # Compute missing minimal features by ticker
    if "Return_1D" not in df.columns:
        df["Return_1D"] = df.groupby("Ticker")["Close"].pct_change()
    if "SMA_20" not in df.columns:
        df["SMA_20"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    if "SMA_50" not in df.columns:
        df["SMA_50"] = df.groupby("Ticker")["Close"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    if "Volatility_20" not in df.columns:
        df["Volatility_20"] = df.groupby("Ticker")["Return_1D"].transform(lambda s: s.rolling(20, min_periods=5).std())

    # Ensure Target (next-day up/down) exists
    if "Target" not in df.columns:
        df["Future_Return_1D"] = df.groupby("Ticker")["Close"].pct_change(-1) * -1  # next day relative to today
        # Simpler: shift(-1) on Close then pct_change from today -> tomorrow:
        df["Future_Close"] = df.groupby("Ticker")["Close"].shift(-1)
        df["Future_Ret"] = (df["Future_Close"] / df["Close"]) - 1.0
        df["Target"] = (df["Future_Ret"] > 0).astype(int)

    # Clean residual NaNs
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def make_date_slider_values(df: pd.DataFrame, market: str):
    if df.empty:
        return None, None
    return pd.to_datetime(df["Date"].min()).date(), pd.to_datetime(df["Date"].max()).date()


def filter_df(df, tickers, start_date, end_date):
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    if tickers:
        df = df[df["Ticker"].isin(tickers)]
    return df[mask].copy()


def plot_price_and_sma(df):
    fig = go.Figure()
    for tkr in sorted(df["Ticker"].unique()):
        sub = df[df["Ticker"] == tkr]
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Close"], name=f"{tkr} Close", mode="lines"))
        if "SMA_20" in sub.columns:
            fig.add_trace(go.Scatter(x=sub["Date"], y=sub["SMA_20"], name=f"{tkr} SMA 20", mode="lines"))
    fig.update_layout(title="Close vs 20-Day SMA", legend_title=None, height=450)
    return fig


def plot_return_hist(df):
    sub = df.copy()
    sub["Return_1D"] = pd.to_numeric(sub["Return_1D"], errors="coerce")
    sub = sub[np.isfinite(sub["Return_1D"])]
    if sub.empty:
        return go.Figure()
    fig = px.histogram(sub, x="Return_1D", nbins=50, title="Daily Return Distribution", marginal="box")
    return fig


def plot_volatility(df):
    if "Volatility_20" not in df.columns:
        return go.Figure()
    fig = px.line(df, x="Date", y="Volatility_20", color="Ticker", title="20‑Day Rolling Volatility")
    return fig


def compute_summary_table(df):
    out = []
    for tkr, g in df.groupby("Ticker"):
        ret = g["Return_1D"].dropna()
        if ret.empty:
            continue
        ann_ret = (1 + ret.mean())**252 - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan
        out.append([tkr, ann_ret, ann_vol, sharpe, ret.skew(), ret.kurtosis()])
    if not out:
        return pd.DataFrame()
    tbl = pd.DataFrame(out, columns=["Ticker", "Ann. Return", "Ann. Vol", "Sharpe", "Skew", "Kurtosis"])
    return tbl.sort_values("Sharpe", ascending=False)


def train_models_classification(df, features, c_val=1.0, n_trees=200, use_ann=False):
    """
    TimeSeriesSplit on concatenated panel (we standardize per split).
    Predicts next-day up/down (Target).
    """
    # Clean
    work = df.copy()
    work = work.dropna(subset=["Target"])
    for f in features:
        if f not in work.columns:
            work[f] = np.nan
    work = work.dropna(subset=features)

    if work.empty:
        return None, pd.DataFrame()

    X = work[features].values
    y = work["Target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    # Models
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    svm = SVC(C=c_val, kernel="rbf", probability=True, random_state=42)

    # Optional ANN
    def build_ann(input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # RF
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict(X_te)
        pr_rf = rf.predict_proba(X_te)[:, 1] if hasattr(rf, "predict_proba") else (p_rf * 1.0)
        rows.append(["RandomForest", fold,
                    accuracy_score(y_te, p_rf),
                    precision_score(y_te, p_rf, zero_division=0),
                    recall_score(y_te, p_rf, zero_division=0),
                    f1_score(y_te, p_rf, zero_division=0),
                    safe_auc(y_te, pr_rf)])

        # SVM
        svm.fit(X_tr_s, y_tr)
        p_svm = svm.predict(X_te_s)
        pr_svm = svm.predict_proba(X_te_s)[:, 1] if hasattr(svm, "predict_proba") else (p_svm * 1.0)
        rows.append(["SVM (RBF)", fold,
                    accuracy_score(y_te, p_svm),
                    precision_score(y_te, p_svm, zero_division=0),
                    recall_score(y_te, p_svm, zero_division=0),
                    f1_score(y_te, p_svm, zero_division=0),
                    safe_auc(y_te, pr_svm)])

        # ANN (optional)
        if use_ann and TF_OK:
            ann = build_ann(X_tr_s.shape[1])
            ann.fit(X_tr_s, y_tr, epochs=10, batch_size=256, verbose=0, validation_split=0.1)
            pr_ann = ann.predict(X_te_s, verbose=0).reshape(-1)
            p_ann = (pr_ann >= 0.5).astype(int)
            rows.append(["ANN (MLP)", fold,
                        accuracy_score(y_te, p_ann),
                        precision_score(y_te, p_ann, zero_division=0),
                        recall_score(y_te, p_ann, zero_division=0),
                        f1_score(y_te, p_ann, zero_division=0),
                        safe_auc(y_te, pr_ann)])

    res = pd.DataFrame(rows, columns=["Model", "Fold", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"])
    return {"rf": rf, "svm": svm}, res


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


# ---------------------------
# UI
# ---------------------------
st.title("Machine Learning Applications in Finance — Dashboard")

tab_over, tab_data, tab_eda, tab_models, tab_auto = st.tabs(
    ["Overview", "Data Explorer", "EDA", "Models", "Auto‑Results"]
)

# -------- Overview --------
with tab_over:
    st.subheader("Project Overview")
    st.markdown(
        """
        **Goal:** Benchmark common ML models (RF, SVM, ANN) for **next‑day direction** across **developed vs. emerging markets** and connect results to investment‑relevant metrics (accuracy, F1, ROC‑AUC).

        **What’s here:**
        - _Data Explorer_: browse and filter by market/ticker/date.
        - _EDA_: return distribution, price vs. SMA, rolling volatility, summary stats.
        - _Models_: train RF/SVM/ANN quickly on selected universe + features.
        - _Auto‑Results_: one‑click training + export a CSV for your report.

        **Research linkage:** aligns to objectives on (i) real‑world impact, (ii) model comparability, and (iii) developed vs. emerging contrasts.
        """
    )
    st.caption("Tip: If a market file is missing, that tab will gracefully show an empty state.")

# -------- Data Explorer --------
with tab_data:
    st.subheader("Data Explorer")
    market = st.selectbox("Market", list(DATA_FILES.keys()))
    df_market = load_market_df(market)
    if df_market.empty:
        st.warning(f"No data found for **{market}**. Ensure CSVs exist in repo root.")
    else:
        min_d, max_d = make_date_slider_values(df_market, market)
        # Ensure proper datetime.date types for slider
        min_d, max_d = pd.to_datetime(min_d).date(), pd.to_datetime(max_d).date()
        tickers = sorted(df_market["Ticker"].dropna().unique().tolist())
        sel_tickers = st.multiselect("Tickers", tickers, default=tickers[:3])

        date_range = st.slider(
            "Select Date Range",
            min_value=min_d,
            max_value=max_d,
            value=(min_d, max_d)
        )

        view = filter_df(df_market, sel_tickers, date_range[0], date_range[1])
        st.markdown(f"**Rows:** {len(view):,} | **Tickers:** {len(sel_tickers)} | **Date:** {date_range[0]} → {date_range[1]}")
        st.dataframe(view.head(500), use_container_width=True)

        with st.expander("Columns detected"):
            st.write(list(view.columns))

# -------- EDA --------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    market_e = st.selectbox("Market (EDA)", list(DATA_FILES.keys()), key="eda_market")
    df_e = load_market_df(market_e)
    if df_e.empty:
        st.warning(f"No data for **{market_e}**.")
    else:
        min_d, max_d = make_date_slider_values(df_e, market_e)
        min_d, max_d = pd.to_datetime(min_d).date(), pd.to_datetime(max_d).date()
        tickers_e = sorted(df_e["Ticker"].dropna().unique().tolist())
        sel_tickers_e = st.multiselect("Tickers", tickers_e, default=tickers_e[:2], key="eda_tickers")

        date_range_e = st.slider(
            "Date Range",
            min_value=min_d,
            max_value=max_d,
            value=(min_d, max_d),
            key="eda_dates"
        )
        view_e = filter_df(df_e, sel_tickers_e, date_range_e[0], date_range_e[1])

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_price_and_sma(view_e), use_container_width=True)
            st.caption("**20‑Day SMA vs Close** shows short‑term trend vs. price.")
        with c2:
            st.plotly_chart(plot_return_hist(view_e), use_container_width=True)
            st.caption("**Daily Return Distribution**: location/spread/skew/kurtosis → volatility/risk clues.")

        st.plotly_chart(plot_volatility(view_e), use_container_width=True)
        st.caption("**Rolling Volatility (20‑day)** approximates short‑run risk. Spikes often line up with macro/news shocks.")

        st.markdown("### Summary by Ticker (selected range)")
        table = compute_summary_table(view_e)
        if table.empty:
            st.info("Not enough return data to compute summary.")
        else:
            st.dataframe(table, use_container_width=True, height=300)

# -------- Models --------
with tab_models:
    st.subheader("Quick Model Benchmarks (Classification: next‑day up/down)")
    market_m = st.selectbox("Market (Models)", list(DATA_FILES.keys()), key="mdl_market")
    df_m = load_market_df(market_m)
    if df_m.empty:
        st.warning(f"No data for **{market_m}**.")
    else:
        # Choose tickers
        all_t = sorted(df_m["Ticker"].dropna().unique().tolist())
        sel_t = st.multiselect("Tickers (train on these)", all_t, default=all_t[:5], key="mdl_tickers")

        # Feature selection (only keep what's available)
        avail_feats = [f for f in DEFAULT_FEATURES if f in df_m.columns]
        if not avail_feats:
            st.error("No default features found; falling back to simple features computed from Close.")
            avail_feats = ["Return_1D", "SMA_20", "SMA_50", "Volatility_20"]

        features = st.multiselect("Features", sorted(list(set(avail_feats + DEFAULT_FEATURES))), default=avail_feats)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            C_val = st.number_input("SVM C", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        with col_b:
            n_trees = st.number_input("RF n_estimators", min_value=50, max_value=1000, value=200, step=50)
        with col_c:
            use_ann = st.checkbox("Include ANN (requires TensorFlow)", value=False and TF_OK)
            if use_ann and not TF_OK:
                st.warning("TensorFlow not available in this environment. ANN will be skipped.")

        # Filter to tickers (all dates)
        if sel_t:
            work = df_m[df_m["Ticker"].isin(sel_t)].copy()
        else:
            work = df_m.copy()

        # Train
        if st.button("Train & Evaluate"):
            models, res = train_models_classification(work, features, c_val=C_val, n_trees=n_trees, use_ann=use_ann and TF_OK)
            if res.empty:
                st.error("No rows after cleaning. Check features/Target availability.")
            else:
                st.success("Training complete.")
                st.dataframe(res, use_container_width=True)
                st.markdown("**Fold averages**")
                st.dataframe(res.groupby("Model", as_index=False).mean(numeric_only=True), use_container_width=True)

# -------- Auto‑Results (for report) --------
with tab_auto:
    st.subheader("Auto‑Generate Results CSV")
    st.caption("Runs RF & SVM across each market on default features (panel data, next‑day up/down). Saves a consolidated CSV for your report.")

    target_markets = st.multiselect(
        "Select markets to run",
        list(DATA_FILES.keys()),
        default=list(DATA_FILES.keys())
    )
    include_ann = st.checkbox("Include ANN (TensorFlow required)", value=False and TF_OK)
    run_btn = st.button("Run All & Build CSV")

    if run_btn:
        rows = []
        for m in target_markets:
            dfx = load_market_df(m)
            if dfx.empty:
                continue
            feats = [f for f in DEFAULT_FEATURES if f in dfx.columns]
            if not feats:
                feats = ["Return_1D", "SMA_20", "SMA_50", "Volatility_20"]

            # Use top tickers by count
            top_t = (
                dfx.groupby("Ticker")["Date"]
                .count()
                .sort_values(ascending=False)
                .head(30)
                .index.tolist()
            )
            work = dfx[dfx["Ticker"].isin(top_t)].copy()

            _, res = train_models_classification(work, feats, c_val=1.0, n_trees=250, use_ann=include_ann and TF_OK)
            if not res.empty:
                res["Market"] = m
                res["Features"] = ",".join(feats)
                rows.append(res)

        if not rows:
            st.error("No results produced (check data files).")
        else:
            out = pd.concat(rows, ignore_index=True)
            st.dataframe(out, use_container_width=True)

            # Save to buffer for download
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button(
                "Download results.csv",
                data=buf.getvalue(),
                file_name="model_results.csv",
                mime="text/csv"
            )
            st.success("Results ready. Use this file in your ‘ML model results’ section.")


# ---------------------------
# Footer
# ---------------------------
st.markdown(
    """
    <hr class='small' />
    <div class='small'>
      <b>Notes</b>:
      1) Target = 1 if next‑day close &gt; today’s close (panel classification).<br>
      2) Default features: Return_1D, SMA_20, SMA_50, Volatility_20 (auto‑computed if missing).<br>
      3) Models use TimeSeriesSplit (5 folds). SVM uses scaled features. RF uses raw features.<br>
      4) ANN is optional; enable only if TensorFlow is in requirements.
    </div>
    """,
    unsafe_allow_html=True
)
