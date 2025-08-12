# app.py
# Streamlit ML-in-Finance Dashboard (IBR)
# Author: Rushikesh N. Shinde (MGB) — all-in-one app for data, models, and backtest

import io
import sys
import time
import math
import numpy as np
import pandas as pd
from datetime import timedelta

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# =============== Page Config ===============
st.set_page_config(
    page_title="ML in Finance — Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============== Utils ===============
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df

def ensure_datetime_index(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        df = df.set_index(date_col)
    else:
        # If no date col, try index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("No 'Date' column and index is not DatetimeIndex.")
        df = df.sort_index()
    return df

def to_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().fillna(0.0)

def rolling_max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity/roll_max - 1.0
    return dd.min()

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    # rf is annual risk-free rate; convert to per-period
    if returns.std() == 0:
        return 0.0
    mean = returns.mean()
    std = returns.std()
    sr = (mean - rf/periods_per_year) / std * np.sqrt(periods_per_year)
    return float(sr)

def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0:
        return np.inf
    mean = returns.mean()
    dd = downside.std()
    return float((mean - rf/periods_per_year) / dd * np.sqrt(periods_per_year))

def safe_clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1-1e-6)

def make_demo_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    # Geometric random walk for price
    rets = rng.normal(loc=0.0004, scale=0.012, size=n)
    price = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
    vol = (rng.normal(1_000_000, 150_000, size=n)).clip(50_000, None)
    df = pd.DataFrame({"Date": dates, "Close": price.values, "Volume": vol})
    return df

def add_technical_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    px = out[price_col].astype(float)

    # Returns and lagged returns
    out["ret_1"] = px.pct_change()
    out["ret_5"] = px.pct_change(5)
    out["ret_10"] = px.pct_change(10)

    # Moving averages
    out["sma_5"] = px.rolling(5).mean()
    out["sma_10"] = px.rolling(10).mean()
    out["sma_20"] = px.rolling(20).mean()
    out["ema_10"] = px.ewm(span=10, adjust=False).mean()
    out["ema_20"] = px.ewm(span=20, adjust=False).mean()

    # Momentum / Oscillators
    out["mom_10"] = px.pct_change(10)
    # RSI(14)
    delta = px.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Volatility
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # Price position vs bands (simple %B style)
    ma20 = out["sma_20"]
    sd20 = px.rolling(20).std()
    upper = ma20 + 2*sd20
    lower = ma20 - 2*sd20
    out["pctB"] = (px - lower) / (upper - lower + 1e-9)

    # Volume features (if available)
    if "Volume" in out.columns:
        vol = out["Volume"].astype(float)
        out["vol_chg_1"] = vol.pct_change()
        out["vol_sma_10"] = vol.rolling(10).mean()

    return out

def create_target(df: pd.DataFrame, price_col: str = "Close", horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """
    Binary target: 1 if forward % return over 'horizon' days > threshold, else 0.
    """
    fwd_ret = df[price_col].shift(-horizon) / df[price_col] - 1.0
    target = (fwd_ret > threshold).astype(int)
    return target

def timeseries_train_val_test_split(df, train_size=0.6, val_size=0.2):
    n = len(df)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    return train, val, test

def prepare_Xy(df: pd.DataFrame, feature_cols, target_col: str):
    work = df.copy()
    work = work.dropna(subset=feature_cols + [target_col])
    X = work[feature_cols].values
    y = work[target_col].values.astype(int)
    idx = work.index
    return X, y, idx

def backtest_equity_curve(prices: pd.Series, signal: pd.Series, threshold: float = 0.5, prob: pd.Series | None = None):
    """
    Simple daily strategy:
    - Long when signal==1 (or prob>=threshold), else in cash (0).
    - No short for simplicity.
    """
    if prob is not None:
        pos = (prob >= threshold).astype(int)
    else:
        pos = (signal > 0).astype(int)
    # Daily returns
    rets = prices.pct_change().fillna(0.0)
    strat_rets = rets * pos.shift(1).fillna(0)  # enter next day to avoid lookahead
    equity = (1 + strat_rets).cumprod()
    return strat_rets, equity

def plot_price(df, price_col="Close", title="Price"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], mode="lines", name=price_col))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), title=title)
    return fig

def plot_equity(equity: pd.Series, title="Strategy Equity Curve"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), title=title)
    return fig

def plot_predictions(df, price_col="Close", proba=None, preds=None, title="Predictions vs Price"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], mode="lines", name="Price"))
    if proba is not None:
        fig.add_trace(go.Scatter(x=df.index, y=proba, mode="lines", name="Pred Prob(Up)", yaxis="y2", opacity=0.6))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        title=title,
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Prob Up", overlaying="y", side="right", range=[0,1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# =============== Sidebar Controls ===============
st.sidebar.header("⚙️ Controls")
st.sidebar.markdown("Configure data, features, target and models.")

with st.sidebar.expander("Data Input", expanded=True):
    uploaded = st.file_uploader("Upload CSV (with at least Date & Close). Optional Volume column.", type=["csv"])
    demo = st.checkbox("Use demo data if no file", value=True)
    date_col = st.text_input("Date column name", value="Date")
    price_col = st.text_input("Price column name", value="Close")

with st.sidebar.expander("Target (Label)", expanded=True):
    horizon = st.number_input("Forward horizon (days)", min_value=1, max_value=60, value=5, step=1)
    threshold = st.number_input("Return threshold for label (e.g., 0.0)", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    ensure_label = st.checkbox("Create/overwrite Target automatically", value=True)

with st.sidebar.expander("Train / Val / Test Split", expanded=False):
    tr_size = st.slider("Train size", 0.4, 0.8, 0.6, 0.05)
    va_size = st.slider("Validation size", 0.1, 0.4, 0.2, 0.05)

with st.sidebar.expander("Models", expanded=True):
    model_choices = st.multiselect(
        "Select models to train",
        ["RandomForest", "SVM-RBF", "MLP"],
        default=["RandomForest", "SVM-RBF", "MLP"]
    )
    prob_threshold = st.slider("Prob. threshold for trading", 0.1, 0.9, 0.5, 0.05)

with st.sidebar.expander("Advanced", expanded=False):
    dropna_tail = st.checkbox("Strictly drop tail rows with NaNs", value=True)
    scale_features = st.checkbox("Standardize features", value=True)
    risk_free = st.number_input("Annual risk-free rate (for ratios)", min_value=0.0, max_value=0.2, value=0.02, step=0.005)

# =============== Main ===============
st.title("📊 Machine Learning Applications in Finance — Interactive Dashboard")

# --- Data Loading ---
if uploaded is not None:
    try:
        df_raw = load_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
elif demo:
    df_raw = make_demo_data()
    st.info("Demo data generated (synthetic daily series). Upload a CSV to use real data.")
else:
    st.warning("Please upload a CSV or enable demo data.")
    st.stop()

# Ensure date index
try:
    df = ensure_datetime_index(df_raw.copy(), date_col=date_col)
except Exception as e:
    st.error(f"Date handling error: {e}")
    st.stop()

# Basic sanity checks
if price_col not in df.columns:
    st.error(f"Price column '{price_col}' not found in data.")
    st.stop()

# --- Feature Engineering & Target ---
work = add_technical_features(df, price_col=price_col)

# Prepare Target
if ensure_label or ("Target" not in work.columns):
    work["Target"] = create_target(work, price_col=price_col, horizon=horizon, threshold=threshold)

# Drop last 'horizon' rows where forward label is undefined
if dropna_tail:
    work = work.iloc[:-horizon] if len(work) > horizon else work.iloc[:0]

# Feature set
candidate_features = [
    "ret_1","ret_5","ret_10",
    "sma_5","sma_10","sma_20","ema_10","ema_20",
    "mom_10","rsi_14","vol_10","vol_20","pctB",
    "vol_chg_1","vol_sma_10"
]
feature_cols = [c for c in candidate_features if c in work.columns]

with st.expander("🔎 Data Preview & Feature Engineering", expanded=False):
    st.write("First rows (post features/label):")
    st.dataframe(work.head(20))
    st.caption(f"Features used ({len(feature_cols)}): {feature_cols}")

# Handle NaNs from rolling calculations
work = work.replace([np.inf, -np.inf], np.nan)
work = work.dropna(subset=[price_col, "Target"] + feature_cols)

if work.empty or work["Target"].nunique() < 2:
    st.error("Not enough labeled data after preprocessing (need both classes 0/1). Adjust horizon/threshold or provide more data.")
    st.stop()

# --- Split ---
train_df, val_df, test_df = timeseries_train_val_test_split(work, train_size=tr_size, val_size=va_size)
st.subheader("🔪 Time-Aware Split")
c1, c2, c3 = st.columns(3)
c1.metric("Train rows", len(train_df))
c2.metric("Validation rows", len(val_df))
c3.metric("Test rows", len(test_df))
st.caption(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()} | "
           f"Val: {val_df.index.min().date()} → {val_df.index.max().date()} | "
           f"Test: {test_df.index.min().date()} → {test_df.index.max().date()}")

# Prepare X/y
X_train, y_train, idx_train = prepare_Xy(train_df, feature_cols, "Target")
X_val, y_val, idx_val = prepare_Xy(val_df, feature_cols, "Target")
X_test, y_test, idx_test = prepare_Xy(test_df, feature_cols, "Target")

# Scale
scaler = None
if scale_features:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

# =============== Model Training ===============
st.subheader("🤖 Models & Evaluation")

def train_model(name: str):
    if name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )
        supports_proba = True
    elif name == "SVM-RBF":
        clf = SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=42)
        supports_proba = True
    elif name == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                            alpha=1e-4, learning_rate_init=1e-3, max_iter=300,
                            random_state=42, shuffle=False, early_stopping=True,
                            n_iter_no_change=15, validation_fraction=0.15)
        supports_proba = True
    else:
        raise ValueError("Unknown model.")
    clf.fit(X_train, y_train)
    return clf, supports_proba

results = []
tabs = st.tabs(model_choices if model_choices else ["No Model Selected"])

for tab, name in zip(tabs, model_choices):
    with tab:
        with st.spinner(f"Training {name}..."):
            clf, supports_proba = train_model(name)

        # Validation eval
        val_pred = clf.predict(X_val)
        val_proba = clf.predict_proba(X_val)[:,1] if supports_proba else None

        test_pred = clf.predict(X_test)
        test_proba = clf.predict_proba(X_test)[:,1] if supports_proba else None

        # Metrics
        def metrics_block(y_true, y_hat, y_prob, label="Val"):
            acc = accuracy_score(y_true, y_hat)
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
            except ValueError:
                auc = np.nan
            st.write(f"**{label} Metrics**")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1", f"{f1:.3f}")
            m5.metric("ROC AUC", f"{auc:.3f}" if not math.isnan(auc) else "n/a")
            return acc, prec, rec, f1, auc

        st.markdown("### Validation")
        v_acc, v_prec, v_rec, v_f1, v_auc = metrics_block(y_val, val_pred, val_proba, "Validation")

        st.markdown("### Test")
        t_acc, t_prec, t_rec, t_f1, t_auc = metrics_block(y_test, test_pred, test_proba, "Test")

        # Confusion matrix (Test)
        cm = confusion_matrix(y_test, test_pred)
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Pred", y="True", color="Count"),
                           x=["Down(0)", "Up(1)"], y=["Down(0)", "Up(1)"])
        cm_fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), title="Confusion Matrix (Test)")
        st.plotly_chart(cm_fig, use_container_width=True)

        # Predictions plot on Test
        test_frame = test_df.loc[idx_test].copy()
        if test_proba is not None:
            fig_pred = plot_predictions(test_frame, price_col=price_col, proba=test_proba, title=f"{name}: Price & Prob(Up) — Test")
        else:
            fig_pred = plot_price(test_frame, price_col=price_col, title=f"{name}: Price — Test")
        st.plotly_chart(fig_pred, use_container_width=True)

        # Backtest on Test
        prices_test = test_frame[price_col]
        if test_proba is not None:
            strat_rets, equity = backtest_equity_curve(prices_test, pd.Series(test_pred, index=idx_test),
                                                       threshold=prob_threshold, prob=pd.Series(test_proba, index=idx_test))
        else:
            strat_rets, equity = backtest_equity_curve(prices_test, pd.Series(test_pred, index=idx_test))

        bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
        bt_c1.metric("CAGR (approx)", f"{(equity.iloc[-1]**(252/len(equity)) - 1):.2%}" if len(equity)>0 else "n/a")
        bt_c2.metric("Sharpe", f"{sharpe_ratio(strat_rets, rf=risk_free):.2f}")
        bt_c3.metric("Sortino", f"{sortino_ratio(strat_rets, rf=risk_free):.2f}")
        bt_c4.metric("Max Drawdown", f"{rolling_max_drawdown(equity):.2%}")

        st.plotly_chart(plot_equity(equity, title=f"{name}: Strategy Equity (Test)"), use_container_width=True)

        # Save for comparison table
        results.append({
            "Model": name,
            "Val_Acc": v_acc, "Val_F1": v_f1, "Val_AUC": v_auc,
            "Test_Acc": t_acc, "Test_F1": t_f1, "Test_AUC": t_auc,
            "Test_Sharpe": sharpe_ratio(strat_rets, rf=risk_free),
            "Test_Sortino": sortino_ratio(strat_rets, rf=risk_free),
            "Test_MDD": rolling_max_drawdown(equity),
            "Test_CAGR_est": (equity.iloc[-1]**(252/len(equity)) - 1) if len(equity)>0 else np.nan
        })

# =============== Comparison ===============
if results:
    st.subheader("📌 Model Comparison (Validation & Test)")
    comp = pd.DataFrame(results)
    st.dataframe(
        comp.set_index("Model").style.format({
            "Val_Acc":"{:.3f}", "Val_F1":"{:.3f}", "Val_AUC":"{:.3f}",
            "Test_Acc":"{:.3f}", "Test_F1":"{:.3f}", "Test_AUC":"{:.3f}",
            "Test_Sharpe":"{:.2f}", "Test_Sortino":"{:.2f}",
            "Test_MDD":"{:.2%}", "Test_CAGR_est":"{:.2%}"
        }),
        use_container_width=True
    )

# =============== Raw Charts ===============
st.subheader("📈 Price Chart")
st.plotly_chart(plot_price(work, price_col=price_col, title="Close Price"), use_container_width=True)

# =============== Download Section ===============
st.subheader("⬇️ Exports")
# Processed dataset
processed_csv = work.reset_index().rename(columns={"index":"Date"}).to_csv(index=False).encode("utf-8")
st.download_button(
    "Download processed dataset (CSV)",
    data=processed_csv,
    file_name="processed_features_labels.csv",
    mime="text/csv"
)

# Comparison table
if results:
    comp_csv = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download model comparison (CSV)",
        data=comp_csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

# =============== Template Helper ===============
with st.expander("📄 CSV Template (click to download)"):
    template = pd.DataFrame({
        "Date": pd.date_range("2022-01-03", periods=30, freq="B"),
        "Close": np.linspace(100, 110, 30),
        "Volume": np.random.randint(100000, 200000, 30)
    })
    buf = io.StringIO()
    template.to_csv(buf, index=False)
    st.download_button(
        "Download minimal template",
        data=buf.getvalue().encode("utf-8"),
        file_name="price_data_template.csv",
        mime="text/csv"
    )

st.caption("Tip: Label = 1 if future return over the chosen horizon exceeds threshold; else 0. Strategy goes long when predicted probability ≥ threshold (no short).")
