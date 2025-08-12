import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ----------------------------
# Basic setup
# ----------------------------
st.set_page_config(page_title="ML in Finance — Multi‑Market Dashboard", layout="wide")
st.title("ML in Finance — Multi‑Market Dashboard")
st.caption("S&P 500 · Nifty 50 · FTSE 100 · Bovespa | EDA · Benchmarking · Interpretability | v1.0")

DATA_MAP = {
    "S&P 500": ("sp500_full_data.csv", "sp500_full_features.csv"),
    "Nifty 50": ("nifty50_full_data.csv", "nifty50_full_features.csv"),
    "FTSE 100": ("ftse100_full_data.csv", "ftse100_full_features.csv"),
    "Bovespa": ("bovespa_full_data.csv", "bovespa_full_features.csv"),
}

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data
def load_price_data(csv_path: str) -> pd.DataFrame:
    """Expected columns: Date, Ticker, Close, High, Low, Open, Volume."""
    df = pd.read_csv(csv_path)
    # Robust date parsing
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].sort_values("Date")
    return df

@st.cache_data
def load_feature_data(csv_path: str) -> pd.DataFrame:
    """Expected columns: Date, Ticker, Close, High, Low, Open, Volume, features..., Target"""
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df[df["Date"].notna()].sort_values(["Ticker", "Date"])
    return df

def feature_cols(df: pd.DataFrame) -> list:
    block = {"Date","Ticker","Target","Close","High","Low","Open","Volume"}
    return [c for c in df.columns if c not in block and df[c].dtype != "O"]

def train_benchmark_models(df_feat: pd.DataFrame, ticker_sel: list | None = None):
    # Filter by tickers if provided
    work = df_feat.copy()
    if ticker_sel:
        work = work[work["Ticker"].isin(ticker_sel)].copy()
    # Drop any rows without target
    work = work.dropna(subset=["Target"])
    feats = feature_cols(work)
    if not feats:
        return None, "No feature columns found."

    # TimeSeries split by date
    work = work.sort_values("Date")
    X = work[feats].values
    y = work["Target"].astype(int).values

    # Use last 25% as test (simple split)
    split_idx = int(len(work) * 0.75)
    if split_idx == 0 or split_idx >= len(work)-1:
        return None, "Not enough rows to split train/test."

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results = []
    extras = {}

    # 1) Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(("RandomForest",
                    accuracy_score(y_test, y_pred_rf),
                    precision_score(y_test, y_pred_rf, zero_division=0),
                    recall_score(y_test, y_pred_rf, zero_division=0),
                    f1_score(y_test, y_pred_rf, zero_division=0)))
    extras["rf_conf"] = confusion_matrix(y_test, y_pred_rf)
    extras["rf_importance"] = (feats, rf.feature_importances_)

    # 2) SVM (scaled)
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42)),
    ])
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    results.append(("SVM",
                    accuracy_score(y_test, y_pred_svm),
                    precision_score(y_test, y_pred_svm, zero_division=0),
                    recall_score(y_test, y_pred_svm, zero_division=0),
                    f1_score(y_test, y_pred_svm, zero_division=0)))
    extras["svm_conf"] = confusion_matrix(y_test, y_pred_svm)

    # Permutation importance for SVM (costly but informative; subsample to speed up)
    try:
        sample_idx = np.linspace(0, len(X_test)-1, min(1000, len(X_test)), dtype=int)
        perm = permutation_importance(
            svm, X_test[sample_idx], y_test[sample_idx],
            n_repeats=5, random_state=42, n_jobs=-1
        )
        extras["svm_perm"] = (feats, perm.importances_mean)
    except Exception:
        extras["svm_perm"] = None

    # 3) ANN / MLP (scaled)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                              max_iter=300, random_state=42))
    ])
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    results.append(("ANN",
                    accuracy_score(y_test, y_pred_mlp),
                    precision_score(y_test, y_pred_mlp, zero_division=0),
                    recall_score(y_test, y_pred_mlp, zero_division=0),
                    f1_score(y_test, y_pred_mlp, zero_division=0)))
    extras["mlp_conf"] = confusion_matrix(y_test, y_pred_mlp)

    # Pack result table
    res_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
    res_df = res_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return (res_df, extras, (y_test, {"RF":y_pred_rf,"SVM":y_pred_svm,"ANN":y_pred_mlp})), None

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")
market = st.sidebar.selectbox("Market", list(DATA_MAP.keys()))
price_csv, feat_csv = DATA_MAP[market]

if not os.path.exists(price_csv) or not os.path.exists(feat_csv):
    st.error(f"Missing files for {market}. Expected:\n- {price_csv}\n- {feat_csv}")
    st.stop()

# Load data
price_df = load_price_data(price_csv)
feat_df  = load_feature_data(feat_csv)

# Global ticker selection (shared by tabs)
tickers = sorted(price_df["Ticker"].dropna().unique().tolist())
sel_tickers = st.sidebar.multiselect("Tickers (optional subset)", tickers, default=[])

# Date range filter for EDA
min_d, max_d = price_df["Date"].min(), price_df["Date"].max()
date_range = st.sidebar.date_input(
    "EDA date range",
    value=(pd.to_datetime(min_d).date(), pd.to_datetime(max_d).date()),
    min_value=pd.to_datetime(min_d).date(),
    max_value=pd.to_datetime(max_d).date()
)
if isinstance(date_range, tuple):
    d0, d1 = [pd.to_datetime(x) for x in date_range]
else:
    d0, d1 = pd.to_datetime(min_d), pd.to_datetime(max_d)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Visualization (EDA)",
    "ML Model Results",
    "Cross‑Market Compare",
    "About & Research Objectives"
])

# ----------------------------
# TAB 1 — EDA
# ----------------------------
with tab1:
    st.subheader(f"Exploratory Data Analysis — {market}")
    eda = price_df.copy()
    if sel_tickers:
        eda = eda[eda["Ticker"].isin(sel_tickers)]
    eda = eda[(eda["Date"]>=d0) & (eda["Date"]<=d1)]

    if eda.empty:
        st.info("No rows in the selected date range / tickers.")
        st.stop()

    t_opt = st.selectbox("Choose one ticker for charts", sorted(eda["Ticker"].unique().tolist()))
    sub = eda[eda["Ticker"]==t_opt].sort_values("Date").copy()

    # 1) Price + 20‑day SMA
    sub["SMA20"] = sub["Close"].rolling(20).mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(sub["Date"], sub["Close"], label="Close")
    ax.plot(sub["Date"], sub["SMA20"], label="SMA(20)")
    ax.set_title(f"{t_opt}: Close vs 20‑Day SMA")
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    st.caption("SMA smooths price; crossovers often signal momentum shifts.")

    # 2) Daily Return Distribution
    sub["Return"] = sub["Close"].pct_change()
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(sub["Return"].dropna(), bins=50)
    ax2.set_title(f"{t_opt}: Daily Return Distribution")
    ax2.set_xlabel("Daily Return"); ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    st.caption("Shows asymmetry and tails in returns; useful for risk awareness beyond the mean.")

    # 3) 20‑day rolling volatility
    sub["Vol20"] = sub["Return"].rolling(20).std() * np.sqrt(252)
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.plot(sub["Date"], sub["Vol20"])
    ax3.set_title(f"{t_opt}: 20‑Day Rolling Volatility (annualized)")
    ax3.set_xlabel("Date"); ax3.set_ylabel("Volatility")
    st.pyplot(fig3)
    st.caption("Rising volatility usually coincides with uncertainty/regime shifts (risk management).")

    # 4) Correlation heatmap (Close → pct_change, wide pivot)
    st.markdown("**Correlation (returns) across selected tickers**")
    corr_df = eda.pivot(index="Date", columns="Ticker", values="Close").pct_change()
    corr = corr_df.corr().fillna(0)
    fig4, ax4 = plt.subplots(figsize=(6,5))
    im = ax4.imshow(corr.values, aspect="auto")
    ax4.set_xticks(range(len(corr.columns))); ax4.set_xticklabels(corr.columns, rotation=90)
    ax4.set_yticks(range(len(corr.index)));  ax4.set_yticklabels(corr.index)
    ax4.set_title("Correlation Heatmap (Daily Returns)")
    fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    st.pyplot(fig4)
    st.caption("Helps spot co‑movement/clustered risk; useful for diversification.")

# ----------------------------
# TAB 2 — ML Model Results
# ----------------------------
with tab2:
    st.subheader(f"Model Benchmarking — {market}")
    st.write("Models: RandomForest, SVM (RBF), ANN/MLP. Target assumed as classification label in *_full_features.csv.")
    if sel_tickers:
        st.caption(f"Training on selected tickers only: {', '.join(sel_tickers)}")

    with st.spinner("Training & evaluating models…"):
        outcome = train_benchmark_models(feat_df, sel_tickers if sel_tickers else None)

    if outcome[1] is not None:
        st.error(outcome[1])
    else:
        (res_df, extras, (y_test, preds)) = outcome[0]
        st.dataframe(res_df, use_container_width=True)

        # Download results
        csv_bytes = res_df.to_csv(index=False).encode()
        st.download_button("Download results (CSV)", csv_bytes, file_name=f"{market.replace(' ','_').lower()}_model_results.csv")

        # Confusion matrices
        colc1, colc2, colc3 = st.columns(3)
        for name, cm, col in [
            ("RandomForest", extras["rf_conf"], colc1),
            ("SVM",          extras["svm_conf"], colc2),
            ("ANN",          extras["mlp_conf"], colc3),
        ]:
            with col:
                fig, ax = plt.subplots(figsize=(3.5,3.2))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_title(f"{name} — Confusion Matrix")
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                for (i,j), v in np.ndenumerate(cm):
                    ax.text(j, i, int(v), ha='center', va='center')
                st.pyplot(fig)

        # Feature importance / permutation importance
        st.markdown("### Interpretability")
        rf_feats, rf_imp = extras["rf_importance"]
        imp_rf = pd.DataFrame({"Feature": rf_feats, "Importance": rf_imp}).sort_values("Importance", ascending=False).head(15)
        st.write("**RandomForest — Top Features**")
        st.dataframe(imp_rf, use_container_width=True)
        figi, axi = plt.subplots(figsize=(6,4))
        axi.barh(imp_rf["Feature"][::-1], imp_rf["Importance"][::-1])
        axi.set_title("RandomForest Feature Importance (Top 15)")
        st.pyplot(figi)

        st.write("**SVM — Permutation Importance (Top 15)**")
        if extras["svm_perm"] is not None:
            svm_feats, svm_imp = extras["svm_perm"]
            imp_svm = pd.DataFrame({"Feature": svm_feats, "Importance": svm_imp}).sort_values("Importance", ascending=False).head(15)
            st.dataframe(imp_svm, use_container_width=True)
            figp, axp = plt.subplots(figsize=(6,4))
            axp.barh(imp_svm["Feature"][::-1], imp_svm["Importance"][::-1])
            axp.set_title("SVM Permutation Importance (Top 15)")
            st.pyplot(figp)
        else:
            st.info("Permutation importance for SVM not available (insufficient data or pipeline constraints).")

        # Light narrative
        best_row = res_df.iloc[0]
        st.markdown(
            f"""**Quick take:** On this sample split, **{best_row['Model']}** performs best by accuracy
            ({best_row['Accuracy']:.3f}). Feature ranking above helps interpret what drives signals —
            key for Objective 1 (practical decision impact) & Objective 2 (model traits)."""
        )

# ----------------------------
# TAB 3 — Cross‑Market Compare
# ----------------------------
with tab3:
    st.subheader("Cross‑Market Comparison")
    st.caption("Runs a fast benchmark on all markets with the same settings to compare headline metrics.")

    if st.button("Run cross‑market benchmark"):
        rows = []
        for mkt, (pcsv, fcsv) in DATA_MAP.items():
            try:
                fdf = load_feature_data(fcsv)
                out = train_benchmark_models(fdf, sel_tickers if sel_tickers else None)
                if out[1] is None:
                    res = out[0][0]
                    res["Market"] = mkt
                    rows.append(res)
            except Exception as e:
                st.warning(f"{mkt}: {e}")
        if rows:
            allres = pd.concat(rows, ignore_index=True)
            allres = allres[["Market","Model","Accuracy","Precision","Recall","F1"]]
            st.dataframe(allres, use_container_width=True)

            # Simple compare plot
            figc, axc = plt.subplots(figsize=(9,4))
            for mkt in allres["Market"].unique():
                sub = allres[allres["Market"]==mkt]
                axc.plot(sub["Model"], sub["Accuracy"], marker="o", label=mkt)
            axc.set_ylabel("Accuracy"); axc.set_title("Accuracy by Model across Markets")
            axc.legend()
            st.pyplot(figc)

            st.download_button(
                "Download cross‑market results (CSV)",
                allres.to_csv(index=False).encode(),
                file_name="cross_market_results.csv"
            )
        else:
            st.info("No results computed.")

# ----------------------------
# TAB 4 — About & Research Objectives
# ----------------------------
with tab4:
    st.subheader("About this Project")
    st.write("""
This dashboard supports the thesis **“Machine Learning applications in Finance”** by operationalising the three objectives:

- **Objective 1 — Real‑world effect:** We translate model predictions into investment‑relevant signals (classification of next‑day move) and show risk views (volatility) in EDA.
- **Objective 2 — Model compatibility & traits:** We benchmark **RF, SVM, ANN** on identical data and expose **confusion matrices** and **feature/permutation importance**.
- **Objective 3 — Emerging vs developed markets:** The **Cross‑Market** tab lets you contrast performance across S&P 500, Nifty 50, FTSE 100, and Bovespa.

**Data expectations**  
`*_full_data.csv` contain OHLCV by Date & Ticker.  
`*_full_features.csv` contain engineered features plus a binary `Target` column.

> Tip: keep engineered features consistent across markets so comparisons remain fair.
""")
