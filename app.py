from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from data import fetch_many, default_universe
from features import build_features
from models import time_split_train_eval, fit_best_and_signal
from eval import backtest_long_flat
from utils import safe_pct, non_empty

st.set_page_config(page_title="IBR Finance: EDA + ML Comparison", layout="wide")

st.title("📊 IBR Finance Dashboard — EDA + RF vs SVM vs ANN")

with st.sidebar:
    st.header("Data & Settings")
    market = st.selectbox("Market", ["US","EM (India)"])
    period = st.selectbox("History Window", ["2y","3y","5y","10y"], index=2)
    interval = st.selectbox("Frequency", ["1d","1wk"], index=0)
    default_syms = default_universe("US" if market=="US" else "EM")
    symbols_text = st.text_area("Symbols (comma separated)", value=",".join(default_syms), height=80)
    symbols = [s.strip() for s in symbols_text.split(",") if s.strip()]
    st.caption("Tip: add/remove tickers. For India, use the .NS suffix.")

    run_btn = st.button("Run Analysis", type="primary")

st.markdown(
    "> **Objectives covered**: Emphasis on EDA; automatic data (no uploads); "
    "model comparison **RF vs SVM vs ANN(MLP)**; simple backtest; risk metrics; "
    "developed vs emerging toggle, aligned to your IBR."
)

@st.cache_data(show_spinner=True, ttl=3600)
def _load_data(symbols, period, interval) -> pd.DataFrame:
    return fetch_many(symbols, period=period, interval=interval)

def _eda_section(df: pd.DataFrame):
    st.subheader("1) Exploratory Data Analysis (EDA)")
    # universe snapshot
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Symbols", f"{df['Symbol'].nunique():,}")
    with c2:
        st.metric("Observations", f"{df.shape[0]:,}")
    with c3:
        st.metric("Start → End", f"{df.index.min().date()} → {df.index.max().date()}")

    # missingness
    miss = df[["Open","High","Low","Close","Price","Volume"]].isna().mean().sort_values(ascending=False)
    miss_df = miss.to_frame("Missing%")
    miss_df["Missing%"] = (miss_df["Missing%"]*100).round(2)
    st.write("**Missingness by column (%)**")
    st.dataframe(miss_df)

    # returns distribution (stacked across symbols)
    ret = df.groupby("Symbol")["Price"].pct_change()
    hist = alt.Chart(pd.DataFrame({"Ret": ret.dropna()})).mark_bar().encode(
        alt.X("Ret", bin=alt.Bin(maxbins=60), title="Daily Return"),
        y="count()"
    ).properties(height=200)
    st.altair_chart(hist, use_container_width=True)

    # seasonality line (average by month)
    df["Ret1"] = df.groupby("Symbol")["Price"].pct_change()
    df["Month"] = df.index.month
    seas = df.groupby("Month")["Ret1"].mean().reset_index()
    line = alt.Chart(seas).mark_line(point=True).encode(
        x=alt.X("Month:O"), y=alt.Y("Ret1:Q", title="Avg Daily Return")
    ).properties(height=220)
    st.altair_chart(line, use_container_width=True)

    # correlation heatmap of returns across top symbols (by data count)
    top_syms = df["Symbol"].value_counts().index[:10]
    wide = df[df["Symbol"].isin(top_syms)].pivot_table(index=df.index, columns="Symbol", values="Ret1")
    corr = wide.corr().stack().reset_index()
    corr.columns = ["SymA","SymB","Corr"]
    heat = alt.Chart(corr).mark_rect().encode(
        x="SymA:O", y="SymB:O", color=alt.Color("Corr:Q", scale=alt.Scale(scheme="redblue", domain=(-1,1))),
        tooltip=["SymA","SymB",alt.Tooltip("Corr:Q", format=".2f")]
    ).properties(height=300)
    st.altair_chart(heat, use_container_width=True)

def _model_section(feat: pd.DataFrame):
    st.subheader("2) Model Benchmark: RF vs SVM vs ANN (MLP)")
    with st.spinner("Training time-series CV…"):
        results = time_split_train_eval(feat.dropna())
    rows = []
    for name, d in results.items():
        rows.append({"Model": name, "Accuracy (CV)": d["acc"], "ROC-AUC (CV)": d["auc"]})
    res_df = pd.DataFrame(rows).sort_values("Accuracy (CV)", ascending=False)
    st.dataframe(res_df.style.format({"Accuracy (CV)": "{:.3f}", "ROC-AUC (CV)": "{:.3f}"}), use_container_width=True)
    best = res_df.iloc[0]["Model"]
    st.success(f"Best CV model: **{best}**")
    return best

def _backtest_section(feat: pd.DataFrame, best_model: str, raw: pd.DataFrame):
    st.subheader("3) Backtest (Long/Flat on cross-sectional avg)")
    with st.spinner("Fitting best model and generating signals…"):
        pipe, signal = fit_best_and_signal(feat.dropna(), best_model)
    # stitch signal back
    feat2 = feat.copy()
    feat2["Signal"] = signal
    bt = backtest_long_flat(feat2, "Signal")
    met = bt["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ann. Return", safe_pct(met["Annualized Return"]))
    c2.metric("Sharpe", f"{met['Sharpe']:.2f}")
    c3.metric("Sortino", f"{met['Sortino']:.2f}")
    c4.metric("Max Drawdown", safe_pct(abs(met["Max Drawdown"])))

    # equity vs benchmark
    eq = pd.DataFrame({
        "Date": bt["curves"]["equity"].index,
        "Strategy": bt["curves"]["equity"].values,
        "Benchmark": bt["curves"]["benchmark"].reindex(bt["curves"]["equity"].index).values
    })
    eq_long = eq.melt("Date", var_name="Series", value_name="Value")
    line = alt.Chart(eq_long).mark_line().encode(
        x="Date:T", y=alt.Y("Value:Q", title="Growth of 1.0"), color="Series:N"
    ).properties(height=300)
    st.altair_chart(line, use_container_width=True)

    st.caption(
        f"Hit‑rate={met['Hit Rate (Days)']:.2f}, "
        f"CumRet={safe_pct(met['Cumulative Return'])}, "
        f"Benchmark={safe_pct(met['Benchmark CumRet'])}."
    )

if run_btn:
    try:
        raw = _load_data(symbols, period, interval)
        if not non_empty(raw):
            st.error("No data returned — try different symbols/period.")
        else:
            _eda_section(raw.copy())  # EDA uses raw OHLCV + returns

            # Feature engineering
            st.subheader("Feature Engineering")
            feat = build_features(raw.copy())
            st.write("Feature sample")
            st.dataframe(
                feat.groupby("Symbol").tail(5)[["Symbol","Price","Ret1","SMA20","EMA12","MACD","RSI14","Volatility20","TargetNextUp"]]
            )

            # Model benchmark
            best_model = _model_section(feat.copy())

            # Backtest
            _backtest_section(feat.copy(), best_model, raw.copy())

            st.info("Notes: This is a simplified empirical setup with time‑series CV, pooled symbols, and a long/flat backtest. "
                    "For your thesis, you can extend with per‑market splits, sector stratification, and transaction costs.")
    except Exception as e:
        st.exception(e)
else:
    st.write("⬅️ Configure options, then click **Run Analysis**.")
