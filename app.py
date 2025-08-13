# app.py
import streamlit as st
import pandas as pd
import numpy as np

from data import fetch_many
from features import build_indicators, prepare_ml_frame, compute_kpis
from modeling import train_and_score_models, backtest_from_proba
from ui import (
    kpi_row, price_with_bands, rsi_chart, macd_chart, returns_hist,
    model_table, equity_curve_chart, corr_heatmap, info_note
)

st.set_page_config(page_title="IBR Finance — ML & Markets", page_icon="📈", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

markets = st.sidebar.selectbox(
    "Market set",
    ["Developed (DM)", "Emerging (EM)", "Custom"],
    index=0
)

DM_DEFAULT = ["SPY", "AAPL", "MSFT", "NVDA", "QQQ", "GLD"]
EM_DEFAULT = ["NIFTYBEES.NS", "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "HDFCBANK.NS", "KOTAKBANK.NS"]

if markets == "Developed (DM)":
    tickers = st.sidebar.text_input("Tickers (comma-separated)", ", ".join(DM_DEFAULT))
elif markets == "Emerging (EM)":
    tickers = st.sidebar.text_input("Tickers (comma-separated)", ", ".join(EM_DEFAULT))
else:
    tickers = st.sidebar.text_input("Tickers (comma-separated)", "SPY, AAPL, BTC-USD")

period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Indicators")
rsi_win = st.sidebar.slider("RSI window", 5, 30, 14, 1)
bb_win = st.sidebar.slider("Bollinger window", 10, 40, 20, 1)
bb_std = st.sidebar.slider("Bollinger std", 1.0, 3.0, 2.0, 0.5)
ema_fast = st.sidebar.slider("EMA fast", 5, 20, 12, 1)
ema_slow = st.sidebar.slider("EMA slow", 15, 50, 26, 1)
macd_sig = st.sidebar.slider("MACD signal", 5, 20, 9, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("ML Target")
horizon = st.sidebar.slider("Forward horizon (days)", 1, 30, 5, 1)
threshold = st.sidebar.slider("Forward return threshold", -0.05, 0.05, 0.0, 0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")
prob_threshold = st.sidebar.slider("Long when Prob(Up) ≥", 0.1, 0.9, 0.5, 0.05)
est_cost_bps = st.sidebar.slider("Transaction cost (bps per trade)", 0, 50, 5, 1)

# ---------------- HEADER ----------------
st.title("📊 IBR Finance — ML Applications in Finance (DM vs EM)")
st.caption("No upload required • Live data via yfinance • EDA + ML + Backtest • Objectives-aligned")

# ---------------- DATA ----------------
symbols = [t.strip() for t in tickers.split(",") if t.strip()]
with st.spinner("Fetching market data..."):
    panel = fetch_many(symbols, period=period, interval=interval)

if not panel:
    st.error("No data fetched. Check tickers or internet access.")
    st.stop()

# ---------------- TABS PER TICKER ----------------
tabs = st.tabs(symbols)

for tab, sym in zip(tabs, symbols):
    with tab:
        df = panel.get(sym)
        if df is None or df.empty:
            st.warning(f"{sym}: no data.")
            continue

        # Indicators & EDA frame
        df_ind = build_indicators(df,
                                  rsi_window=rsi_win,
                                  bb_window=bb_win,
                                  bb_std=bb_std,
                                  ema_fast=ema_fast,
                                  ema_slow=ema_slow,
                                  macd_signal=macd_sig)

        # KPIs (Objective 1: real-world effect, baseline)
        k = compute_kpis(df_ind)
        kpi_row(k)

        # Charts
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            st.plotly_chart(price_with_bands(df_ind), use_container_width=True)
        with c2:
            st.plotly_chart(rsi_chart(df_ind), use_container_width=True)
            st.plotly_chart(macd_chart(df_ind), use_container_width=True)

        c3, c4 = st.columns([1, 1], gap="large")
        with c3:
            st.plotly_chart(returns_hist(df_ind), use_container_width=True)
        with c4:
            st.plotly_chart(corr_heatmap(df_ind), use_container_width=True)

        st.markdown("### 🤖 Model Comparison (RF • SVM • MLP) — Objective 2")
        # ML frame (features + label)
        ml_df = prepare_ml_frame(df_ind, horizon=horizon, threshold=threshold)
        if ml_df is None or ml_df.empty or ml_df["Target"].nunique() < 2:
            info_note("Not enough labeled data after preprocessing (need both classes 0/1). Adjust horizon/threshold or choose a longer period.")
            continue

        # Train & Evaluate (time-aware split)
        with st.spinner("Training models..."):
            results, preds = train_and_score_models(ml_df)

        st.plotly_chart(model_table(results), use_container_width=True)

        st.markdown("### 📈 Backtest — Objective 1 (Real-world effect from model output)")
        # Choose the best validation AUC model for backtest
        best = results.sort_values("Val_AUC", ascending=False).iloc[0]
        best_name = best["Model"]
        proba_test = preds[best_name]["test_proba"]
        price_test = preds[best_name]["price_test"]
        dates_test = preds[best_name]["index_test"]

        # Backtest from probas
        bt = backtest_from_proba(
            price=price_test, proba=proba_test, index=dates_test,
            prob_threshold=prob_threshold, cost_bps=est_cost_bps
        )

        # Equity curve and metrics
        st.plotly_chart(equity_curve_chart(bt["equity"]), use_container_width=True)

        bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
        bt_c1.metric("CAGR", f"{bt['CAGR']:.2%}" if np.isfinite(bt['CAGR']) else "—")
        bt_c2.metric("Sharpe", f"{bt['Sharpe']:.2f}" if np.isfinite(bt['Sharpe']) else "—")
        bt_c3.metric("Sortino", f"{bt['Sortino']:.2f}" if np.isfinite(bt['Sortino']) else "—")
        bt_c4.metric("Max Drawdown", f"{bt['MaxDD']:.2%}" if np.isfinite(bt['MaxDD']) else "—")

        st.caption(f"Backtest based on best validation AUC model: **{best_name}**. "
                   f"Long when Prob(Up) ≥ {prob_threshold:.2f}, cost = {est_cost_bps} bps/trade.")

st.markdown("---")
st.markdown("**Objective 3 (DM vs EM):** Use the tabs with your selected DM/EM watchlists to compare KPIs, distributions, model AUC/F1, and backtest outcomes across regions.")
