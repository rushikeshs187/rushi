import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data import fetch_many, SUPPORTED_PERIODS, SUPPORTED_INTERVALS
from features import build_indicators
from backtest import run_backtest
from utils import kpis_from_returns, format_kpi

st.set_page_config(
    page_title="IBR Finance – ML & TA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar ----
st.sidebar.title("⚙️ Settings")

default_universe = "US Mega Caps"
universes = {
    "US Mega Caps": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "NIFTY 50 (sample)": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
    "Crypto (spot)": ["BTC-USD", "ETH-USD"],
}
universe_name = st.sidebar.selectbox("Universe", list(universes.keys()), index=list(universes.keys()).index(default_universe))
symbols = st.sidebar.multiselect("Symbols", universes[universe_name], default=universes[universe_name])

period = st.sidebar.selectbox("History (period)", SUPPORTED_PERIODS, index=SUPPORTED_PERIODS.index("1y"))
interval = st.sidebar.selectbox("Bar interval", SUPPORTED_INTERVALS, index=SUPPORTED_INTERVALS.index("1d"))
auto_adjust = st.sidebar.checkbox("Auto-adjust prices (splits/dividends)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Indicators")
rsi_win = st.sidebar.number_input("RSI window", min_value=5, max_value=100, value=14, step=1)
ema_fast = st.sidebar.number_input("EMA fast", min_value=2, max_value=200, value=12, step=1)
ema_slow = st.sidebar.number_input("EMA slow", min_value=2, max_value=400, value=26, step=1)
macd_sig = st.sidebar.number_input("MACD signal", min_value=2, max_value=100, value=9, step=1)
bb_win = st.sidebar.number_input("Bollinger window", min_value=5, max_value=200, value=20, step=1)
bb_std = st.sidebar.number_input("Bollinger std dev", min_value=1.0, max_value=4.0, value=2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")
strategy = st.sidebar.selectbox("Strategy", ["SMA Crossover", "RSI Mean-Revert"])
sma_fast = st.sidebar.number_input("SMA fast (for crossover)", min_value=5, max_value=200, value=20, step=1)
sma_slow = st.sidebar.number_input("SMA slow (for crossover)", min_value=10, max_value=400, value=50, step=1)
rsi_buy = st.sidebar.number_input("RSI buy-threshold", min_value=1, max_value=99, value=30, step=1)
rsi_sell = st.sidebar.number_input("RSI sell-threshold", min_value=1, max_value=99, value=70, step=1)
tcost_bps = st.sidebar.number_input("Transaction cost (bps round-trip)", min_value=0, max_value=100, value=5, step=1)

st.title("📊 IBR Finance – ML/TA Dashboard")
st.caption("Auto data fetch · EDA · Indicators · Backtest · KPIs (CAGR, Sharpe, Sortino, MaxDD)")

# ---- Data ----
if not symbols:
    st.warning("Select at least one symbol in the sidebar.")
    st.stop()

with st.spinner("Fetching data..."):
    panel = fetch_many(symbols, period=period, interval=interval, auto_adjust=auto_adjust)

valid_symbols = [s for s, df in panel.items() if isinstance(df, pd.DataFrame) and not df.empty]
if not valid_symbols:
    st.error("No data could be retrieved for the requested symbols. Try a different period/interval or universe.")
    st.stop()

tab_prices, tab_eda, tab_ind, tab_bt = st.tabs(["Prices", "EDA", "Indicators", "Backtest"])

# ---- Prices Tab ----
with tab_prices:
    sym = st.selectbox("Symbol", valid_symbols, key="price_symbol")
    df = panel[sym].copy()
    df = df.sort_index()
    st.write(f"**{sym}** · {df.index.min().date()} → {df.index.max().date()} · {len(df)} rows")

    # Candlestick + EMAs + BB
    df_ind = build_indicators(
        df,
        rsi_window=rsi_win,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        macd_signal=macd_sig,
        bb_window=bb_win,
        bb_std=bb_std,
    )

    candle = go.Figure()
    candle.add_trace(go.Candlestick(
        x=df_ind.index, open=df_ind["Open"], high=df_ind["High"],
        low=df_ind["Low"], close=df_ind["Close"], name="OHLC"
    ))
    for col in ["EMA_Fast", "EMA_Slow", "BB_Mid", "BB_Upper", "BB_Lower"]:
        if col in df_ind.columns:
            candle.add_trace(go.Scatter(x=df_ind.index, y=df_ind[col], mode="lines", name=col))
    candle.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(candle, use_container_width=True)

# ---- EDA Tab ----
with tab_eda:
    sym2 = st.selectbox("Symbol (EDA)", valid_symbols, index=min(1, len(valid_symbols)-1), key="eda_symbol")
    dfe = panel[sym2].copy().sort_index()
    st.write("**Preview**")
    st.dataframe(dfe.tail(10))

    st.write("**Summary**")
    st.dataframe(dfe.describe().T)

    st.write("**Missing values by column**")
    miss = dfe.isna().sum().reset_index()
    miss.columns = ["column", "missing"]
    bar = px.bar(miss, x="column", y="missing", title=None)
    bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, use_container_width=True)

    st.write("**Daily returns distribution**")
    if "Close" in dfe.columns:
        ret = dfe["Close"].pct_change().dropna()
        hist = px.histogram(ret, nbins=60, title=None)
        hist.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(hist, use_container_width=True)
        k = kpis_from_returns(ret, trading_days=252)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", format_kpi(k["cagr"]))
        c2.metric("Sharpe", f'{k["sharpe"]:.2f}')
        c3.metric("Sortino", f'{k["sortino"]:.2f}')
        c4.metric("Max Drawdown", format_kpi(k["max_drawdown"]))
    else:
        st.info("Close column not found; cannot compute returns.")

# ---- Indicators Tab ----
with tab_ind:
    sym3 = st.selectbox("Symbol (Indicators)", valid_symbols, key="ind_symbol")
    dfi = build_indicators(panel[sym3].copy().sort_index(),
                           rsi_window=rsi_win, ema_fast=ema_fast, ema_slow=ema_slow,
                           macd_signal=macd_sig, bb_window=bb_win, bb_std=bb_std)

    # RSI
    if "RSI" in dfi.columns:
        rfig = px.line(dfi, x=dfi.index, y="RSI", title=None)
        rfig.add_hline(y=70, line_dash="dot")
        rfig.add_hline(y=30, line_dash="dot")
        rfig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(rfig, use_container_width=True)

    # MACD
    mac_cols = [c for c in ["MACD", "MACD_Signal", "MACD_Hist"] if c in dfi.columns]
    if mac_cols:
        mfig = px.line(dfi, x=dfi.index, y=mac_cols, title=None)
        mfig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(mfig, use_container_width=True)

# ---- Backtest Tab ----
with tab_bt:
    sym4 = st.selectbox("Symbol (Backtest)", valid_symbols, key="bt_symbol")
    dfb = build_indicators(panel[sym4].copy().sort_index(),
                           rsi_window=rsi_win, ema_fast=ema_fast, ema_slow=ema_slow,
                           macd_signal=macd_sig, bb_window=bb_win, bb_std=bb_std)

    if dfb.empty or "Close" not in dfb.columns:
        st.warning("Not enough data for backtest.")
        st.stop()

    res = run_backtest(
        dfb,
        strategy=strategy,
        sma_fast=int(sma_fast),
        sma_slow=int(sma_slow),
        rsi_buy=int(rsi_buy),
        rsi_sell=int(rsi_sell),
        tcost_bps=int(tcost_bps),
    )

    eq = res["equity"]
    bench = (dfb["Close"] / dfb["Close"].iloc[0]).dropna()

    k = kpis_from_returns(eq.pct_change().dropna(), trading_days=252)
    kb = kpis_from_returns(bench.pct_change().dropna(), trading_days=252)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR (Strat)", format_kpi(k["cagr"]))
    c2.metric("Sharpe (Strat)", f'{k["sharpe"]:.2f}')
    c3.metric("Sortino (Strat)", f'{k["sortino"]:.2f}')
    c4.metric("Max DD (Strat)", format_kpi(k["max_drawdown"]))

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("CAGR (Buy&Hold)", format_kpi(kb["cagr"]))
    c2b.metric("Sharpe (Buy&Hold)", f'{kb["sharpe"]:.2f}')
    c3b.metric("Sortino (Buy&Hold)", f'{kb["sortino"]:.2f}')
    c4b.metric("Max DD (Buy&Hold)", format_kpi(kb["max_drawdown"]))

    curve = go.Figure()
    curve.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="Strategy Equity"))
    curve.add_trace(go.Scatter(x=bench.index, y=bench, mode="lines", name="Buy & Hold (norm)"))
    curve.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(curve, use_container_width=True)

st.caption("Objectives covered: auto-data, EDA, indicators, simple backtests, and risk/return KPIs.")
