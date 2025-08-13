import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data import fetch_many, allowed_periods, allowed_intervals, normalize_close
from features import build_indicators
from utils import (
    compute_kpis, humanize_kpis, rolling_stats, compute_drawdown,
    ensure_tz_safe_today, safe_pct_change
)

# ---------- Streamlit Page Setup ----------
st.set_page_config(
    page_title="Finance EDA & Indicators Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Finance EDA & Indicators Dashboard")
st.caption("Auto-fetching data • No uploads required • EDA + Indicators + Portfolio KPIs")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("⚙️ Controls")

    default_symbols = os.environ.get("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,NVDA")
    symbols = st.text_input("Symbols (comma-separated)", default_symbols).upper()
    symbols = [s.strip() for s in symbols.split(",") if s.strip()]

    period = st.selectbox("History Period", allowed_periods(), index=6)   # "5y"
    interval = st.selectbox("Data Interval", allowed_intervals(), index=4)  # "1d"

    st.divider()
    st.subheader("Indicators")
    rsi_win = st.slider("RSI window", 5, 50, 14, 1)
    ema_fast = st.slider("EMA Fast", 5, 50, 12, 1)
    ema_slow = st.slider("EMA Slow", 10, 200, 26, 1)
    macd_fast = st.slider("MACD Fast", 5, 50, 12, 1)
    macd_slow = st.slider("MACD Slow", 10, 200, 26, 1)
    macd_sig  = st.slider("MACD Signal", 3, 30, 9, 1)
    bb_window = st.slider("Bollinger Window", 10, 60, 20, 1)
    bb_stds   = st.slider("Bollinger Std Dev", 1.0, 4.0, 2.0, 0.5)

    st.divider()
    st.subheader("Benchmark (optional)")
    bench = st.text_input("Benchmark symbol", "SPY").strip().upper()

# ---------- Fetch Data ----------
today = ensure_tz_safe_today()

try:
    # returns Dict[str, DataFrame]
    panel = fetch_many(symbols, period=period, interval=interval)
except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()

if not panel:
    st.warning("No data could be retrieved for the requested symbols. Try changing period/interval or symbols.")
    st.stop()

tabs = st.tabs(["Overview", "Per-Symbol EDA", "Indicators", "Portfolio KPIs"])

# ---------- Overview Tab ----------
with tabs[0]:
    st.subheader("Summary Snapshot")

    rows = []
    for sym, df in panel.items():
        if df.empty or "Adj Close" not in df.columns:
            continue
        last_px = float(df["Adj Close"].dropna().iloc[-1]) if df["Adj Close"].dropna().size else np.nan

        # YTD & 1Y
        ytd = np.nan
        one_year = np.nan
        try:
            if (df.index.year == today.year).any():
                first_this_year = df["Adj Close"][df.index.year == today.year].iloc[0]
                if pd.notna(first_this_year) and first_this_year != 0:
                    ytd = df["Adj Close"].iloc[-1] / first_this_year - 1
        except Exception:
            pass
        if len(df) > 252 and pd.notna(df["Adj Close"].iloc[-252]) and df["Adj Close"].iloc[-252] != 0:
            one_year = float(df["Adj Close"].iloc[-1] / df["Adj Close"].iloc[-252] - 1)
        rows.append({"Symbol": sym, "Last Price": last_px, "YTD": ytd, "1Y": one_year})

    if rows:
        snap = pd.DataFrame(rows).set_index("Symbol")
        st.dataframe(
            snap.style.format({"Last Price": "{:,.2f}", "YTD": "{:.2%}", "1Y": "{:.2%}"}),
            use_container_width=True
        )
    else:
        st.info("No summary available.")

    st.divider()
    st.subheader("Price Comparison")
    norm_df = normalize_close(panel)
    if not norm_df.empty:
        fig = px.line(norm_df, x=norm_df.index, y=norm_df.columns, title="Normalized Adj Close (rebased to 100)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to chart.")

# ---------- Per-Symbol EDA Tab ----------
with tabs[1]:
    st.subheader("Explore a Symbol")
    sym = st.selectbox("Pick symbol", list(panel.keys()))
    df = panel.get(sym, pd.DataFrame()).copy()

    if df.empty or "Adj Close" not in df.columns:
        st.warning("Selected symbol has no data.")
    else:
        # NOTE: avoid vertical_alignment kw (older Streamlit)
        colA, colB = st.columns([2, 1])

        with colA:
            pfig = go.Figure()
            pfig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="OHLC"
            ))
            pfig.update_layout(title=f"{sym} OHLC", xaxis_rangeslider_visible=False)
            st.plotly_chart(pfig, use_container_width=True)

            st.caption("Rolling Stats on Adj Close")
            rs = rolling_stats(df["Adj Close"], windows=[20, 50, 100, 200])
            if not rs.empty:
                rfig = px.line(rs, title="Rolling Mean/Volatility")
                st.plotly_chart(rfig, use_container_width=True)

        with colB:
            st.markdown("")  # small spacer to pin top on older Streamlit
            st.write("Quick Stats")
            dd = compute_drawdown(df["Adj Close"])
            metrics = {
                "Last": df["Adj Close"].iloc[-1],
                "Max Drawdown": dd["drawdown"].min(),
                "Vol (20d)": df["Adj Close"].pct_change().rolling(20).std().iloc[-1],
                "Vol (60d)": df["Adj Close"].pct_change().rolling(60).std().iloc[-1],
            }
            sm = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
            st.dataframe(sm.style.format("{:,.4f}"))

        st.divider()
        st.write("Return Distribution (daily)")
        rets = df["Adj Close"].pct_change().dropna()
        if len(rets):
            hfig = px.histogram(rets, nbins=50, title=f"{sym} Daily Returns")
            st.plotly_chart(hfig, use_container_width=True)
        else:
            st.info("Not enough daily returns to plot.")

# ---------- Indicators Tab ----------
with tabs[2]:
    st.subheader("Technical Indicators")
    sym2 = st.selectbox("Pick symbol for indicators", list(panel.keys()), key="ind_sym")
    df2 = panel.get(sym2, pd.DataFrame()).copy()

    if df2.empty or "Adj Close" not in df2.columns:
        st.warning("Selected symbol has no data.")
    else:
        ind = build_indicators(
            df2,
            rsi_window=rsi_win,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_sig,
            bb_window=bb_window,
            bb_stds=bb_stds
        )

        # Price + EMAs + Bollinger
        p = go.Figure()
        p.add_trace(go.Scatter(x=ind.index, y=ind["Adj Close"], name="Adj Close", mode="lines"))
        for c in ["EMA_Fast", "EMA_Slow"]:
            if c in ind:
                p.add_trace(go.Scatter(x=ind.index, y=ind[c], name=c, mode="lines"))
        if {"BB_Upper", "BB_Lower"}.issubset(ind.columns):
            p.add_trace(go.Scatter(x=ind.index, y=ind["BB_Upper"], name="BB Upper", mode="lines"))
            p.add_trace(go.Scatter(x=ind.index, y=ind["BB_Lower"], name="BB Lower", mode="lines"))
        p.update_layout(title=f"{sym2} Price + EMAs + Bollinger")
        st.plotly_chart(p, use_container_width=True)

        # RSI
        if "RSI" in ind.columns:
            rfig = px.line(ind, x=ind.index, y="RSI", title="RSI")
            rfig.add_hline(y=70, line_dash="dash")
            rfig.add_hline(y=30, line_dash="dash")
            st.plotly_chart(rfig, use_container_width=True)

        # MACD
        if {"MACD", "MACD_Signal"}.issubset(ind.columns):
            m = go.Figure()
            m.add_trace(go.Scatter(x=ind.index, y=ind["MACD"], name="MACD", mode="lines"))
            m.add_trace(go.Scatter(x=ind.index, y=ind["MACD_Signal"], name="Signal", mode="lines"))
            m.update_layout(title="MACD")
            st.plotly_chart(m, use_container_width=True)

        st.caption("Indicator table (last 10 rows)")
        st.dataframe(ind.tail(10), use_container_width=True)

# ---------- Portfolio KPIs Tab ----------
with tabs[3]:
    st.subheader("Portfolio KPIs")

    aligned = []
    for sym, df in panel.items():
        if "Adj Close" in df.columns:
            r = df["Adj Close"].pct_change().rename(sym)
            aligned.append(r)
    if not aligned:
        st.warning("No returns available.")
        st.stop()

    returns = pd.concat(aligned, axis=1).dropna(how="all")
    weights = np.ones(len(returns.columns)) / len(returns.columns)
    port_ret = returns.fillna(0).dot(weights)

    bench_ret = None
    if bench:
        try:
            bpanel = fetch_many([bench], period=period, interval=interval)
            bdf = bpanel.get(bench, pd.DataFrame())
            if not bdf.empty and "Adj Close" in bdf.columns:
                bench_ret = bdf["Adj Close"].pct_change()
        except Exception:
            pass

    k = compute_kpis(port_ret, benchmark=bench_ret)
    st.markdown(humanize_kpis(k))

    # Cumulative curves
    cum = (1 + port_ret.fillna(0)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum.index, y=cum, name="Portfolio", mode="lines"))
    if bench_ret is not None and len(bench_ret.dropna()):
        bcum = (1 + bench_ret.fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=bcum.index, y=bcum, name=bench, mode="lines"))
    fig.update_layout(title="Cumulative Growth (rebased to 1)")
    st.plotly_chart(fig, use_container_width=True)

st.info("Tip: tweak windows & intervals in the sidebar to see how indicators and KPIs react.")
