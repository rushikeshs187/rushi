# app.py
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Try Plotly; fall back to Streamlit charts if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from utils import (
    load_csv,
    fetch_ticker_history,
    add_technical_indicators,
    compute_kpis,
    to_returns,
)

st.set_page_config(
    page_title="IBR Finance Dashboard",
    page_icon="📊",
    layout="wide",
)

# =========================
# Sidebar controls
# =========================
st.sidebar.title("📁 Data source")
source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Yahoo Finance (yfinance)"])

uploaded = None
date_col = None
price_col = None
ticker = None
years = None

if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload a CSV with Date and Close (OHLC optional)", type=["csv"])
    date_col = st.sidebar.text_input("Date column name", value="Date")
    price_col = st.sidebar.text_input("Close/Price column name", value="Close")
else:
    ticker = st.sidebar.text_input("Ticker (e.g., AAPL, TSLA, ^GSPC, BTC-USD)", value="AAPL")
    years = st.sidebar.slider("Years of history", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.title("🧮 Indicators")
show_sma = st.sidebar.checkbox("SMA (20/50)", value=True)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
show_bbands = st.sidebar.checkbox("Bollinger Bands (20, 2σ)", value=True)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Chart options")
log_scale = st.sidebar.checkbox("Log scale", value=False)
show_volume = st.sidebar.checkbox("Show Volume", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Built for IBR Finance • Streamlit + Plotly")

# =========================
# Load data
# =========================
@st.cache_data(show_spinner=False)
def _load_data(source, uploaded, date_col, price_col, ticker, years):
    if source == "Upload CSV":
        if uploaded is None:
            return None, "Please upload a CSV to proceed."
        df, err = load_csv(uploaded, date_col=date_col)
        if err:
            return None, err
        # Standardize price column name to 'Close'
        if price_col not in df.columns:
            return None, f"Couldn't find price column '{price_col}' in the file."
        if "Close" not in df.columns:
            df = df.rename(columns={price_col: "Close"})
        return df, None
    else:
        df, err = fetch_ticker_history(ticker, years)
        return df, err

df, err = _load_data(
    source,
    uploaded if source == "Upload CSV" else None,
    date_col if source == "Upload CSV" else None,
    price_col if source == "Upload CSV" else None,
    ticker if source == "Yahoo Finance (yfinance)" else None,
    years if source == "Yahoo Finance (yfinance)" else None,
)

if err:
    st.warning(err)
    st.stop()
if df is None or df.empty:
    st.info("No data yet. Provide a CSV or a ticker to begin.")
    st.stop()

df = add_technical_indicators(df, sma=show_sma, rsi=show_rsi, bb=show_bbands)

# =========================
# Header & KPIs
# =========================
left, mid, right = st.columns([2, 3, 2], gap="large")

with left:
    st.title("📈 Finance Dashboard")
    src_label = "CSV Upload" if source == "Upload CSV" else f"Yahoo Finance • {ticker.upper()}"
    st.caption(f"Data source: {src_label}  |  Rows: {len(df):,}")

with mid:
    k = compute_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    last_str = f"{k['price_last']:.2f}" if np.isfinite(k['price_last']) else "—"
    c1.metric("Price (last)", last_str, f"{k['d1']:+.2f}% D1" if np.isfinite(k['d1']) else None)
    c2.metric("MTD", f"{k['mtd']:+.2f}%" if np.isfinite(k['mtd']) else "—")
    c3.metric("YTD", f"{k['ytd']:+.2f}%" if np.isfinite(k['ytd']) else "—")
    c4.metric("Sharpe (daily≈252)", f"{k['sharpe']:.2f}" if np.isfinite(k['sharpe']) else "—")

with right:
    st.write("")
    st.write("")
    st.caption("Tip: toggle indicators in the sidebar. Export below.")

# =========================
# Price Chart
# =========================
st.subheader("Price & Indicators")

def _plot_main(df):
    if _HAS_PLOTLY:
        fig = go.Figure()
        # Price
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"))
        # Indicators
        if show_sma and "SMA20" in df and "SMA50" in df:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", mode="lines"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", mode="lines"))
        if show_bbands and {"BB_M","BB_U","BB_L"}.issubset(df.columns):
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"], name="BB Upper", mode="lines"))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_M"], name="BB Mid", mode="lines"))
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"], name="BB Lower", mode="lines"))
        if log_scale:
            fig.update_yaxes(type="log")
        fig.update_layout(height=500, legend_title_text="Legend", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        cols = ["Close"] + [c for c in ["SMA20","SMA50","BB_U","BB_M","BB_L"] if c in df.columns]
        st.line_chart(df[cols])

_plot_main(df)

# =========================
# RSI (secondary)
# =========================
if show_rsi and "RSI" in df:
    st.subheader("RSI (14)")
    if _HAS_PLOTLY:
        rfig = px.line(df, x=df.index, y="RSI", title=None)
        rfig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="lightgray", opacity=0.2)
        rfig.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(rfig, use_container_width=True)
    else:
        st.line_chart(df[["RSI"]])

# =========================
# Returns & simple distribution
# =========================
st.subheader("Returns & Distribution")
ret_df = to_returns(df)
c1, c2 = st.columns(2)
with c1:
    st.caption("Daily returns (last 10)")
    st.dataframe(ret_df.tail(10))
with c2:
    if _HAS_PLOTLY:
        clean = ret_df.dropna()
        if not clean.empty:
            h = px.histogram(clean, x="Returns", nbins=50, title="Return distribution")
            h.update_layout(height=300, margin=dict(l=10,r=10,t=35,b=10), title_x=0.2)
            st.plotly_chart(h, use_container_width=True)
        else:
            st.info("Not enough returns to plot a histogram yet.")
    else:
        st.bar_chart(ret_df.tail(60)["Returns"])

st.markdown("---")
st.download_button(
    "⬇️ Download enriched dataset (CSV)",
    data=df.to_csv(index=True).encode("utf-8"),
    file_name="enriched_market_data.csv",
    mime="text/csv",
)

st.caption("If Plotly isn't installed, charts fall back to Streamlit's native charts.")
