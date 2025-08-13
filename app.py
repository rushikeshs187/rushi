import io
import pandas as pd
import streamlit as st
import plotly.express as px
from utils import prepare_timeseries, compute_returns, compute_kpis
from eda import basic_profile, plot_price_with_bands, plot_rsi, plot_returns_hist, plot_corr

st.set_page_config(page_title="IBR Finance: ML-in-Finance EDA", layout="wide")

st.title("📊 IBR Finance — Markets EDA Dashboard")
st.caption("Focus: real-world effect (KPIs), model compatibility signals (indicators), and EM/DV markets readiness.")

with st.sidebar:
    st.header("Data")
    src = st.radio("Choose data source", ["Upload CSV"], horizontal=True)
    date_col = st.text_input("Date column name (optional)", value="")
    price_col = st.text_input("Price column name", value="Adj Close")
    rsi_window = st.number_input("RSI window", 5, 50, 14, 1)
    periods_per_year = st.selectbox("Periods/year", [252, 365, 52, 12], index=0)
    st.markdown("---")
    st.header("About")
    st.markdown("Objectives: 1) Real-world KPIs, 2) Model-readiness indicators, 3) Emerging-vs-Developed comparability.")

@st.cache_data(show_spinner=False)
def _read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

df = None
uploaded = st.file_uploader("Upload a CSV with price data", type=["csv"])
if uploaded is not None:
    try:
        raw = _read_csv(uploaded.getvalue())
        df = prepare_timeseries(raw, date_col=date_col or None, price_col=price_col)
        # Recompute RSI with chosen window if available price:
        if price_col in df.columns:
            from utils import rsi as _rsi
            df["RSI"] = _rsi(df[price_col], window=rsi_window)
    except Exception as e:
        st.error(f"Could not parse file: {e}")

if df is None:
    st.info("Upload a CSV to start. Expected columns: a date column + price (Adj Close or Close).")
    st.stop()

# ---------- KPIs (Objective 1) ----------
ret, px_series = compute_returns(df, price_col=price_col)
k = compute_kpis(df, price_col=price_col, risk_free_annual=0.0, periods_per_year=periods_per_year)

kpi_cols = st.columns(6)
kpi_cols[0].metric("Sharpe", f"{k['Sharpe']:.2f}" if pd.notna(k['Sharpe']) else "NA")
kpi_cols[1].metric("CAGR", f"{k['CAGR']*100:.2f}%" if pd.notna(k['CAGR']) else "NA")
kpi_cols[2].metric("Max Drawdown", f"{k['Max Drawdown']*100:.2f}%" if pd.notna(k['Max Drawdown']) else "NA")
kpi_cols[3].metric("Hit Rate", f"{k['Hit Rate']*100:.2f}%" if pd.notna(k['Hit Rate']) else "NA")
kpi_cols[4].metric("Ann Return", f"{k['Ann Return']*100:.2f}%" if pd.notna(k['Ann Return']) else "NA")
kpi_cols[5].metric("Ann Vol", f"{k['Ann Vol']*100:.2f}%" if pd.notna(k['Ann Vol']) else "NA")

st.markdown("### Price & Bands (Model‑readiness signals)")
st.plotly_chart(plot_price_with_bands(df, price_col=price_col), use_container_width=True)

c1, c2 = st.columns([1,1])
with c1:
    st.markdown("### RSI")
    st.plotly_chart(plot_rsi(df), use_container_width=True)
with c2:
    st.markdown("### Returns distribution")
    st.plotly_chart(plot_returns_hist(ret), use_container_width=True)

# ---------- EDA Summary (Objective 2 & 3 context) ----------
st.markdown("### EDA Overview")
prof = basic_profile(df.select_dtypes(include="number"))
st.dataframe(prof, height=320, use_container_width=True)

st.markdown("### Correlation heatmap (numeric features)")
st.plotly_chart(plot_corr(df), use_container_width=True)

# ---------- Notes aligned to objectives ----------
with st.expander("How this aligns to your objectives"):
    st.markdown(
        """
- **Real‑world effect:** KPIs (Sharpe, CAGR, MDD, Hit‑Rate) are computed on your uploaded data with robust guards (no ambiguous truth comparisons).
- **Model compatibility:** Indicators (RSI, SMA/EMA, Bollinger) + correlation give quick signal quality checks before RF/SVM/ANN.
- **Emerging vs Developed:** Upload multiple assets/markets; compare KPI panels and distributions to discuss regional factors.
        """
    )
