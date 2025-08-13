# ui.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def kpi_row(k: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{k['CAGR']:.2%}" if np.isfinite(k['CAGR']) else "—")
    c2.metric("Sharpe", f"{k['Sharpe']:.2f}" if np.isfinite(k['Sharpe']) else "—")
    c3.metric("Sortino", f"{k['Sortino']:.2f}" if np.isfinite(k['Sortino']) else "—")
    c4.metric("Max Drawdown", f"{k['MaxDD']:.2%}" if np.isfinite(k['MaxDD']) else "—")
    c5.metric("Hit Rate", f"{k['HitRate']:.2%}" if np.isfinite(k['HitRate']) else "—")

def price_with_bands(df: pd.DataFrame):
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    use = df[[price_col, "BB_Upper", "BB_Mid", "BB_Lower"]].dropna(how="all")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=use.index, y=use[price_col], name=price_col, mode="lines"))
    for col, dash in [("BB_Upper", "dot"), ("BB_Mid", "dash"), ("BB_Lower", "dot")]:
        if col in use.columns and not use[col].isna().all():
            fig.add_trace(go.Scatter(x=use.index, y=use[col], name=col, mode="lines", line=dict(dash=dash)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def rsi_chart(df: pd.DataFrame):
    if "RSI" not in df.columns:
        return go.Figure()
    tidy = pd.DataFrame({"Date": df.index, "RSI": pd.to_numeric(df["RSI"], errors="coerce")}).dropna()
    fig = px.line(tidy, x="Date", y="RSI", title=None)
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def macd_chart(df: pd.DataFrame):
    cols = ["MACD", "MACD_Signal", "MACD_Hist"]
    if not all(c in df.columns for c in cols):
        return go.Figure()
    tidy = df[cols].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tidy.index, y=tidy["MACD"], name="MACD", mode="lines"))
    fig.add_trace(go.Scatter(x=tidy.index, y=tidy["MACD_Signal"], name="Signal", mode="lines"))
    fig.add_trace(go.Bar(x=tidy.index, y=tidy["MACD_Hist"], name="Hist"))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def returns_hist(df: pd.DataFrame, bins=50):
    if "Return" not in df.columns:
        return go.Figure()
    s = pd.to_numeric(df["Return"], errors="coerce").dropna()
    if s.empty:
        return go.Figure()
    fig = px.histogram(s, nbins=bins, title="Return Distribution")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title_x=0.3)
    return fig

def corr_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return go.Figure()
    corr = num.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def model_table(results_df: pd.DataFrame):
    # Render a small bar+table hybrid: we’ll just return a heatmap-like table as a figure
    df = results_df.copy()
    df = df[["Model","Val_AUC","Val_F1","Test_AUC","Test_F1","Test_Acc"]]
    df = df.round(3)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color="#2a3f5f", font=dict(color="white"), align="left"),
        cells=dict(values=[df[c] for c in df.columns], fill_color="#f7f7f7", align="left")
    )])
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def equity_curve_chart(equity: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Date", yaxis_title="Equity (× start)")
    return fig

def info_note(msg: str):
    st.info(msg)
