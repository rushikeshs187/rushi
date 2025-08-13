import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def basic_profile(df):
    # Summary table
    desc = df.describe().T.reset_index().rename(columns={"index": "feature"})
    nulls = df.isna().sum().rename("missing").reset_index().rename(columns={"index":"feature"})
    out = desc.merge(nulls, on="feature", how="left")
    return out

def plot_price_with_bands(df, price_col="Adj Close"):
    cols = [c for c in [price_col, "BB_mid", "BB_upper", "BB_lower"] if c in df.columns]
    if price_col not in df.columns:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[price_col], name=price_col, mode="lines"))
    if "BB_mid" in cols: fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB mid", mode="lines"))
    if "BB_upper" in cols: fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB upper", mode="lines"))
    if "BB_lower" in cols: fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB lower", mode="lines"))
    fig.update_layout(margin=dict(l=10,r=10,b=10,t=10))
    return fig

def plot_rsi(df):
    if "RSI" not in df.columns:
        return go.Figure()
    fig = px.line(df.reset_index(), x=df.index.name or "index", y="RSI", title=None)
    fig.add_hline(y=70, line_dash="dot", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dot", annotation_text="Oversold")
    fig.update_layout(margin=dict(l=10,r=10,b=10,t=10))
    return fig

def plot_returns_hist(ret, bins=50):
    s = pd.to_numeric(ret, errors="coerce").dropna()
    if s.empty:
        return go.Figure()
    fig = px.histogram(s, nbins=bins, opacity=0.9)
    fig.update_layout(margin=dict(l=10,r=10,b=10,t=10), bargap=0.05)
    return fig

def plot_corr(df, cols=None):
    use = df[cols].select_dtypes(include=[np.number]) if cols else df.select_dtypes(include=[np.number])
    if use.shape[1] < 2:
        return go.Figure()
    corr = use.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    fig.update_layout(margin=dict(l=10,r=10,b=10,t=10))
    return fig
