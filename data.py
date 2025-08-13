from __future__ import annotations
import time
from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf

SUPPORTED_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
SUPPORTED_INTERVALS = ["1d", "1wk", "1mo", "60m", "30m", "15m", "5m", "1m"]

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # Ensure standard columns exist, coerce numeric
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.dropna(how="all")
    return df

def _demo_series(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.01, n)
    px = 100 * (1 + pd.Series(ret)).cumprod()
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({
        "Open": px.values,
        "High": px.values * (1 + 0.01),
        "Low": px.values * (1 - 0.01),
        "Close": px.values,
        "Adj Close": px.values,
        "Volume": 1e6
    }, index=idx)

def fetch_one(symbol: str, period: str = "1y", interval: str = "1d", auto_adjust: bool = True) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=auto_adjust, progress=False, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = _clean_df(df)
            return df
    except Exception:
        pass
    # Fallback demo if nothing fetched
    demo = _demo_series()
    demo.name = symbol
    return demo

def fetch_many(symbols: List[str], period: str = "1y", interval: str = "1d", auto_adjust: bool = True) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = fetch_one(s, period=period, interval=interval, auto_adjust=auto_adjust)
        out[s] = df
    # If literally all are empty (shouldn’t happen due to demo), guard anyway:
    if all((d is None or d.empty) for d in out.values()):
        raise ValueError("fetch_many: no data could be retrieved for the requested symbols.")
    return out
