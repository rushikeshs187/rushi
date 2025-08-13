from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import Dict, List

# Allowed selections (kept small to avoid invalid combos on Streamlit Cloud)
def allowed_periods() -> List[str]:
    return ["1mo","3mo","6mo","1y","2y","5y","10y","max"]

def allowed_intervals() -> List[str]:
    # intraday intervals may be restricted by period; we default to daily-safe set
    return ["1h","2h","4h","1d","1wk","1mo"]

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # standard columns if present
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df

def fetch_one(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
        # For single symbols, yfinance may return multiindex columns; flatten if so
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]
        return _clean(df)
    except Exception:
        return pd.DataFrame()

def fetch_many(symbols: List[str], period: str = "5y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    if not isinstance(symbols, list) or not symbols:
        raise TypeError("fetch_many expects a non-empty list of symbols.")
    out: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        s = s.strip().upper()
        if not s:
            continue
        df = fetch_one(s, period, interval)
        # Some tickers only return "Close" without "Adj Close" for certain combos; backfill
        if not df.empty and "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        if not df.empty:
            out[s] = df
    if not out:
        raise ValueError("fetch_many: no data could be retrieved for the requested symbols.")
    return out

def normalize_close(panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for s, df in panel.items():
        if "Adj Close" in df.columns and df["Adj Close"].notna().sum() > 0:
            base = df["Adj Close"].dropna().iloc[0]
            if base and base != 0:
                series = df["Adj Close"] / base * 100.0
                series = series.rename(s)
                frames.append(series)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()
