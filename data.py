# data.py
from __future__ import annotations
import os
from typing import Iterable, List, Tuple, Dict, Optional
import datetime as dt

import pandas as pd

# Safe import for Streamlit Cloud:
try:
    import yfinance as yf
except Exception as e:
    raise ImportError(
        "yfinance is required. Add `yfinance==0.2.43` (or latest) to requirements.txt"
    ) from e


# ----------------------------
# Defaults & small helpers
# ----------------------------

_DEFAULT_INTERVAL = "1d"
_DEFAULT_PERIOD = "5y"   # used if start/end not given

def _to_datetime(x: Optional[str | dt.date | dt.datetime]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    return pd.to_datetime(x)

def _validate_symbols(symbols: Iterable[str]) -> List[str]:
    syms = [s.strip().upper() for s in symbols if str(s).strip()]
    if not syms:
        raise ValueError("No valid tickers provided.")
    return syms

def _coerce_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # ensure numeric OHLCV
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # drop rows that are fully NaN in prices
    price_cols = [c for c in ["Close", "Adj Close"] if c in df.columns]
    if price_cols:
        df = df.dropna(subset=price_cols, how="all")
    return df

def _add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col in df.columns:
        df["Return"] = df[price_col].pct_change()
        df["LogReturn"] = (1.0 + df["Return"]).apply(lambda x: pd.NA if pd.isna(x) or x<=0 else pd.Series(pd.np.log(x)))  # stays nullable
    return df


# ----------------------------
# Public API
# ----------------------------

def fetch_one(
    symbol: str,
    start: Optional[str | dt.date | dt.datetime] = None,
    end: Optional[str | dt.date | dt.datetime] = None,
    interval: str = _DEFAULT_INTERVAL,
    auto_adjust: bool = False,
    actions: bool = True,
) -> pd.DataFrame:
    """
    Fetch a single ticker from Yahoo Finance.
    Returns a DataFrame indexed by Datetime with OHLCV (+ Dividends/Splits if actions=True).
    """
    sym = symbol.strip().upper()
    s = _to_datetime(start)
    e = _to_datetime(end)

    if s is None and e is None:
        df = yf.download(sym, period=_DEFAULT_PERIOD, interval=interval, auto_adjust=auto_adjust, actions=actions, progress=False)
    else:
        df = yf.download(sym, start=s, end=e, interval=interval, auto_adjust=auto_adjust, actions=actions, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for {sym}. Check ticker and network access.")

    # yfinance returns multiindex columns for multi-ticker; single ticker should be flat
    if isinstance(df.columns, pd.MultiIndex):
        # flatten if needed
        df.columns = [" ".join([c for c in tup if c]).strip() for tup in df.columns]

    df = _coerce_ohlcv(df)
    df = df.sort_index()
    df = _add_basic_returns(df)
    df.index.name = "Date"
    return df


def fetch_many(
    symbols: Iterable[str],
    start: Optional[str | dt.date | dt.datetime] = None,
    end: Optional[str | dt.date | dt.datetime] = None,
    interval: str = _DEFAULT_INTERVAL,
    auto_adjust: bool = False,
    actions: bool = True,
    wide: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Fetch multiple tickers and return:
      - a long-format DataFrame with a 'Symbol' column (default), or
      - wide=False -> long; wide=True -> wide pivot of the 'Adj Close' (and Close fallback)
      - a dict of per-symbol DataFrames for convenience
    """
    syms = _validate_symbols(symbols)
    frames = {}
    for s in syms:
        df = fetch_one(
            s, start=start, end=end, interval=interval,
            auto_adjust=auto_adjust, actions=actions
        )
        df["Symbol"] = s
        frames[s] = df

    # Long frame
    long_df = (
        pd.concat(frames.values(), axis=0)
        .reset_index()
        .set_index("Date")
        .sort_index()
    )

    if wide:
        price_col = "Adj Close" if "Adj Close" in long_df.columns else "Close"
        wide_df = long_df.pivot_table(values=price_col, index=long_df.index, columns="Symbol")
        return wide_df, frames

    return long_df, frames


def fetch_index_constituents(index_name: str) -> List[str]:
    """
    Minimal helper to return a handful of default tickers per index if you don't
    have an index membership API. Extend as needed.
    """
    idx = index_name.strip().upper()
    if idx in {"S&P500", "SP500", "SPX"}:
        # small representative subset; replace with full set if you wire an API
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    if idx in {"NIFTY50", "NIFTY"}:
        return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    if idx in {"FTSE100", "FTSE"}:
        return ["ULVR.L", "HSBA.L", "BP.L", "VOD.L", "GSK.L"]
    # default fallback
    return ["AAPL", "MSFT", "GOOGL"]
