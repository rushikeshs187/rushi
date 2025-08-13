# data.py
# Centralized data access & cleaning layer for the dashboard.
# - No file uploads required: pulls live/history from yfinance (+optional FRED)
# - Safe to import without Streamlit; adds Streamlit caching automatically if present
# - Returns numeric Series/DataFrames (float64) to avoid TypeErrors in features.py
# - Robust to empty results, missing cols, mixed dtypes, and timezone quirks

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional imports
try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    raise ImportError(
        "yfinance is required. Add `yfinance` to requirements.txt"
    ) from e

# Optional: Streamlit-aware cache (works even if Streamlit is not installed)
try:
    import streamlit as st

    def cache_data(**kwargs):
        return st.cache_data(show_spinner=False, **kwargs)
except Exception:  # pragma: no cover
    def cache_data(**kwargs):
        # No-op decorator if Streamlit isn't available
        def _wrap(fn):
            return fn
        return _wrap


# ------------------------------ Config & Constants ------------------------------ #

DUBAI_TZ = "Asia/Dubai"
UTC = timezone.utc

# Use ETFs or representative tickers so data is always available without API keys
DEFAULT_UNIVERSE = [
    # US large-cap reps
    "SPY", "QQQ", "DIA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    # India & EM proxies
    "^NSEI",          # NIFTY 50 index
    "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS",
    "EEM",            # iShares MSCI Emerging Markets
    # Crypto (via yfinance)
    "BTC-USD", "ETH-USD",
]

# Reasonable fallbacks if user hasn't chosen dates
DEFAULT_START = (datetime.now(tz=UTC) - timedelta(days=365 * 5)).date()  # last 5 years
DEFAULT_END = datetime.now(tz=UTC).date()
DEFAULT_INTERVAL = "1d"  # accepted by yfinance: "1d", "1h", "1wk", etc.

_RETRY_PAUSE = 1.5
_MAX_RETRIES = 3


# ------------------------------ Helpers ------------------------------ #

def _to_utc_index(idx: Union[pd.Index, pd.DatetimeIndex]) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, errors="coerce")
    # Normalize to UTC tz-aware DatetimeIndex
    if idx.tz is None:
        return idx.tz_localize(UTC)
    return idx.tz_convert(UTC)


def _coerce_numeric_series(
    s: Union[pd.Series, pd.DataFrame, np.ndarray, List[float]],
    name: str = "Close",
) -> pd.Series:
    """
    Returns a 1-D float64 Series. If given a DataFrame with a single column,
    squeeze it; if multiple columns, prefer ['Close', 'Adj Close'] by name.
    Avoids TypeError in downstream to_numeric by ensuring 1-D numeric.
    """
    if isinstance(s, pd.DataFrame):
        cols = [c for c in ["Close", "Adj Close", name] if c in s.columns]
        if len(cols) >= 1:
            s = s[cols[0]]
        else:
            if s.shape[1] == 1:
                s = s.iloc[:, 0]
            else:
                # take the first numeric-looking column as last resort
                num_cols = s.select_dtypes(include=[np.number]).columns
                s = s[num_cols[0]] if len(num_cols) else s.iloc[:, 0]

    s = pd.to_numeric(pd.Series(s), errors="coerce")
    s.name = name
    return s.astype("float64")


def _sanitize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure OHLCV numeric, proper DatetimeIndex in UTC, no all-NaN rows, forward-fill small gaps.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])

    # Fix index
    df = df.copy()
    df.index = _to_utc_index(df.index)

    # Standard column set
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in df.columns:
            # yfinance sometimes uses 'Adj Close' vs 'Adj Close' spacing; already handled
            df[col] = np.nan

    # Coerce numerics
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that are completely NaN on key fields
    key = ["Open", "High", "Low", "Close"]
    df = df.dropna(how="all", subset=key)

    # Light forward-fill to smooth tiny gaps, then backfill just once
    df[key] = df[key].ffill(limit=2).bfill(limit=1)

    # Keep columns in tidy order
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return df[cols]


def _retryable_download(
    tickers: Union[str, List[str]],
    start: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end: Optional[Union[str, datetime, pd.Timestamp]] = None,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    yfinance wrapper with light retry logic and consistent DataFrame shape.
    """
    last_err = None
    for _ in range(_MAX_RETRIES):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=True,
            )
            # If single ticker, yfinance returns a simple DF
            # If multiple, it's a column MultiIndex — we normalize later in fetch_multiple
            return df
        except Exception as e:
            last_err = e
            time.sleep(_RETRY_PAUSE)
    raise RuntimeError(f"Failed to download data for {tickers}: {last_err}")


# ------------------------------ Public API ------------------------------ #

@cache_data(ttl=60 * 30)  # 30 min
def fetch_prices(
    ticker: str,
    start: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end: Optional[Union[str, datetime, pd.Timestamp]] = None,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV for a single ticker from yfinance and return a clean DataFrame.
    Ensures numeric floats and UTC DatetimeIndex. Guaranteed to include 'Close' (float64).

    Examples:
        df = fetch_prices("AAPL", "2018-01-01", "2025-08-13")
        btc = fetch_prices("BTC-USD", start=None, end=None, interval="1d")
    """
    if start is None:
        start = DEFAULT_START
    if end is None:
        end = DEFAULT_END

    raw = _retryable_download(ticker, start, end, interval=interval, auto_adjust=auto_adjust)

    # yfinance sometimes returns a Series-like when interval is '1m' and short windows
    if isinstance(raw, pd.Series):
        raw = raw.to_frame("Close")

    # If yfinance returned a column MultiIndex for a single ticker (can happen), drop the top level
    if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.levels[1].astype(str):
        raw = raw.xs(ticker, axis=1, level=1)

    df = _sanitize_price_df(raw)

    # Strong guarantee for downstream features: 'Close' is float Series
    df["Close"] = _coerce_numeric_series(df["Close"], name="Close")
    return df


@cache_data(ttl=60 * 30)
def fetch_multiple(
    tickers: Iterable[str],
    start: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end: Optional[Union[str, datetime, pd.Timestamp]] = None,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Batch download. Returns:
      - dict[ticker] -> clean OHLCV DataFrame
      - merged_close: wide DataFrame of Close prices (float64) aligned on UTC index
    """
    tickers = list(dict.fromkeys([t.strip() for t in tickers if t and isinstance(t, str)]))
    if not tickers:
        return {}, pd.DataFrame()

    if start is None:
        start = DEFAULT_START
    if end is None:
        end = DEFAULT_END

    raw = _retryable_download(tickers, start, end, interval=interval, auto_adjust=auto_adjust)

    frames: Dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        # MultiIndex: (Field, Ticker)
        for t in tickers:
            try:
                sub = raw.xs(t, axis=1, level=1)
                frames[t] = _sanitize_price_df(sub)
                frames[t]["Close"] = _coerce_numeric_series(frames[t]["Close"], name="Close")
            except Exception:
                frames[t] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    else:
        # Single DF. Try to split by ticker if columns carry tickers; else treat as single.
        t = tickers[0]
        frames[t] = _sanitize_price_df(raw)
        frames[t]["Close"] = _coerce_numeric_series(frames[t]["Close"], name="Close")

    # Merge Close columns
    merged = []
    for t, df in frames.items():
        if "Close" in df.columns and not df["Close"].dropna().empty:
            s = df["Close"].copy()
            s.name = t
            merged.append(s)

    merged_close = pd.concat(merged, axis=1).sort_index() if merged else pd.DataFrame()
    # Ensure float64
    merged_close = merged_close.apply(pd.to_numeric, errors="coerce").astype("float64")
    merged_close.index = _to_utc_index(merged_close.index)

    return frames, merged_close


@cache_data(ttl=60 * 60)  # 1 hour
def fetch_fundamentals(ticker: str) -> Dict[str, Union[pd.DataFrame, pd.Series, dict]]:
    """
    Light fundamentals via yfinance. Returns dict with safe keys:
      - 'fast_info': dict    (prices, market cap, etc.)
      - 'info': dict         (may be large; rely on fast_info primarily)
      - 'balance_sheet': DataFrame
      - 'income_stmt': DataFrame
      - 'cashflow': DataFrame
      - 'sustainability': DataFrame or None
    """
    t = yf.Ticker(ticker)
    out: Dict[str, Union[pd.DataFrame, pd.Series, dict]] = {}

    # Try attributes defensively; yfinance can vary by symbol
    try:
        out["fast_info"] = dict(getattr(t, "fast_info", {}))
    except Exception:
        out["fast_info"] = {}

    try:
        info = getattr(t, "info", {})
        # info can be a yfinance TickerInfo proxy; make it plain dict
        out["info"] = dict(info) if isinstance(info, dict) else {}
    except Exception:
        out["info"] = {}

    def _clean_fin(df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        d = df.copy()
        # Financials often have columns as periods; ensure numeric and sorted
        for c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return d

    try:
        out["balance_sheet"] = _clean_fin(t.balance_sheet)
    except Exception:
        out["balance_sheet"] = pd.DataFrame()

    try:
        out["income_stmt"] = _clean_fin(t.income_stmt)
    except Exception:
        out["income_stmt"] = pd.DataFrame()

    try:
        out["cashflow"] = _clean_fin(t.cashflow)
    except Exception:
        out["cashflow"] = pd.DataFrame()

    try:
        sust = t.sustainability
        out["sustainability"] = sust if isinstance(sust, pd.DataFrame) else pd.DataFrame()
    except Exception:
        out["sustainability"] = pd.DataFrame()

    return out


@cache_data(ttl=60 * 60)
def fetch_macro_fred(
    series: Iterable[str],
    start: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end: Optional[Union[str, datetime, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    OPTIONAL macro pull from FRED via pandas_datareader (no API key required for public).
    If pandas_datareader isn't available (or blocked), returns empty DataFrame gracefully.
    """
    try:
        from pandas_datareader import data as pdr  # lazy import
    except Exception:
        return pd.DataFrame()

    series = list(dict.fromkeys([s.strip() for s in series if s and isinstance(s, str)]))
    if not series:
        return pd.DataFrame()

    if start is None:
        start = "2000-01-01"
    if end is None:
        end = DEFAULT_END

    frames = []
    for s in series:
        try:
            df = pdr.DataReader(s, "fred", start=start, end=end)
            if not df.empty:
                df.index = _to_utc_index(df.index)
                df.columns = [s]
                frames.append(df)
        except Exception:
            # continue on individual failures
            pass

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    # Coerce to float
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.astype("float64")


# ------------------------------ Convenience / Presets ------------------------------ #

def default_universe() -> List[str]:
    return DEFAULT_UNIVERSE.copy()


@cache_data(ttl=60 * 10)
def quick_price_panel(
    tickers: Optional[Iterable[str]] = None,
    start: Optional[Union[str, datetime, pd.Timestamp]] = None,
    end: Optional[Union[str, datetime, pd.Timestamp]] = None,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Quickly get a wide Close-price panel (float64) plus per-ticker OHLCV dict.
    Designed for EDA pages and KPI computations.
    """
    if not tickers:
        tickers = default_universe()
    frames, close_wide = fetch_multiple(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    # Drop columns that are entirely NaN and lightly forward-fill
    close_wide = close_wide.dropna(how="all").ffill(limit=2).bfill(limit=1)
    return close_wide, frames


def get_close_series(
    df_or_frames: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ticker: Optional[str] = None,
) -> pd.Series:
    """
    Uniformly return a single clean Close series (float64) for downstream indicators.

    - If given a dict[str->DF], you must specify ticker.
    - If given a DF with a 'Close' column, that column is returned.
    - If given a wide DF of multiple tickers, you must specify ticker (column).
    """
    if isinstance(df_or_frames, dict):
        if not ticker or ticker not in df_or_frames:
            raise ValueError("Provide a valid `ticker` when passing a dict of frames.")
        df = df_or_frames[ticker]
        s = _coerce_numeric_series(df.get("Close", pd.Series(dtype=float)), name="Close")
        s.index = _to_utc_index(s.index)
        return s

    df = df_or_frames
    if "Close" in df.columns:
        s = _coerce_numeric_series(df["Close"], name="Close")
        s.index = _to_utc_index(s.index)
        return s

    if ticker and ticker in df.columns:
        s = _coerce_numeric_series(df[ticker], name="Close")
        s.index = _to_utc_index(s.index)
        return s

    raise ValueError("Could not resolve a single Close series; specify `ticker` or pass a frame with 'Close'.")


# ------------------------------ Minimal Inline Tests (optional) ------------------------------ #

if __name__ == "__main__":  # quick sanity check when running locally
    panel, frames = quick_price_panel(["AAPL", "MSFT", "BTC-USD"], start="2020-01-01")
    assert "AAPL" in frames and "MSFT" in frames
    s = get_close_series(frames, "AAPL")
    assert s.dtype == "float64" and s.index.tz is not None
    print("data.py sanity checks passed:", panel.shape, len(frames))
