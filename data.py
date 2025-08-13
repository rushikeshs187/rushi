# data.py
# -----------------------------------------------------------------------------
# Centralized data access layer for the dashboard.
# - Pulls OHLCV from Yahoo Finance via yfinance
# - Optional Alpha Vantage fallback (if ALPHAVANTAGE_API_KEY is set)
# - Clean, typed, time-indexed DataFrames
# - Batch fetching (fetch_many) returns a MultiIndex-column DataFrame:
#       top level = symbol, lower level = ["Open","High","Low","Close","Adj Close","Volume","Return"]
# - Helper utilities to safely extract a numeric price Series for indicators
# - No Streamlit dependency here (pure Python). Caching uses functools.lru_cache
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import math
import time
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Dict

import pandas as pd

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    raise ImportError(
        "yfinance is required. Add 'yfinance' to your requirements.txt."
    ) from e

# ------------------------------ Configuration --------------------------------

_DEFAULT_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
_YF_MAX_RETRIES = 3
_YF_RETRY_SLEEP = 1.2  # seconds between retries


# ------------------------------ Utilities ------------------------------------

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric (where applicable) and sort by index."""
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Ensure DateTimeIndex named 'Date'
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=False)
    df = df[~df.index.isna()].sort_index()
    df.index.name = "Date"
    return df


def _add_returns(df: pd.DataFrame, col: str = "Adj Close") -> pd.DataFrame:
    """Add simple daily return based on selected price column; falls back to Close."""
    base = col if col in df.columns else ("Close" if "Close" in df.columns else None)
    if base is None:
        df["Return"] = pd.Series(dtype=float)
        return df
    ret = df[base].pct_change(fill_method=None)
    df["Return"] = ret
    return df


def pick_price(df: pd.DataFrame, prefer: str = "Adj Close") -> pd.Series:
    """
    Return a 1-D numeric Series of prices from a standard OHLCV frame.
    Avoids the 'arg must be Series' TypeError (common when passing a full DF).

    Order: prefer -> Close -> first numeric column.

    Example:
        px = pick_price(df)  # safe Series for indicators
    """
    candidates = [prefer, "Close", "Adj Close", "Open"]
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s.name = c
            return s
    # Fallback: first numeric column
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            s.name = c
            return s
    # Empty fallback
    return pd.Series(dtype=float, name="price")


def _yf_download_single(
    symbol: str,
    period: Optional[str] = None,
    interval: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = False,
    back_adjust: bool = False,
    threads: bool = True,
) -> pd.DataFrame:
    """
    Download a single symbol via yfinance with retries, returning clean OHLCV (+Return).
    """
    last_err = None
    for attempt in range(1, _YF_MAX_RETRIES + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                back_adjust=back_adjust,
                progress=False,
                threads=threads,
            )
            # yfinance returns an empty frame on unknown symbol/interval mismatch
            if df is None or df.empty:
                raise ValueError(f"No data for {symbol} (attempt {attempt}).")
            # If a single symbol, yfinance returns flat columns
            if isinstance(df.columns, pd.MultiIndex):
                # Some combos can return MultiIndex even for one symbol; drop the first level
                df.columns = df.columns.get_level_values(-1)
            # Keep standard fields if present
            keep = [c for c in _DEFAULT_FIELDS if c in df.columns]
            if keep:
                df = df[keep]
            df = _coerce_numeric(df)
            df = _add_returns(df, col="Adj Close")
            return df
        except Exception as e:
            last_err = e
            if attempt < _YF_MAX_RETRIES:
                time.sleep(_YF_RETRY_SLEEP * attempt)
            else:
                # On final failure bubble up
                raise
    # Should not reach here
    raise last_err  # pragma: no cover


def _alpha_vantage_fallback(
    symbol: str,
    outputsize: str = "compact",
) -> Optional[pd.DataFrame]:
    """
    Optional: very light fallback using Alpha Vantage (needs env ALPHAVANTAGE_API_KEY).
    We purposely avoid adding an extra dependency. If key is absent, return None.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return None

    import urllib.request
    import json

    url = (
        "https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}"
        f"&outputsize={outputsize}&apikey={api_key}"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        ts = payload.get("Time Series (Daily)", {})
        if not ts:
            return None
        records = []
        for d, vals in ts.items():
            records.append(
                {
                    "Date": d,
                    "Open": vals.get("1. open"),
                    "High": vals.get("2. high"),
                    "Low": vals.get("3. low"),
                    "Close": vals.get("4. close"),
                    "Adj Close": vals.get("5. adjusted close"),
                    "Volume": vals.get("6. volume"),
                }
            )
        df = pd.DataFrame.from_records(records).set_index("Date")
        df = _coerce_numeric(df)
        df = _add_returns(df, col="Adj Close")
        return df
    except Exception:
        return None


# ------------------------------ Public API -----------------------------------

@lru_cache(maxsize=256)
def fetch_one(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = False,
    back_adjust: bool = False,
) -> pd.DataFrame:
    """
    Fetch a single symbol as a clean OHLCV DataFrame with DateTimeIndex and a 'Return' column.

    Parameters
    ----------
    symbol : str
    period : str        e.g., '1mo','3mo','6mo','1y','2y','5y','10y','max'
    interval : str      e.g., '1d','1wk','1mo','1h','30m','15m','5m','1m' (intraday needs shorter period)
    start, end : str    optional ISO dates. If provided, they override 'period'
    auto_adjust : bool  if True, yfinance returns adjusted prices (no separate Adj Close)
    back_adjust : bool  back-adjust splits/dividends

    Returns
    -------
    pd.DataFrame with columns subset of: Open, High, Low, Close, Adj Close, Volume, Return
    """
    # Prefer explicit start/end if given
    if start or end:
        df = _yf_download_single(
            symbol=symbol,
            period=None,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
            back_adjust=back_adjust,
        )
    else:
        # Normal path: period+interval
        try:
            df = _yf_download_single(
                symbol=symbol,
                period=period,
                interval=interval,
                start=None,
                end=None,
                auto_adjust=auto_adjust,
                back_adjust=back_adjust,
            )
        except Exception:
            # Fallback to AV if available
            av = _alpha_vantage_fallback(symbol)
            if av is not None and not av.empty:
                df = av
            else:
                raise
    # Drop rows that are entirely NA (can happen on some intervals)
    df = df.dropna(how="all")
    return df


def _concat_as_panel(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine dict[symbol -> DataFrame] into one DataFrame with MultiIndex columns:
    (symbol, field). Aligns on the union of dates.
    """
    reindexed = {}
    for sym, df in frames.items():
        # Ensure columns cover a stable set for consistency
        cols = [c for c in (_DEFAULT_FIELDS + ["Return"]) if c in df.columns]
        if not cols:
            cols = list(df.columns)
        reindexed[sym] = df[cols].copy()
        reindexed[sym].columns = pd.MultiIndex.from_product([[sym], reindexed[sym].columns])
    if not reindexed:
        return pd.DataFrame()
    panel = pd.concat(reindexed.values(), axis=1).sort_index()
    # Ensure deterministic column order: by symbol then by field in _DEFAULT_FIELDS order
    def _sort_key(col_pair: Tuple[str, str]) -> Tuple[str, int]:
        sym, fld = col_pair
        try:
            rank = (_DEFAULT_FIELDS + ["Return"]).index(fld)
        except ValueError:
            rank = 999
        return (sym, rank)

    new_cols = sorted(panel.columns, key=_sort_key)
    panel = panel.reindex(columns=new_cols)
    panel.index.name = "Date"
    return panel


def _validate_symbols(symbols: Iterable[str]) -> List[str]:
    out = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    # De-duplicate preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _infer_reasonable_period_interval(
    period: Optional[str], interval: Optional[str]
) -> Tuple[str, str]:
    """
    Guardrail for incompatible period/interval combos that return empty frames.
    """
    p = period or "1y"
    i = interval or "1d"
    # Basic sanity: intraday intervals need short period on Yahoo
    intraday = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    if i in intraday and p not in {"1d", "5d", "1mo", "3mo"}:
        p = "1mo"
    return p, i


def fetch_many(
    symbols: Iterable[str],
    period: Optional[str] = "1y",
    interval: Optional[str] = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = False,
    back_adjust: bool = False,
    as_dict: bool = False,
) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    """
    Fetch multiple symbols and return a clean, aligned panel.

    Parameters
    ----------
    symbols : iterable of str
    period, interval, start, end, auto_adjust, back_adjust : see fetch_one
    as_dict : if True, return dict[symbol -> DataFrame]; otherwise a single panel DataFrame
              with MultiIndex columns (symbol, field).

    Returns
    -------
    pd.DataFrame (MultiIndex columns) or Dict[str, pd.DataFrame]

    Notes
    -----
    This function is designed to be called exactly like:
        panel = fetch_many(symbols, period=period, interval=interval)

    The returned 'panel' can be used like:
        px = panel.xs('Adj Close', axis=1, level=1)  # wide price table (Date x Symbols)
        r  = panel.xs('Return', axis=1, level=1)     # wide returns table
    """
    syms = _validate_symbols(symbols or [])
    if not syms:
        raise ValueError("fetch_many: at least one valid symbol is required.")

    # If start/end provided, let them override period/interval in fetch_one;
    # otherwise keep a compatible combo to avoid empty pulls.
    p, i = _infer_reasonable_period_interval(period, interval)

    frames: Dict[str, pd.DataFrame] = {}
    for s in syms:
        try:
            df = fetch_one(
                s,
                period=p,
                interval=i,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                back_adjust=back_adjust,
            )
            # Minimal sanity: ensure numeric, ensure at least a few rows
            if df.empty or df.dropna(how="all").shape[0] < 2:
                continue
            frames[s] = df
        except Exception:
            # Try Alpha Vantage fallback for this symbol only
            av = _alpha_vantage_fallback(s)
            if av is not None and not av.empty:
                frames[s] = av

    if not frames:
        raise ValueError("fetch_many: no data could be retrieved for the requested symbols.")

    if as_dict:
        return frames

    panel = _concat_as_panel(frames)
    if panel.empty:
        raise ValueError("fetch_many: combined panel is empty after alignment.")
    return panel


# ------------------------------ Convenience ----------------------------------

def latest_close(symbol: str) -> Optional[float]:
    """Return the latest available close (Adj Close preferred), or None."""
    df = fetch_one(symbol, period="5d", interval="1d")
    if df.empty:
        return None
    if "Adj Close" in df.columns:
        val = df["Adj Close"].iloc[-1]
    else:
        val = df["Close"].iloc[-1] if "Close" in df.columns else None
    return float(val) if val is not None and not pd.isna(val) else None


def wide_prices(panel: pd.DataFrame, field: str = "Adj Close") -> pd.DataFrame:
    """
    From a panel (MultiIndex columns), return a wide table Date x Symbols for a chosen field.
    """
    if not isinstance(panel.columns, pd.MultiIndex):
        raise TypeError("wide_prices expects a panel with MultiIndex columns (symbol, field).")
    if field not in panel.columns.get_level_values(1):
        # fall back to Close if Adj Close missing
        field = "Close" if "Close" in panel.columns.get_level_values(1) else list(
            dict.fromkeys(panel.columns.get_level_values(1))
        )[0]
    return panel.xs(field, axis=1, level=1)


# ------------------------------ Self-test ------------------------------------

if __name__ == "__main__":
    # Quick smoke test (won't run on Streamlit Cloud automatically)
    tickers = ["AAPL", "MSFT", "SPY"]
    pnl = fetch_many(tickers, period="6mo", interval="1d")
    print(pnl.tail())
    prices = wide_prices(pnl, "Adj Close")
    print(prices.tail())
    print("Latest AAPL:", latest_close("AAPL"))
