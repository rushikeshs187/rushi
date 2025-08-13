# utils.py
from __future__ import annotations
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ----------------------------
# IO helpers
# ----------------------------
def load_csv(file, date_col="Date") -> tuple[pd.DataFrame | None, str | None]:
    """Load a CSV, coerce date column, set DatetimeIndex, normalize common OHLC names."""
    try:
        df = pd.read_csv(file)
        if date_col not in df.columns:
            return None, f"Date column '{date_col}' not found in CSV."
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        df = df.set_index(date_col)

        # Normalize common OHLC names (case/spacing tolerant)
        rename_map = {}
        for c in df.columns:
            key = c.strip()
            if key.lower() in {"open","high","low","close","adj close","volume"}:
                # Title-case except Adj Close capitalization
                if key.lower() == "adj close":
                    rename_map[c] = "Adj Close"
                else:
                    rename_map[c] = key.title()
        if rename_map:
            df = df.rename(columns=rename_map)

        return df, None
    except Exception as e:
        return None, f"Failed to read CSV: {e}"

def fetch_ticker_history(ticker: str, years: int = 3) -> tuple[pd.DataFrame | None, str | None]:
    """Fetch daily history via yfinance. Requires yfinance in requirements."""
    try:
        import yfinance as yf
    except Exception:
        return None, "yfinance is not installed. Add it to requirements.txt"

    try:
        end = date.today()
        start = end - timedelta(days=365 * years + 7)
        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data is None or data.empty:
            return None, f"No data returned for '{ticker}'. Check the symbol."
        data.index = pd.to_datetime(data.index)

        # Ensure expected columns exist
        for col in ["Open","High","Low","Close","Adj Close","Volume"]:
            if col not in data.columns:
                data[col] = np.nan
        return data, None
    except Exception as e:
        return None, f"Failed to fetch {ticker}: {e}"

# ----------------------------
# Indicators
# ----------------------------
def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower

def add_technical_indicators(df: pd.DataFrame, sma=True, rsi=True, bb=True) -> pd.DataFrame:
    out = df.copy()
    # If duplicate 'Close' columns exist, reduce to first one
    if "Close" not in out.columns:
        return out
    s = out["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
        out = out.drop(columns=["Close"]).assign(Close=s)

    close = pd.to_numeric(out["Close"], errors="coerce")
    out["Close"] = close

    if sma:
        out["SMA20"] = _sma(close, 20)
        out["SMA50"] = _sma(close, 50)
    if rsi:
        out["RSI"] = _rsi(close, 14)
    if bb:
        m, u, l = _bollinger(close, 20, 2.0)
        out["BB_M"], out["BB_U"], out["BB_L"] = m, u, l
    return out

# ----------------------------
# KPIs & returns (robust)
# ----------------------------
def _first_close_series(df: pd.DataFrame) -> pd.Series:
    """Return a single numeric Close series even if duplicate 'Close' columns are present."""
    if "Close" not in df.columns:
        raise KeyError("Column 'Close' not found.")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna().sort_index()
    return s

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Robust KPIs: last price, D1, MTD, YTD, Sharpe (daily, ~252).
    Fixes the 'truth value of a Series is ambiguous' issue by ensuring scalars.
    """
    out = {"price_last": np.nan, "d1": np.nan, "mtd": np.nan, "ytd": np.nan, "sharpe": np.nan}

    # Ensure DatetimeIndex if possible
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            tmp = df.copy()
            tmp.index = pd.to_datetime(tmp.index, errors="coerce")
            tmp = tmp[~tmp.index.isna()]
            df = tmp
        except Exception:
            return out

    # Get a single, numeric Close series
    try:
        s = _first_close_series(df)
    except Exception:
        return out

    if s.empty:
        return out

    s = s.sort_index()
    out["price_last"] = float(s.iloc[-1])

    # D1
    if len(s) >= 2 and s.iloc[-2] not in (0, np.nan):
        out["d1"] = float((s.iloc[-1] / s.iloc[-2] - 1) * 100.0)

    # MTD / YTD
    last_dt = s.index[-1]
    try:
        same_month = s[s.index.to_period("M") == last_dt.to_period("M")]
        if not same_month.empty and same_month.iloc[0] != 0:
            out["mtd"] = float((s.iloc[-1] / same_month.iloc[0] - 1) * 100.0)
    except Exception:
        pass

    try:
        same_year = s[s.index.year == last_dt.year]
        if not same_year.empty and same_year.iloc[0] != 0:
            out["ytd"] = float((s.iloc[-1] / same_year.iloc[0] - 1) * 100.0)
    except Exception:
        pass

    # Sharpe (daily)
    ret = s.pct_change().dropna()
    if not ret.empty:
        std = float(ret.std(ddof=0))  # scalar
        if np.isfinite(std) and std > 0:
            mean = float(ret.mean())
            daily_sharpe = mean / std
            out["sharpe"] = float(daily_sharpe * np.sqrt(252.0))

    return out

def to_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Safe returns DataFrame with a single 'Returns' column."""
    try:
        s = _first_close_series(df)
    except Exception:
        return pd.DataFrame(columns=["Returns"])
    r = s.pct_change().rename("Returns")
    return pd.DataFrame(r)
