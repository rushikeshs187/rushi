from __future__ import annotations
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

# yfinance is optional until we need it, so we import lazily inside the function.

def load_csv(file, date_col="Date") -> tuple[pd.DataFrame | None, str | None]:
    try:
        if isinstance(file, (str, io.IOBase)):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
        if date_col not in df.columns:
            return None, f"Date column '{date_col}' not found in CSV."
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        df = df.set_index(date_col)
        # Normalize common OHLC names
        rename_map = {c: c.strip().title() for c in df.columns}
        df = df.rename(columns=rename_map)
        return df, None
    except Exception as e:
        return None, f"Failed to read CSV: {e}"

def fetch_ticker_history(ticker: str, years: int = 3) -> tuple[pd.DataFrame | None, str | None]:
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
        # Ensure expected columns
        for col in ["Open","High","Low","Close","Adj Close","Volume"]:
            if col not in data.columns:
                data[col] = np.nan
        return data, None
    except Exception as e:
        return None, f"Failed to fetch {ticker}: {e}"

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
    if "Close" not in out.columns:
        return out
    close = out["Close"]
    if sma:
        out["SMA20"] = _sma(close, 20)
        out["SMA50"] = _sma(close, 50)
    if rsi:
        out["RSI"] = _rsi(close, 14)
    if bb:
        m, u, l = _bollinger(close, 20, 2.0)
        out["BB_M"], out["BB_U"], out["BB_L"] = m, u, l
    return out

def compute_kpis(df: pd.DataFrame) -> dict:
    out = {
        "price_last": float(df["Close"].iloc[-1]) if "Close" in df.columns else float("nan"),
        "d1": float("nan"),
        "mtd": float("nan"),
        "ytd": float("nan"),
        "sharpe": float("nan"),
    }
    if "Close" not in df.columns or len(df) < 3:
        return out

    s = df["Close"]
    out["d1"] = (s.iloc[-1] / s.iloc[-2] - 1) * 100

    # Month-to-date
    last = s.index[-1]
    m_start = s.index[(s.index.year == last.year) & (s.index.month == last.month)][0]
    out["mtd"] = (s.iloc[-1] / s.loc[m_start] - 1) * 100

    # Year-to-date
    y_start = s.index[(s.index.year == last.year)][0]
    out["ytd"] = (s.iloc[-1] / s.loc[y_start] - 1) * 100

    # Sharpe (daily, risk-free ~0)
    ret = s.pct_change().dropna()
    if ret.std(ddof=0) > 0:
        daily_sharpe = ret.mean() / ret.std(ddof=0)
        out["sharpe"] = float(daily_sharpe * np.sqrt(252))
    return out

def to_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        return pd.DataFrame()
    r = df["Close"].pct_change().rename("Returns")
    return pd.DataFrame(r)
