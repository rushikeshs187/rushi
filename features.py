import numpy as np
import pandas as pd

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _ema(s: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.ewm(span=span, adjust=False).mean()

def _sma(s: pd.Series, win: int) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.rolling(win, min_periods=max(2, win//2)).mean()

def build_indicators(
    df: pd.DataFrame,
    rsi_window: int = 14,
    ema_fast: int = 12,
    ema_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Ensure numeric
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Return"] = out["Close"].pct_change()
    out["LogRet"] = np.log(out["Close"]).diff()

    # EMAs
    out["EMA_Fast"] = _ema(out["Close"], ema_fast)
    out["EMA_Slow"] = _ema(out["Close"], ema_slow)

    # MACD
    macd = out["EMA_Fast"] - out["EMA_Slow"]
    sig = macd.ewm(span=macd_signal, adjust=False).mean()
    out["MACD"] = macd
    out["MACD_Signal"] = sig
    out["MACD_Hist"] = macd - sig

    # RSI
    out["RSI"] = _rsi(out["Close"], window=rsi_window)

    # Bollinger
    mid = _sma(out["Close"], bb_window)
    std = out["Close"].rolling(bb_window, min_periods=max(2, bb_window//2)).std()
    out["BB_Mid"] = mid
    out["BB_Upper"] = mid + bb_std * std
    out["BB_Lower"] = mid - bb_std * std

    # SMAs for crossover strategy convenience
    out["SMA_Fast"] = _sma(out["Close"], max(5, min(200, 20)))
    out["SMA_Slow"] = _sma(out["Close"], max(10, min(400, 50)))

    return out.dropna(how="all")
