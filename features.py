import numpy as np
import pandas as pd

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return ema_fast, ema_slow, macd, macd_signal

def _bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return ma, upper, lower

def build_indicators(
    df: pd.DataFrame,
    rsi_window: int = 14,
    ema_fast: int = 12,
    ema_slow: int = 26,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_stds: float = 2.0
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    px = pd.to_numeric(out.get("Adj Close", out.get("Close")), errors="coerce")
    out["EMA_Fast"] = _ema(px, ema_fast)
    out["EMA_Slow"] = _ema(px, ema_slow)
    out["RSI"] = _rsi(px, rsi_window)
    ef, es, macd, macd_sig = _macd(px, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    out["MACD"] = macd
    out["MACD_Signal"] = macd_sig
    ma, up, lo = _bollinger(px, bb_window, bb_stds)
    out["BB_MA"] = ma
    out["BB_Upper"] = up
    out["BB_Lower"] = lo
    return out
