from __future__ import annotations
import numpy as np
import pandas as pd

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def _bbands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = _sma(series, window)
    sd = series.rolling(window).std(ddof=0)
    upper = ma + num_std*sd
    lower = ma - num_std*sd
    return upper, ma, lower

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def build_features(df: pd.DataFrame, by_symbol: bool = True) -> pd.DataFrame:
    """Add returns + TA features per symbol. Expects columns: Symbol, Price, Volume."""
    if df.empty: 
        return df
    out = []
    for sym, g in df.groupby("Symbol"):
        g = g.sort_index().copy()
        px = pd.to_numeric(g["Price"], errors="coerce")
        vol = pd.to_numeric(g["Volume"], errors="coerce")
        ret = px.pct_change()
        g["Ret1"] = ret
        g["Ret5"] = px.pct_change(5)
        g["VolChg"] = vol.pct_change().replace([np.inf, -np.inf], np.nan)

        # momentum/mean reversion
        g["SMA20"] = _sma(px, 20)
        g["SMA50"] = _sma(px, 50)
        g["EMA12"] = _ema(px, 12)
        g["EMA26"] = _ema(px, 26)
        macd, macd_sig, macd_hist = _macd(px)
        g["MACD"] = macd
        g["MACDsig"] = macd_sig
        g["MACDhist"] = macd_hist
        g["RSI14"] = _rsi(px, 14)
        bbU, bbM, bbL = _bbands(px, 20, 2.0)
        g["BBU"] = bbU
        g["BBM"] = bbM
        g["BBL"] = bbL
        # rolling vol
        g["Volatility20"] = g["Ret1"].rolling(20).std(ddof=0)
        # lags for autocorrelation
        for k in (1,2,3,5,10):
            g[f"Ret1_lag{k}"] = g["Ret1"].shift(k)
        # target: next-day direction (1 up, 0 down/flat)
        g["TargetNextUp"] = (g["Ret1"].shift(-1) > 0).astype(int)
        out.append(g)
    f = pd.concat(out).dropna(subset=["Price"])
    return f
