import numpy as np
import pandas as pd

def _positions_sma(df: pd.DataFrame, sma_fast: int, sma_slow: int) -> pd.Series:
    fast = df["Close"].rolling(sma_fast, min_periods=max(2, sma_fast//2)).mean()
    slow = df["Close"].rolling(sma_slow, min_periods=max(2, sma_slow//2)).mean()
    pos = (fast > slow).astype(int)  # long=1, flat=0
    return pos

def _positions_rsi(df: pd.DataFrame, rsi_buy: int, rsi_sell: int) -> pd.Series:
    if "RSI" not in df.columns:
        return pd.Series(0, index=df.index)
    rsi = df["RSI"]
    # Long when oversold; exit when overbought
    pos = pd.Series(0, index=df.index, dtype=int)
    pos = np.where(rsi < rsi_buy, 1, np.where(rsi > rsi_sell, 0, np.nan))
    pos = pd.Series(pos, index=df.index).ffill().fillna(0).astype(int)
    return pos

def run_backtest(
    df: pd.DataFrame,
    strategy: str = "SMA Crossover",
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_buy: int = 30,
    rsi_sell: int = 70,
    tcost_bps: int = 5,
) -> dict:
    d = df.copy()
    d = d.sort_index()
    d = d.dropna(subset=["Close"])

    if strategy == "SMA Crossover":
        pos = _positions_sma(d, sma_fast, sma_slow)
    else:
        pos = _positions_rsi(d, rsi_buy, rsi_sell)

    # Returns
    ret = d["Close"].pct_change().fillna(0.0)
    trades = pos.diff().abs().fillna(0.0)  # 1 on position change
    tcost = (tcost_bps / 10000.0) * trades  # cost as fraction
    strat_ret = pos.shift(1).fillna(0) * ret - tcost

    equity = (1 + strat_ret).cumprod()
    equity.name = "equity"
    return {"equity": equity, "position": pos, "returns": strat_ret}
