from __future__ import annotations
import numpy as np
import pandas as pd

def annualize_return(daily_ret: pd.Series) -> float:
    m = daily_ret.mean()
    return (1 + m) ** 252 - 1

def sharpe_ratio(daily_ret: pd.Series, rf: float = 0.0) -> float:
    if daily_ret.std(ddof=0) == 0 or daily_ret.dropna().empty:
        return 0.0
    er = daily_ret - rf/252
    return np.sqrt(252) * er.mean() / er.std(ddof=0)

def sortino_ratio(daily_ret: pd.Series, rf: float = 0.0) -> float:
    downside = daily_ret[daily_ret < 0]
    if downside.std(ddof=0) == 0 or daily_ret.dropna().empty:
        return 0.0
    er = daily_ret - rf/252
    return np.sqrt(252) * er.mean() / downside.std(ddof=0)

def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.dropna().empty:
        return 0.0
    roll_max = equity_curve.cummax()
    dd = equity_curve/roll_max - 1.0
    return dd.min()

def hit_rate(pred: pd.Series, actual: pd.Series) -> float:
    ok = (pred == actual).astype(float)
    return float(ok.mean()) if not ok.empty else 0.0

def to_direction(x: pd.Series) -> pd.Series:
    return (x > 0).astype(int)

def safe_pct(x: float, digits: int = 2) -> str:
    try:
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "—"

def non_empty(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and df.shape[0] > 5
