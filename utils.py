import numpy as np
import pandas as pd
from math import sqrt

def kpis_from_returns(ret: pd.Series, trading_days: int = 252) -> dict:
    r = ret.dropna()
    if r.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0}

    # CAGR
    cum = (1 + r).cumprod()
    n_years = len(r) / trading_days if trading_days > 0 else 1
    cagr = (cum.iloc[-1] ** (1 / max(n_years, 1e-9))) - 1

    # Sharpe (risk-free ~ 0)
    sharpe = (r.mean() / (r.std(ddof=0) + 1e-12)) * sqrt(trading_days)

    # Sortino (downside deviation)
    downside = r.copy()
    downside[downside > 0] = 0
    dd = downside.std(ddof=0)
    sortino = (r.mean() / (dd + 1e-12)) * sqrt(trading_days)

    # Max Drawdown
    roll_max = cum.cummax()
    dd_series = (cum / roll_max) - 1.0
    max_dd = dd_series.min()

    return {"cagr": float(cagr), "sharpe": float(sharpe), "sortino": float(sortino), "max_drawdown": float(max_dd)}

def format_kpi(x: float) -> str:
    return f"{x*100:.2f}%"
