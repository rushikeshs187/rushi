from __future__ import annotations
import numpy as np
import pandas as pd
from utils import sharpe_ratio, sortino_ratio, max_drawdown, hit_rate, annualize_return, to_direction

def backtest_long_flat(df: pd.DataFrame, signal_col: str = "Signal") -> dict:
    """Simple long/flat on symbol-level average return."""
    # Average across symbols to form a mini 'portfolio'
    panel = df.copy()
    # compute per-symbol daily return
    panel["Ret1"] = panel.groupby("Symbol")["Price"].pct_change()
    # pivot to wide
    wide = panel.pivot_table(index=panel.index, columns="Symbol", values="Ret1").dropna(how="all")
    # average cross-section
    mkt_ret = wide.mean(axis=1).fillna(0.0)
    # align signal (already pooled in df): take median across symbols of signal for day
    sig = df.groupby(df.index)["Signal"].median().reindex(mkt_ret.index).fillna(0)
    strat_ret = mkt_ret * sig
    eq = (1 + strat_ret).cumprod()
    bench = (1 + mkt_ret).cumprod()

    metrics = {
        "Annualized Return": annualize_return(strat_ret),
        "Sharpe": sharpe_ratio(strat_ret),
        "Sortino": sortino_ratio(strat_ret),
        "Max Drawdown": max_drawdown(eq),
        "Hit Rate (Days)": hit_rate(to_direction(strat_ret), to_direction(mkt_ret)),
        "Cumulative Return": eq.iloc[-1] - 1 if not eq.empty else 0.0,
        "Benchmark CumRet": bench.iloc[-1] - 1 if not bench.empty else 0.0
    }
    curves = {"equity": eq, "benchmark": bench, "signal": sig, "strat_ret": strat_ret, "mkt_ret": mkt_ret}
    return {"metrics": metrics, "curves": curves}
