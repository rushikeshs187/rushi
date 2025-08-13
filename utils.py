import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252

def ensure_tz_safe_today():
    # Avoid timezone pitfalls in hosted envs
    return pd.Timestamp.utcnow().tz_localize(None).normalize().date()

def safe_pct_change(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series([], dtype=float)
    return series.astype(float).pct_change()

def compute_drawdown(price: pd.Series) -> pd.DataFrame:
    s = price.astype(float).dropna()
    if s.empty:
        return pd.DataFrame(columns=["peak", "value", "drawdown"])
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    return pd.DataFrame({"peak": roll_max, "value": s, "drawdown": dd})

def rolling_stats(price: pd.Series, windows=[20, 50, 100, 200]) -> pd.DataFrame:
    out = pd.DataFrame(index=price.index)
    for w in windows:
        out[f"MA_{w}"] = price.rolling(w).mean()
        out[f"VOL_{w}"] = price.pct_change().rolling(w).std()
    return out

def _annualize_mean(ret: pd.Series) -> float:
    m = float(ret.mean())
    return (1 + m) ** TRADING_DAYS - 1 if not math.isnan(m) else float("nan")

def _annualize_vol(ret: pd.Series) -> float:
    v = float(ret.std(ddof=0))
    return v * (TRADING_DAYS ** 0.5) if not math.isnan(v) else float("nan")

def _max_drawdown_from_returns(ret: pd.Series) -> float:
    cum = (1 + ret.fillna(0)).cumprod()
    dd = compute_drawdown(cum)
    if dd.empty:
        return float("nan")
    return float(dd["drawdown"].min())

def compute_kpis(ret: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return dict(CAGR=np.nan, AnnRet=np.nan, AnnVol=np.nan, Sharpe=np.nan, Sortino=np.nan,
                    MaxDD=np.nan, Calmar=np.nan, Beta=np.nan, Alpha=np.nan, HitRate=np.nan)

    rf = 0.0  # risk-free placeholder; could be extended with FRED
    ann_ret = _annualize_mean(ret)
    ann_vol = _annualize_vol(ret)

    downside = ret.copy()
    downside[downside > 0] = 0
    downside_vol = float(downside.std(ddof=0)) * (TRADING_DAYS ** 0.5)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol and not math.isnan(ann_vol) and ann_vol != 0 else float("nan")
    sortino = (ann_ret - rf) / downside_vol if downside_vol and not math.isnan(downside_vol) and downside_vol != 0 else float("nan")

    maxdd = _max_drawdown_from_returns(ret)
    calmar = ann_ret / abs(maxdd) if maxdd and not math.isnan(maxdd) and maxdd != 0 else float("nan")

    beta = alpha = np.nan
    if benchmark is not None and len(benchmark.dropna()) > 10:
        b = benchmark.align(ret, join="inner")[0].pct_change().dropna()
        r = ret.align(benchmark, join="inner")[0].dropna()
        # re-align properly on returns
        r = r.reindex(b.index).dropna()
        if len(b) == len(r) and len(b) > 2:
            cov = np.cov(r, b)[0, 1]
            var = np.var(b)
            beta = cov / var if var != 0 else np.nan
            # Alpha (approx): ann_ret - (rf + beta*(bench_ann - rf))
            bench_ann = _annualize_mean(b)
            alpha = ann_ret - (rf + (beta if not math.isnan(beta) else 0.0) * (bench_ann - rf))

    # CAGR from first to last
    cum = (1 + ret.fillna(0)).cumprod()
    if len(cum) >= 2:
        n_years = len(cum) / TRADING_DAYS
        cagr = (cum.iloc[-1] ** (1 / n_years)) - 1 if cum.iloc[-1] > 0 else np.nan
    else:
        cagr = np.nan

    hit = (ret > 0).mean() if len(ret) else np.nan

    return dict(
        CAGR=cagr, AnnRet=ann_ret, AnnVol=ann_vol,
        Sharpe=sharpe, Sortino=sortino, MaxDD=maxdd, Calmar=calmar,
        Beta=beta, Alpha=alpha, HitRate=hit
    )

def humanize_kpis(k: Dict[str, float]) -> str:
    fmt_pct = lambda x: "—" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2%}"
    fmt_num = lambda x: "—" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"

    lines = [
        f"**CAGR:** {fmt_pct(k.get('CAGR'))}  |  **Ann. Return:** {fmt_pct(k.get('AnnRet'))}  |  **Ann. Vol:** {fmt_pct(k.get('AnnVol'))}",
        f"**Sharpe:** {fmt_num(k.get('Sharpe'))}  |  **Sortino:** {fmt_num(k.get('Sortino'))}  |  **Calmar:** {fmt_num(k.get('Calmar'))}",
        f"**Max Drawdown:** {fmt_pct(k.get('MaxDD'))}  |  **Hit Rate:** {fmt_pct(k.get('HitRate'))}",
        f"**Beta vs Bench:** {fmt_num(k.get('Beta'))}  |  **Alpha (ann):** {fmt_pct(k.get('Alpha'))}",
    ]
    return "\n\n".join(lines)
