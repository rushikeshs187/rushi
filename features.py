# features.py
import numpy as np
import pandas as pd

# ---------- Indicators ----------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    out = 100 - 100 / (1 + rs)
    return out.clip(0, 100).rename("RSI")

def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()

def build_indicators(df: pd.DataFrame,
                     rsi_window=14, bb_window=20, bb_std=2.0,
                     ema_fast=12, ema_slow=26, macd_signal=9) -> pd.DataFrame:
    out = df.copy()
    px = out["Adj Close"] if "Adj Close" in out.columns else out["Close"]
    px = pd.to_numeric(px, errors="coerce")
    out["Return"] = px.pct_change()
    out["LogReturn"] = np.log(px / px.shift(1))

    # RSI
    out["RSI"] = rsi(px, rsi_window)

    # Bollinger
    mid = px.rolling(bb_window, min_periods=bb_window).mean()
    sd = px.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    out["BB_Mid"] = mid
    out["BB_Upper"] = mid + bb_std * sd
    out["BB_Lower"] = mid - bb_std * sd

    # MACD
    ema_f = ema(px, ema_fast)
    ema_s = ema(px, ema_slow)
    macd = ema_f - ema_s
    out["MACD"] = macd
    out["MACD_Signal"] = macd.ewm(span=macd_signal, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    return out

# ---------- KPIs ----------
def sharpe(returns: pd.Series, periods=252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty or r.std(ddof=0) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=0)) * np.sqrt(periods)

def sortino(returns: pd.Series, periods=252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    downside = r[r < 0]
    if r.empty or downside.std(ddof=0) == 0:
        return np.nan
    return (r.mean() / downside.std(ddof=0)) * np.sqrt(periods)

def max_drawdown_from_equity(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if not dd.empty else np.nan

def cagr_from_equity(equity: pd.Series, periods=252) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return np.nan
    n_years = len(equity) / periods
    if n_years <= 0:
        return np.nan
    return float(equity.iloc[-1] ** (1 / n_years) - 1)

def compute_kpis(df: pd.DataFrame) -> dict:
    r = pd.to_numeric(df.get("Return", pd.Series(dtype=float)), errors="coerce").dropna()
    equity = (1 + r).cumprod()
    return {
        "CAGR": cagr_from_equity(equity),
        "Sharpe": sharpe(r),
        "Sortino": sortino(r),
        "MaxDD": max_drawdown_from_equity(equity),
        "HitRate": (r > 0).mean() if len(r) else np.nan
    }

# ---------- ML Frame ----------
def make_target(df: pd.DataFrame, price_col="Adj Close", horizon=5, threshold=0.0) -> pd.Series:
    px = pd.to_numeric(df[price_col] if price_col in df.columns else df["Close"], errors="coerce")
    fwd = px.shift(-horizon) / px - 1.0
    return (fwd > threshold).astype(int).rename("Target")

def prepare_ml_frame(df_ind: pd.DataFrame, horizon=5, threshold=0.0) -> pd.DataFrame | None:
    out = df_ind.copy()
    out["Target"] = make_target(out, price_col="Adj Close" if "Adj Close" in out.columns else "Close",
                                horizon=horizon, threshold=threshold)
    # Drop tail with undefined forward
    out = out.iloc[:-horizon] if len(out) > horizon else out.iloc[:0]

    # Features (numeric, no leakage)
    feats = [
        "Return","LogReturn","RSI","BB_Mid","BB_Upper","BB_Lower",
        "MACD","MACD_Signal","MACD_Hist",
        "Open","High","Low","Close","Adj Close","Volume"
    ]
    feats = [c for c in feats if c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + ["Target"])
    if out.empty:
        return None
    # Simple lag features to avoid look-ahead
    for col in ["Return","RSI","MACD","MACD_Signal","MACD_Hist"]:
        if col in out.columns:
            out[col + "_lag1"] = out[col].shift(1)
    out = out.dropna()
    return out
