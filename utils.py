import numpy as np
import pandas as pd

# ---------- Safe helpers ----------

def _to_datetime_index(df, date_col=None):
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        # last resort: try to parse index
        try:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
            df = df.dropna(subset=[df.index.name]).sort_index()
        except Exception:
            pass
    return df

def _ensure_float(series):
    return pd.to_numeric(series, errors="coerce")

def compute_returns(df, price_col="Adj Close"):
    if price_col not in df.columns:
        # try common fallbacks
        for c in ["Close", "close", "Adj_Close", "adj_close"]:
            if c in df.columns:
                price_col = c
                break
    if price_col not in df.columns:
        raise ValueError("Could not find a price column (expected 'Adj Close' or 'Close').")

    px = _ensure_float(df[price_col]).copy()
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return ret.rename("returns"), px.rename("price")

# ---------- Indicators ----------

def rsi(series, window=14):
    s = _ensure_float(series).copy()
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Avoid division by zero by adding small epsilon
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.rename("RSI")

def sma(series, window=20):
    return _ensure_float(series).rolling(window).mean().rename(f"SMA_{window}")

def ema(series, window=20):
    return _ensure_float(series).ewm(span=window, adjust=False).mean().rename(f"EMA_{window}")

def bollinger(series, window=20, n_std=2):
    m = sma(series, window)
    s = _ensure_float(series).rolling(window).std(ddof=0)
    upper = (m + n_std * s).rename("BB_upper")
    lower = (m - n_std * s).rename("BB_lower")
    return m.rename("BB_mid"), upper, lower

# ---------- KPIs (robust) ----------

def max_drawdown(cumret):
    peak = cumret.cummax()
    dd = (cumret / peak) - 1.0
    return dd.min()

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    r = _ensure_float(returns)
    if r.std(ddof=0) == 0 or r.empty:
        return np.nan
    excess = r - (risk_free / periods_per_year)
    ann_ret = excess.mean() * periods_per_year
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    if ann_vol == 0:
        return np.nan
    return ann_ret / ann_vol

def cagr(returns, periods_per_year=252):
    r = _ensure_float(returns)
    if r.empty:
        return np.nan
    cum = (1 + r).cumprod()
    n_years = len(r) / periods_per_year
    if n_years <= 0:
        return np.nan
    return cum.iloc[-1]**(1/n_years) - 1

def hit_rate(returns):
    r = _ensure_float(returns)
    if r.empty:
        return np.nan
    return (r > 0).sum() / len(r)

def compute_kpis(df, price_col="Adj Close", risk_free_annual=0.0, periods_per_year=252):
    """
    Robust KPI calc. Returns a dict; never raises the ambiguous truth ValueError.
    """
    returns, price = compute_returns(df, price_col)
    if returns.empty:
        return {
            "Sharpe": np.nan, "CAGR": np.nan, "Max Drawdown": np.nan,
            "Hit Rate": np.nan, "Ann Return": np.nan, "Ann Vol": np.nan
        }
    cum = (1 + returns).cumprod()
    k = {}
    k["Sharpe"] = sharpe_ratio(returns, risk_free=risk_free_annual, periods_per_year=periods_per_year)
    k["CAGR"] = cagr(returns, periods_per_year=periods_per_year)
    k["Max Drawdown"] = max_drawdown(cum)
    k["Hit Rate"] = hit_rate(returns)
    k["Ann Return"] = returns.mean() * periods_per_year
    k["Ann Vol"] = returns.std(ddof=0) * (periods_per_year ** 0.5)
    return k

# ---------- Preprocess entrypoint ----------

def prepare_timeseries(df, date_col=None, price_col="Adj Close"):
    df = df.copy()
    df = _to_datetime_index(df, date_col)
    # Ensure numeric price
    if price_col in df.columns:
        df[price_col] = _ensure_float(df[price_col])
    # Technicals
    _, price = compute_returns(df, price_col)
    df["RSI"] = rsi(price)
    df["SMA_20"] = sma(price, 20)
    df["EMA_20"] = ema(price, 20)
    bb_mid, bb_u, bb_l = bollinger(price, 20, 2)
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bb_mid, bb_u, bb_l
    return df
