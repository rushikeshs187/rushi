from __future__ import annotations
from typing import Dict, List
import time
import numpy as np
import pandas as pd
import yfinance as yf

# ------------------ Market universes (expandable) ------------------ #
UNIVERSES: Dict[str, List[str]] = {
    "US (MegaCaps)": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","XOM","UNH","V"],
    "India (NSE)": ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
                    "BHARTIARTL.NS","ITC.NS","SBIN.NS","LT.NS","ASIANPAINT.NS"],
    "UK (LSE)": ["ULVR.L","HSBA.L","BP.L","GSK.L","VOD.L","DGE.L","BATS.L","AZN.L","RIO.L","SHEL.L"],
    "Japan (TSE)": ["7203.T","6758.T","9432.T","9984.T","9433.T","8306.T","6861.T","4063.T","6954.T","8035.T"],
    "Europe (ETFs)": ["EZU","VGK","FEZ"],  # Eurozone/Europe large-cap proxies
    "China (ETFs)": ["MCHI","FXI"],
    "Global (ETFs)": ["ACWI","VT","VEA","VWO"],
    "Crypto": ["BTC-USD","ETH-USD","SOL-USD"],
    "FX (majors)": ["EURUSD=X","GBPUSD=X","USDJPY=X","USDINR=X"],
    "Commodities": ["GC=F","SI=F","CL=F","NG=F"],
}

def list_markets() -> List[str]:
    return list(UNIVERSES.keys())

def universe_for(markets: List[str]) -> List[str]:
    syms: List[str] = []
    for m in markets:
        syms.extend(UNIVERSES.get(m, []))
    # de-duplicate, preserve order
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

# ------------------ Core download helpers ------------------ #
def _download_one(symbol: str, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if isinstance(df, pd.DataFrame) and not df.empty:
        # yfinance sometimes returns MultiIndex columns for a single ticker; flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]
        df = df.rename(columns=str.title)  # Open/High/Low/Close/Adj Close/Volume
        df = df.sort_index()
        # Create Price (prefer Adj Close)
        price = df["Adj Close"] if "Adj Close" in df.columns else df.get("Close")
        out = pd.DataFrame({
            "Open": pd.to_numeric(df.get("Open"), errors="coerce"),
            "High": pd.to_numeric(df.get("High"), errors="coerce"),
            "Low": pd.to_numeric(df.get("Low"), errors="coerce"),
            "Close": pd.to_numeric(df.get("Close"), errors="coerce"),
            "Adj Close": pd.to_numeric(df.get("Adj Close"), errors="coerce") if "Adj Close" in df.columns else pd.NA,
            "Price": pd.to_numeric(price, errors="coerce"),
            "Volume": pd.to_numeric(df.get("Volume"), errors="coerce"),
        }, index=pd.to_datetime(df.index))
        out["Symbol"] = symbol
        return out.dropna(how="all")
    return pd.DataFrame()

# ------------------ Public API ------------------ #
def fetch_many(
    symbols: List[str],
    period: str = "5y",
    interval: str = "1d",
    auto_adjust: bool = True,
    throttle: float = 0.35,
) -> pd.DataFrame:
    """
    Returns a tidy panel with columns:
      Date (index), Symbol, Open, High, Low, Close, Adj Close, Price, Volume
    Deduplicated on (Date, Symbol) reliably (no KeyError).
    """
    frames = []
    for s in symbols:
        s = s.strip()
        if not s:
            continue
        try:
            d = _download_one(s, period, interval, auto_adjust)
            if not d.empty:
                frames.append(d)
        except Exception:
            pass
        time.sleep(throttle)

    if not frames:
        raise ValueError("fetch_many: no data could be retrieved for the requested symbols.")

    df = pd.concat(frames).sort_index()

    # ---- SAFE de-dup: always materialize a 'Date' column, then drop dupes ---- #
    tmp = df.reset_index()  # column will be 'index' if index has no name
    # Normalize the datetime column name to 'Date'
    if "Date" not in tmp.columns:
        # The datetime column is likely named 'index'
        dt_col = "Date" if "Date" in tmp.columns else "index"
        tmp = tmp.rename(columns={dt_col: "Date"})
    # now we have ['Date','Symbol', ...]
    tmp = tmp.drop_duplicates(subset=["Date", "Symbol"])
    # restore DateTimeIndex named 'Date'
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    tmp = tmp.set_index("Date").sort_index()
    return tmp[["Symbol","Open","High","Low","Close","Adj Close","Price","Volume"]]

