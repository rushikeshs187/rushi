from __future__ import annotations
import time
import pandas as pd
import yfinance as yf

US_DEFAULTS = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","XOM","UNH","V"]
EM_DEFAULTS = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS","BHARTIARTL.NS",
               "ITC.NS","SBIN.NS","LT.NS","ASIANPAINT.NS"]

def _download_one(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.rename(columns=str.title)
        df.index = pd.to_datetime(df.index)
        df["Symbol"] = symbol
        # prefer Adj Close if present
        if "Adj Close" in df.columns:
            df["Price"] = df["Adj Close"]
        else:
            df["Price"] = df["Close"]
        return df[["Symbol","Open","High","Low","Close","Price","Volume"]].dropna(how="all")
    return pd.DataFrame()

def fetch_many(symbols, period="5y", interval="1d", throttle=0.4) -> pd.DataFrame:
    frames = []
    tried = 0
    for s in symbols:
        tried += 1
        try:
            d = _download_one(s, period, interval)
            if not d.empty:
                frames.append(d)
        except Exception:
            pass
        time.sleep(throttle)
    if not frames:
        raise ValueError("fetch_many: no data could be retrieved for the requested symbols.")
    df = pd.concat(frames).sort_index()
    # sanity: drop duplicated timestamps per symbol
    df = df[~df.reset_index().duplicated(subset=["Date","Symbol"]).values]
    return df

def default_universe(market: str) -> list[str]:
    market = (market or "US").upper()
    return US_DEFAULTS if market == "US" else EM_DEFAULTS
