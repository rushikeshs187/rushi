# data.py
import pandas as pd
import yfinance as yf

def fetch_one(ticker: str, period="2y", interval="1d") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        # Ensure required columns exist
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df
    except Exception:
        return None

def fetch_many(tickers: list[str], period="2y", interval="1d") -> dict[str, pd.DataFrame]:
    panel = {}
    for t in tickers:
        df = fetch_one(t, period=period, interval=interval)
        if df is not None and not df.empty:
            panel[t] = df
    return panel
