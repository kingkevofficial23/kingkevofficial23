"""Market data loading: CSV files, Yahoo Finance (optional), or synthetic demo data."""

import numpy as np
import pandas as pd

OHLC = ["Open", "High", "Low", "Close"]


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV with a Date column plus Open/High/Low/Close (Volume optional)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().title() for c in df.columns]
    date_col = next((c for c in df.columns if c in ("Date", "Datetime", "Timestamp")), None)
    if date_col is None:
        raise ValueError("CSV needs a Date/Datetime/Timestamp column")
    df[date_col] = pd.to_datetime(df[date_col])
    return _clean(df.set_index(date_col))


def download(ticker: str, start: str = "2005-01-01", end: str = None) -> pd.DataFrame:
    """Download daily bars from Yahoo Finance (requires `pip install yfinance`)."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit("yfinance is not installed: pip install yfinance") from e
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"no data returned for {ticker!r}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _clean(df)


def synthetic(n_bars: int = 2520, seed: int = 7, drift: float = 0.08,
              vol: float = 0.18, mean_reversion: float = 0.15) -> pd.DataFrame:
    """Random-walk prices with mild short-term mean reversion.

    For smoke-testing only: results on synthetic data say nothing about real markets.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.standard_normal(n_bars) * vol * np.sqrt(dt)
    rets = np.empty(n_bars)
    prev = 0.0
    for i in range(n_bars):
        rets[i] = drift * dt - mean_reversion * prev + shocks[i]
        prev = rets[i]
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, vol * 0.1 * np.sqrt(dt), n_bars))
    span = np.abs(rng.normal(0, vol * 0.5 * np.sqrt(dt), n_bars))
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    idx = pd.bdate_range("2015-01-02", periods=n_bars)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(OHLC) - set(df.columns)
    if missing:
        raise ValueError(f"data is missing columns: {sorted(missing)}")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=OHLC)
