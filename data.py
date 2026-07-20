"""
Shared market-data layer.

Design rule (deliberate): every fetch here is a *convenience*. The DCF and
portfolio tools always accept manual numbers, so if Yahoo throttles us or a
ticker is missing a field, you can still run the model by typing the inputs.
Nothing in this file is allowed to be a hard dependency for producing a result.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")  # silence LibreSSL / yfinance chatter


def _retry(fn, tries: int = 3, pause: float = 1.5):
    """Yahoo occasionally 429s. Small backoff; give up gracefully (return None)."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - we intentionally swallow and retry
            last = e
            time.sleep(pause * (i + 1))
    print(f"  [data] fetch failed after {tries} tries: {type(last).__name__}: {str(last)[:80]}")
    return None


@dataclass
class Fundamentals:
    ticker: str
    price: Optional[float] = None
    shares: Optional[float] = None
    fcf: Optional[float] = None          # most recent annual free cash flow
    fcf_history: Optional[list] = None   # oldest..newest, for growth sanity checks
    net_debt: Optional[float] = None     # total debt - cash (neg = net cash)
    beta: Optional[float] = None
    name: Optional[str] = None

    def missing(self) -> list:
        return [k for k in ("price", "shares", "fcf") if getattr(self, k) is None]


def get_prices(tickers, period: str = "1y", interval: str = "1d"):
    """Return a DataFrame of adjusted close prices (columns = tickers). Empty on failure."""
    import pandas as pd
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]

    def _pull():
        raw = yf.download(
            tickers, period=period, interval=interval,
            auto_adjust=True, progress=False, threads=True,
        )
        # yfinance shape differs for 1 vs many tickers; normalize to Close matrix
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        return close.dropna(how="all")

    df = _retry(_pull)
    return df if df is not None else pd.DataFrame()


def get_fundamentals(ticker: str) -> Fundamentals:
    """Pull the inputs a DCF needs. Any field may come back None — that's fine."""
    import yfinance as yf

    tk = yf.Ticker(ticker)
    info = _retry(lambda: tk.info) or {}
    f = Fundamentals(ticker=ticker.upper())
    f.name = info.get("longName") or info.get("shortName")
    f.price = info.get("currentPrice") or info.get("regularMarketPrice")
    f.shares = info.get("sharesOutstanding")
    f.beta = info.get("beta")

    # Free cash flow history from the cashflow statement (preferred, clean)
    cf = _retry(lambda: tk.cashflow)
    if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
        series = cf.loc["Free Cash Flow"].dropna()
        if len(series):
            vals = [float(v) for v in series.values][::-1]  # oldest..newest
            f.fcf_history = vals
            f.fcf = vals[-1]

    # Net debt = total debt - cash. Best effort from info.
    debt = info.get("totalDebt")
    cash = info.get("totalCash")
    if debt is not None and cash is not None:
        f.net_debt = float(debt) - float(cash)

    return f


if __name__ == "__main__":  # quick smoke test
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    f = get_fundamentals(t)
    print(f)
    print("missing:", f.missing())
