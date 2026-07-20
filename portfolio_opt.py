"""
Portfolio optimizer (mean-variance + robust variants).

Usage:
    # optimize your current holdings, cap any position at 15%:
    python portfolio_opt.py --holdings KMI:161.28 GOOG:15 ... --cap 0.15

    # or just a ticker list (equal starting point):
    python portfolio_opt.py --tickers GOOG MSFT KO SCHD SGOV --cap 0.25

Computes four portfolios and compares them to what you hold today:
  - CURRENT      : your actual weights (the baseline to beat)
  - MIN-VOL       : lowest-risk mix (robust — needs no return forecast)
  - RISK-PARITY   : each holding contributes equal risk (robust, well-diversified)
  - MAX-SHARPE    : best historical risk-adjusted return (FRAGILE — see warning)

Honest health warning printed at the end: max-Sharpe over-fits to whichever
assets happened to rise in the lookback window. Trust MIN-VOL / RISK-PARITY.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
TRADING_DAYS = 252
RISK_FREE = 0.043


def _perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - RISK_FREE) / vol if vol else float("nan")
    return ret, vol, sharpe


def _solve(objective, n, cap, extra_constraints=()):
    bounds = tuple((0.0, cap) for _ in range(n))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
    x0 = np.repeat(1.0 / n, n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-9})
    return res.x


def min_vol(mu, cov, cap):
    return _solve(lambda w: w @ cov @ w, len(mu), cap)


def max_sharpe(mu, cov, cap):
    def neg_sharpe(w):
        r, v, _ = _perf(w, mu, cov)
        return -(r - RISK_FREE) / v if v else 1e9
    return _solve(neg_sharpe, len(mu), cap)


def risk_parity(cov, cap):
    """Weights so each asset contributes equal risk. Robust; ignores return guesses."""
    n = len(cov)

    def obj(w):
        port_var = w @ cov @ w
        mrc = cov @ w                      # marginal risk contribution
        rc = w * mrc                       # risk contribution per asset
        target = port_var / n
        return np.sum((rc - target) ** 2)

    return _solve(obj, n, cap)


def optimize(prices: pd.DataFrame, current_w: dict | None = None, cap: float = 0.15):
    rets = prices.pct_change().dropna()
    tickers = list(prices.columns)
    mu = rets.mean().values * TRADING_DAYS
    cov = rets.cov().values * TRADING_DAYS

    ports = {
        "MIN-VOL": min_vol(mu, cov, cap),
        "RISK-PARITY": risk_parity(cov, cap),
        "MAX-SHARPE": max_sharpe(mu, cov, cap),
    }
    out = {"tickers": tickers, "mu": mu, "cov": cov, "ports": {}, "window": (rets.index[0], rets.index[-1])}
    for name, w in ports.items():
        r, v, s = _perf(w, mu, cov)
        out["ports"][name] = {"w": w, "ret": r, "vol": v, "sharpe": s}
    if current_w:
        cw = np.array([current_w.get(t, 0.0) for t in tickers])
        cw = cw / cw.sum()
        r, v, s = _perf(cw, mu, cov)
        out["ports"]["CURRENT"] = {"w": cw, "ret": r, "vol": v, "sharpe": s}
    return out


def print_report(out, cap):
    tickers = out["tickers"]
    order = [k for k in ("CURRENT", "MIN-VOL", "RISK-PARITY", "MAX-SHARPE") if k in out["ports"]]
    w0, w1 = out["window"]
    print("\n" + "=" * 74)
    print(f"  PORTFOLIO OPTIMIZER   (window {str(w0)[:10]} -> {str(w1)[:10]}, cap {cap:.0%}/name)")
    print("=" * 74)

    # weights table
    print(f"  {'Ticker':<8}" + "".join(f"{k:>13}" for k in order))
    for i, t in enumerate(tickers):
        row = "".join(f"{out['ports'][k]['w'][i]:>12.1%} " for k in order)
        print(f"  {t:<8}" + row)
    print("  " + "-" * 70)
    for metric, label, fmt in [("ret", "Exp. return", "{:>11.1%} "),
                               ("vol", "Volatility ", "{:>11.1%} "),
                               ("sharpe", "Sharpe     ", "{:>11.2f} ")]:
        print(f"  {label:<8}" + "".join(fmt.format(out['ports'][k][metric]) for k in order))
    print("=" * 74)
    print("  MIN-VOL / RISK-PARITY are robust (no return forecast needed).")
    print("  !! MAX-SHARPE is FRAGILE: it over-weights whatever rose in this window")
    print("     and rarely repeats out-of-sample. Treat it as an upper bound, not a target.")
    print("  Expected return uses historical means — a weak predictor. Weigh the VOL and")
    print("  diversification more than the return number.\n")


def _parse_holdings(items):
    return {a.split(":")[0].upper(): float(a.split(":")[1]) for a in items}


def main():
    p = argparse.ArgumentParser(description="Mean-variance + robust portfolio optimizer")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--holdings", nargs="+", help="TICKER:SHARES (also computes CURRENT weights)")
    g.add_argument("--tickers", nargs="+", help="just a ticker universe")
    p.add_argument("--cap", type=float, default=0.15, help="max weight per name (0.15)")
    p.add_argument("--period", default="3y", help="lookback (default 3y)")
    args = p.parse_args()

    from data import get_prices
    if args.holdings:
        holds = _parse_holdings(args.holdings)
        prices = get_prices(list(holds), period=args.period).dropna()
        latest = prices.iloc[-1]
        current_val = {t: holds[t] * float(latest[t]) for t in prices.columns}
        total = sum(current_val.values())
        current_w = {t: current_val[t] / total for t in current_val}
    else:
        prices = get_prices([t.upper() for t in args.tickers], period=args.period).dropna()
        current_w = None

    if prices.empty or prices.shape[1] < 2:
        raise SystemExit("Need at least 2 tickers with overlapping price history.")
    out = optimize(prices, current_w=current_w, cap=args.cap)
    print_report(out, args.cap)


if __name__ == "__main__":
    main()
