"""
Portfolio analyzer: risk, return, diversification, and rebalancing.

Usage:
    # Analyze holdings (shares) — auto-fetches prices:
    python portfolio.py --holdings AAPL:50 MSFT:20 KO:100 --cash 5000

    # Or target weights, and get rebalancing trades to a total value:
    python portfolio.py --holdings AAPL:50 MSFT:20 KO:100 \
        --target AAPL:0.4 MSFT:0.4 KO:0.2

Metrics use ~1y of daily returns: annualized return, volatility, Sharpe,
max drawdown, and the correlation matrix (low correlations = real
diversification; a wall of 0.8s means you own one bet in five costumes).
"""
from __future__ import annotations

import argparse
import numpy as np

TRADING_DAYS = 252
RISK_FREE = 0.043


def _parse_pairs(items, cast=float):
    out = {}
    for it in items or []:
        k, v = it.split(":")
        out[k.upper()] = cast(v)
    return out


def analyze(holdings: dict, cash: float = 0.0, period: str = "1y"):
    """holdings = {ticker: shares}. Returns a dict of positions + portfolio metrics."""
    from data import get_prices

    tickers = list(holdings)
    prices = get_prices(tickers, period=period)
    if prices.empty:
        raise SystemExit("Could not fetch prices. Check tickers / connectivity.")
    prices = prices[[t for t in tickers if t in prices.columns]].dropna()
    latest = prices.iloc[-1]

    # Position values and weights
    values = {t: holdings[t] * float(latest[t]) for t in prices.columns}
    invested = sum(values.values())
    total = invested + cash
    weights = {t: values[t] / total for t in values}

    # Return series and portfolio-level stats
    rets = prices.pct_change().dropna()
    w = np.array([values[t] / invested for t in prices.columns])  # weights of invested sleeve
    port_ret = rets[list(prices.columns)] @ w

    ann_return = (1 + port_ret.mean()) ** TRADING_DAYS - 1
    ann_vol = port_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - RISK_FREE) / ann_vol if ann_vol else float("nan")

    curve = (1 + port_ret).cumprod()
    drawdown = (curve / curve.cummax() - 1).min()

    corr = rets.corr()
    return {
        "values": values, "weights": weights, "cash": cash,
        "invested": invested, "total": total,
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "max_drawdown": float(drawdown), "corr": corr, "latest": latest,
    }


def rebalance(latest, target: dict, total_value: float):
    """Trades (in $ and approx shares) to reach target weights for a total value."""
    trades = {}
    for t, wgt in target.items():
        px = float(latest[t])
        dollars = wgt * total_value
        trades[t] = {"target_$": dollars, "target_shares": dollars / px, "price": px}
    return trades


def print_report(res: dict, target: dict = None):
    print("\n" + "=" * 60)
    print("  PORTFOLIO ANALYSIS")
    print("=" * 60)
    print(f"  {'Ticker':<8}{'Value $':>14}{'Weight':>10}")
    for t, v in sorted(res["values"].items(), key=lambda kv: -kv[1]):
        print(f"  {t:<8}{v:>14,.0f}{res['weights'][t]:>10.1%}")
    if res["cash"]:
        print(f"  {'CASH':<8}{res['cash']:>14,.0f}{res['cash']/res['total']:>10.1%}")
    print("-" * 60)
    print(f"  Total value ......... ${res['total']:,.0f}")
    print(f"  Annualized return ... {res['ann_return']:+.1%}")
    print(f"  Volatility .......... {res['ann_vol']:.1%}")
    print(f"  Sharpe ratio ........ {res['sharpe']:.2f}"
          + ("   (>1 good, <0.5 weak risk-adjusted)"))
    print(f"  Max drawdown ........ {res['max_drawdown']:.1%}")

    # Concentration flag
    top = max(res["weights"].values())
    if top > 0.35:
        big = max(res["weights"], key=res["weights"].get)
        print(f"  ! Concentrated: {big} is {top:.0%} of the book.")

    print("\n  CORRELATION (1.0 = moves identically; lower = more diversified)")
    corr = res["corr"].round(2)
    cols = list(corr.columns)
    print("        " + "".join(f"{c:>7}" for c in cols))
    for r in cols:
        print(f"  {r:<6}" + "".join(f"{corr.loc[r, c]:>7.2f}" for c in cols))
    avg_off = (corr.values.sum() - len(cols)) / (len(cols) ** 2 - len(cols)) if len(cols) > 1 else 0
    print(f"  Avg pairwise correlation: {avg_off:.2f}"
          + ("   <-- high; limited diversification" if avg_off > 0.7 else ""))

    if target:
        print("\n  REBALANCE TO TARGET")
        trades = rebalance(res["latest"], target, res["total"])
        print(f"  {'Ticker':<8}{'Target $':>14}{'Target sh':>12}")
        for t, tr in trades.items():
            print(f"  {t:<8}{tr['target_$']:>14,.0f}{tr['target_shares']:>12,.1f}")
    print("=" * 60 + "\n")


def main():
    p = argparse.ArgumentParser(description="Portfolio risk/return + rebalancing")
    p.add_argument("--holdings", nargs="+", required=True,
                   help="TICKER:SHARES pairs, e.g. AAPL:50 MSFT:20")
    p.add_argument("--cash", type=float, default=0.0)
    p.add_argument("--target", nargs="+", help="TICKER:WEIGHT pairs summing to ~1.0")
    p.add_argument("--period", default="1y")
    args = p.parse_args()

    holdings = _parse_pairs(args.holdings, cast=float)
    target = _parse_pairs(args.target) if args.target else None
    if target and abs(sum(target.values()) - 1) > 0.02:
        raise SystemExit(f"Target weights sum to {sum(target.values()):.2f}, expected ~1.0")

    res = analyze(holdings, cash=args.cash, period=args.period)
    print_report(res, target=target)


if __name__ == "__main__":
    main()
