#!/usr/bin/env python3
"""
stress.py — how far does the current book fall in a real drawdown?

Sharpe ratios and trailing returns are measured in a bull market and hide the
downside. This tool shows two honest things:
  1. EMPIRICAL — the worst drawdown the current holdings (at today's weights)
     would actually have suffered over the lookback window. No assumptions.
  2. PARAMETRIC — the book's beta to the S&P, then estimated loss under
     market shocks of -10% / -20% / -35% (correction / bear / crash), in dollars.

    ~/investing/.venv/bin/python ~/investing/stress.py
    ~/investing/.venv/bin/python ~/investing/stress.py --period 5y
"""
import argparse
import json
import os

import numpy as np

from data import get_prices

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = "SPY"


def money(x):
    return f"${x:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", default=os.path.join(HERE, "holdings.json"))
    ap.add_argument("--period", default="3y")
    args = ap.parse_args()

    doc = json.load(open(args.holdings))
    positions = doc["positions"]
    cash = doc.get("cash", 0.0)
    shares = {t: p["shares"] for t, p in positions.items()}

    px = get_prices(list(shares) + [BENCH], period=args.period)
    if px.empty or BENCH not in px.columns:
        raise SystemExit("Could not fetch prices for the stress test.")

    held = [t for t in shares if t in px.columns]
    latest = px[held].iloc[-1]
    values = {t: shares[t] * float(latest[t]) for t in held}
    equity = sum(values.values())
    total = equity + cash

    # Daily returns; book return = value-weighted across whatever data exists each day
    rets = px[held + [BENCH]].pct_change().dropna(how="all")
    w = np.array([values[t] / equity for t in held])
    book = (rets[held].fillna(0.0) @ w)
    spy = rets[BENCH]
    common = book.index.intersection(spy.index)
    book, spy = book.loc[common], spy.loc[common]

    # --- beta of the equity sleeve to SPY ---
    var = float(np.var(spy))
    beta = float(np.cov(book, spy)[0, 1] / var) if var else float("nan")

    # --- empirical worst drawdown of the current book over the window ---
    curve = (1 + book).cumprod()
    dd = curve / curve.cummax() - 1
    max_dd = float(dd.min())
    worst_day = float(book.min())
    worst_month = float(book.rolling(21).sum().min())
    worst_qtr = float(book.rolling(63).sum().min())

    # cash cushions the % move: equity takes the hit, cash doesn't
    eq_frac = equity / total

    print("\n" + "=" * 64)
    print("  PORTFOLIO STRESS TEST")
    print("=" * 64)
    print(f"  Book {money(total)}  ·  equity {money(equity)} ({eq_frac:.0%})  ·  cash {money(cash)}")
    print(f"  Equity beta to S&P: {beta:.2f}  (>1 = falls harder than the market)")
    print(f"  Window: {str(px.index[0])[:10]} -> {str(px.index[-1])[:10]} ({args.period})")

    print("\n  WHAT ACTUALLY HAPPENED (this book, these weights, past {}):".format(args.period))
    print(f"    Worst peak-to-trough drawdown ... {max_dd:>7.1%}   ({money(max_dd*equity)})")
    print(f"    Worst single day ................ {worst_day:>7.1%}   ({money(worst_day*equity)})")
    print(f"    Worst month (21d) ............... {worst_month:>7.1%}   ({money(worst_month*equity)})")
    print(f"    Worst quarter (63d) ............. {worst_qtr:>7.1%}   ({money(worst_qtr*equity)})")

    print("\n  IF THE MARKET DROPS (beta-estimated book loss, in dollars):")
    print(f"    {'S&P shock':<16}{'book move':>12}{'$ loss':>16}{'new value':>16}")
    for shock in (-0.10, -0.20, -0.35):
        move = beta * shock                 # first-order estimate
        loss = move * equity
        print(f"    {shock:>+7.0%} ({'correction' if shock>-0.15 else 'bear' if shock>-0.30 else 'crash'})"
              f"{'':>1}{move:>+11.1%}{money(loss):>16}{money(total+loss):>16}")

    print("\n  Note: parametric = beta x shock, a first-order estimate. High-beta")
    print("  concentration (semis, mega-cap tech) means real crashes hit tech-heavy")
    print("  books HARDER than beta implies. Treat these as optimistic floors.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
