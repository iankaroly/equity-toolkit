"""
Discounted Cash Flow (DCF) valuation.

Usage:
    # Auto-fetch inputs from Yahoo, then value:
    python dcf.py AAPL

    # Override any assumption:
    python dcf.py AAPL --wacc 0.09 --terminal 0.025 --growth 0.08 --years 10

    # Fully manual (no network needed):
    python dcf.py --fcf 108e9 --shares 14.7e9 --net-debt -50e9 \
                  --wacc 0.09 --growth 0.08 --terminal 0.025

Method: project free cash flow for `years`, discount each to today at `wacc`,
add a Gordon-growth terminal value, subtract net debt, divide by shares.
Then print a WACC x terminal-growth sensitivity grid so you can see how much
the answer swings — a single fair-value number is false precision.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


# --- Assumptions you can reason about, all overridable ------------------------
RISK_FREE = 0.043     # ~10y Treasury, adjust as rates move
EQUITY_RISK_PREMIUM = 0.05


def wacc_from_capm(beta: float, rf: float = RISK_FREE, erp: float = EQUITY_RISK_PREMIUM) -> float:
    """Cost of equity via CAPM. A clean proxy for WACC for equity-heavy large caps."""
    return rf + beta * erp


@dataclass
class DCFInputs:
    fcf: float            # base free cash flow (year 0)
    shares: float
    wacc: float           # discount rate
    growth: float         # near-term annual FCF growth
    terminal: float       # perpetual growth after the projection window
    years: int = 10
    net_debt: float = 0.0  # subtracted from enterprise value; negative = net cash


def intrinsic_value(inp: DCFInputs) -> dict:
    if inp.terminal >= inp.wacc:
        raise ValueError(
            f"terminal growth ({inp.terminal:.1%}) must be < WACC ({inp.wacc:.1%}); "
            "the Gordon model diverges otherwise."
        )
    pv_fcf = 0.0
    fcf = inp.fcf
    for yr in range(1, inp.years + 1):
        fcf *= (1 + inp.growth)
        pv_fcf += fcf / (1 + inp.wacc) ** yr

    terminal_fcf = fcf * (1 + inp.terminal)
    terminal_value = terminal_fcf / (inp.wacc - inp.terminal)
    pv_terminal = terminal_value / (1 + inp.wacc) ** inp.years

    enterprise = pv_fcf + pv_terminal
    equity = enterprise - inp.net_debt
    per_share = equity / inp.shares
    return {
        "pv_fcf": pv_fcf,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise,
        "equity_value": equity,
        "fair_value_per_share": per_share,
        "terminal_pct_of_value": pv_terminal / enterprise,
    }


def implied_growth(inp: DCFInputs, target_price: float) -> float | None:
    """
    Reverse DCF: solve for the near-term FCF growth rate that makes fair value ==
    the market price. Answers 'what is the market actually assuming?' — far more
    honest than declaring a stock over/undervalued off one growth guess.
    Binary search on growth in [-20%, +40%].
    """
    lo, hi = -0.20, 0.40
    fv = lambda g: intrinsic_value(DCFInputs(**{**inp.__dict__, "growth": g}))["fair_value_per_share"]
    if fv(hi) < target_price:  # even 40% growth can't reach the price
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if fv(mid) < target_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def growth_sensitivity(inp: DCFInputs, wacc_step=0.01, span=2,
                       growths=(0.04, 0.08, 0.12, 0.16, 0.20)):
    """Fair value across WACC (rows) x near-term growth (cols) — the axis that
    actually moves the answer, so the grid brackets the market price."""
    waccs = [round(inp.wacc + wacc_step * d, 4) for d in range(-span, span + 1)]
    grid = []
    for w in waccs:
        row = [intrinsic_value(DCFInputs(**{**inp.__dict__, "wacc": w, "growth": g}))["fair_value_per_share"]
               for g in growths]
        grid.append(row)
    return waccs, list(growths), grid


def sensitivity(inp: DCFInputs, wacc_step=0.01, term_step=0.005, span=2):
    """Grid of fair value per share across WACC (rows) x terminal growth (cols)."""
    waccs = [round(inp.wacc + wacc_step * d, 4) for d in range(-span, span + 1)]
    terms = [round(inp.terminal + term_step * d, 4) for d in range(-span, span + 1)]
    grid = []
    for w in waccs:
        row = []
        for g in terms:
            try:
                v = intrinsic_value(DCFInputs(**{**inp.__dict__, "wacc": w, "terminal": g}))
                row.append(v["fair_value_per_share"])
            except ValueError:
                row.append(None)  # g >= w, undefined
        grid.append(row)
    return waccs, terms, grid


def _fmt(x):
    return "   n/a" if x is None else f"{x:7.2f}"


def print_report(inp: DCFInputs, price: float = None, name: str = None, ticker: str = None):
    res = intrinsic_value(inp)
    fv = res["fair_value_per_share"]
    title = f"{name or ticker or 'Manual DCF'}"
    print("\n" + "=" * 60)
    print(f"  DCF VALUATION — {title}")
    print("=" * 60)
    print(f"  Base FCF ............ {inp.fcf/1e9:,.2f} B")
    print(f"  Growth (near-term) .. {inp.growth:.1%}  for {inp.years} yrs")
    print(f"  Terminal growth ..... {inp.terminal:.1%}")
    print(f"  WACC (discount) ..... {inp.wacc:.1%}")
    print(f"  Net debt ............ {inp.net_debt/1e9:,.2f} B")
    print(f"  Shares .............. {inp.shares/1e9:,.3f} B")
    print("-" * 60)
    print(f"  Enterprise value .... {res['enterprise_value']/1e9:,.1f} B")
    print(f"  Terminal % of value . {res['terminal_pct_of_value']:.0%}"
          + ("   <-- heavy; result leans on the perpetuity" if res['terminal_pct_of_value'] > 0.75 else ""))
    print(f"  Equity value ........ {res['equity_value']/1e9:,.1f} B")
    print(f"\n  FAIR VALUE / SHARE .. ${fv:,.2f}")
    if price:
        gap = fv / price - 1
        verdict = "UNDERVALUED" if gap > 0.15 else "OVERVALUED" if gap < -0.15 else "ROUGHLY FAIR"
        print(f"  Market price ........ ${price:,.2f}")
        print(f"  Upside/(downside) ... {gap:+.0%}   -> {verdict} at {inp.growth:.0%} growth")

        # Reverse DCF: what growth does the market price imply?
        ig = implied_growth(inp, price)
        if ig is None:
            print(f"  Market-implied FCF growth: >40%/yr — price is off this model's grid")
        else:
            print(f"  Market-implied FCF growth: {ig:.1%}/yr for {inp.years} yrs "
                  f"(you assumed {inp.growth:.0%})")

    # WACC x GROWTH grid — brackets the price, since growth dominates the answer
    waccs, growths, ggrid = growth_sensitivity(inp)
    print("\n  SENSITIVITY — fair value / share  (rows=WACC, cols=near-term growth)")
    print("        " + "".join(f"{g:>8.0%}" for g in growths))
    for w, row in zip(waccs, ggrid):
        print(f"  {w:5.1%} " + "".join(_fmt(v) for v in row))

    # Secondary: terminal-growth sensitivity at the base growth
    waccs, terms, grid = sensitivity(inp)
    print("\n  SENSITIVITY — fair value / share  (rows=WACC, cols=terminal g)")
    print("        " + "".join(f"{g:>8.1%}" for g in terms))
    for w, row in zip(waccs, grid):
        print(f"  {w:5.1%} " + "".join(_fmt(v) for v in row))
    print("=" * 60 + "\n")
    return res


def _build_inputs(args) -> tuple:
    price, name = args.price, None
    if args.ticker:
        from data import get_fundamentals
        f = get_fundamentals(args.ticker)
        name, price = f.name, (args.price or f.price)
        fcf = args.fcf if args.fcf is not None else f.fcf
        shares = args.shares if args.shares is not None else f.shares
        net_debt = args.net_debt if args.net_debt is not None else (f.net_debt or 0.0)
        wacc = args.wacc if args.wacc is not None else (
            wacc_from_capm(f.beta) if f.beta else 0.09)
        if None in (fcf, shares):
            raise SystemExit(
                f"Could not auto-fetch {'FCF' if fcf is None else 'shares'} for {args.ticker}. "
                "Pass it manually, e.g. --fcf 108e9 --shares 14.7e9")
    else:
        fcf, shares = args.fcf, args.shares
        net_debt = args.net_debt or 0.0
        wacc = args.wacc if args.wacc is not None else 0.09
        if None in (fcf, shares):
            raise SystemExit("Manual mode needs at least --fcf and --shares.")

    inp = DCFInputs(
        fcf=fcf, shares=shares, wacc=wacc,
        growth=args.growth, terminal=args.terminal,
        years=args.years, net_debt=net_debt,
    )
    return inp, price, name


def main():
    p = argparse.ArgumentParser(description="DCF valuation with sensitivity table")
    p.add_argument("ticker", nargs="?", help="ticker to auto-fetch (optional)")
    p.add_argument("--fcf", type=float, help="base free cash flow, e.g. 108e9")
    p.add_argument("--shares", type=float, help="shares outstanding, e.g. 14.7e9")
    p.add_argument("--net-debt", type=float, dest="net_debt", help="total debt - cash")
    p.add_argument("--price", type=float, help="override market price for the gap calc")
    p.add_argument("--wacc", type=float, help="discount rate (default: CAPM or 0.09)")
    p.add_argument("--growth", type=float, default=0.08, help="near-term FCF growth (0.08)")
    p.add_argument("--terminal", type=float, default=0.025, help="perpetual growth (0.025)")
    p.add_argument("--years", type=int, default=10, help="projection years (10)")
    args = p.parse_args()

    inp, price, name = _build_inputs(args)
    print_report(inp, price=price, name=name, ticker=args.ticker)


if __name__ == "__main__":
    main()
