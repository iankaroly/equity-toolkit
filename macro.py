"""
Macro snapshot: the rate / curve / volatility / commodity backdrop, live.

Pulls the indicators that actually move a portfolio's discount rate and risk
appetite, labels the regime, and suggests a DCF risk-free rate from the current
10-yr. Meant to open every analysis so advice is grounded in what's happening
now — not stale narrative.

    python macro.py            # print the dashboard
    python macro.py --json     # machine-readable

Also importable: macro.snapshot() -> dict, macro.snapshot_text() -> str.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

# symbol -> (label, kind)  kind: 'yield' (already %), 'level', 'price'
INDICATORS = {
    "^IRX": ("3-mo T-bill", "yield"),
    "^FVX": ("5-yr Treasury", "yield"),
    "^TNX": ("10-yr Treasury", "yield"),
    "^TYX": ("30-yr Treasury", "yield"),
    "^VIX": ("VIX (volatility)", "level"),
    "CL=F": ("WTI oil", "price"),
    "GC=F": ("Gold", "price"),
    "DX-Y.NYB": ("Dollar index", "price"),
    "^GSPC": ("S&P 500", "price"),
}


def _pull_one(tk):
    import yfinance as yf
    h = yf.Ticker(tk).history(period="4mo")["Close"].dropna()
    if len(h) < 2:
        return None
    now = float(h.iloc[-1])
    mo1 = float(h.iloc[-22]) if len(h) > 22 else float(h.iloc[0])
    mo3 = float(h.iloc[0])
    return {"now": now, "chg_1mo": now - mo1, "chg_3mo": now - mo3,
            "pct_3mo": (now / mo3 - 1) if mo3 else 0.0}


def snapshot() -> dict:
    data = {}
    for tk in INDICATORS:
        try:
            d = _pull_one(tk)
            if d:
                data[tk] = d
        except Exception:
            pass

    out = {"raw": data, "flags": {}}
    g = lambda tk: data[tk]["now"] if tk in data else None

    # Rate regime
    tnx = g("^TNX")
    if tnx is not None:
        chg = data["^TNX"]["chg_3mo"]
        out["flags"]["rates"] = ("rising" if chg > 0.15 else "falling" if chg < -0.15 else "flat")
        out["suggested_risk_free"] = round(tnx / 100, 4)  # for DCF WACC

    # Yield curve 10y - 3mo
    if g("^TNX") is not None and g("^IRX") is not None:
        slope = g("^TNX") - g("^IRX")
        out["curve_slope"] = slope
        out["flags"]["curve"] = ("normal" if slope > 0.2 else "flat" if slope > -0.2 else "INVERTED")

    # Volatility regime
    vix = g("^VIX")
    if vix is not None:
        out["flags"]["vol"] = ("calm/complacent" if vix < 15 else "normal" if vix < 20
                               else "elevated" if vix < 30 else "STRESSED")
    return out


def _fmt_row(label, d, kind):
    now, c3 = d["now"], d["chg_3mo"]
    if kind == "yield":
        return f"  {label:<18}{now:>8.2f}%   {c3:>+6.2f} pts 3mo"
    if kind == "level":
        return f"  {label:<18}{now:>8.1f}    {c3:>+6.1f}     3mo"
    return f"  {label:<18}{now:>8,.0f}    {d['pct_3mo']:>+6.1%}     3mo"


def snapshot_text() -> str:
    import datetime as dt
    s = snapshot()
    data = s["raw"]
    L = [f"\n  MACRO SNAPSHOT  ({dt.date.today()})", "  " + "=" * 46]
    L.append("  RATES & CURVE")
    for tk in ("^IRX", "^FVX", "^TNX", "^TYX"):
        if tk in data:
            L.append(_fmt_row(INDICATORS[tk][0], data[tk], "yield"))
    if "curve_slope" in s:
        L.append(f"  {'Curve 10y-3mo':<18}{s['curve_slope']:>+8.2f} pts   [{s['flags'].get('curve','')}]")
    L.append("  RISK & COMMODITIES")
    for tk in ("^VIX", "CL=F", "GC=F", "DX-Y.NYB", "^GSPC"):
        if tk in data:
            L.append(_fmt_row(INDICATORS[tk][0], data[tk], INDICATORS[tk][1]))
    L.append("  " + "-" * 46)
    fl = s["flags"]
    L.append(f"  Regime: rates {fl.get('rates','?')} | curve {fl.get('curve','?')} "
             f"| vol {fl.get('vol','?')}")
    if "suggested_risk_free" in s:
        L.append(f"  Suggested DCF risk-free (current 10y): {s['suggested_risk_free']:.2%}")

    # One-line takeaway
    bits = []
    if fl.get("rates") == "rising":
        bits.append("rising rates pressure long-duration/growth & lift the discount rate")
    if fl.get("vol") in ("calm/complacent",):
        bits.append("low VIX = complacency, don't chase froth")
    if fl.get("curve") == "INVERTED":
        bits.append("inverted curve = recession warning")
    if bits:
        L.append("  Read: " + "; ".join(bits) + ".")
    L.append("  (Levels are live; policy path/why is not forecast here.)\n")
    return "\n".join(L)


def main():
    import sys
    if "--json" in sys.argv:
        import json
        print(json.dumps(snapshot(), indent=2, default=float))
    else:
        print(snapshot_text())


if __name__ == "__main__":
    main()
