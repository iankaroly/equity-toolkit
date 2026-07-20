"""
Single-ticker quant analysis: return/risk stats, beta, momentum, technicals.

    python quant.py UUUU
    python quant.py UUUU --bench SPY --period 3y

Not a valuation (use dcf.py for that) — this is the statistical/technical read:
how volatile, how it moves vs the market, momentum across horizons, and where
it sits vs its moving averages and 52-week range.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")
TRADING_DAYS = 252
RISK_FREE = 0.043


def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn
    return float((100 - 100 / (1 + rs)).iloc[-1])


def analyze(ticker, bench="SPY", period="3y"):
    from data import get_prices, get_fundamentals

    px = get_prices([ticker, bench], period=period).dropna()
    s = px[ticker]
    b = px[bench]
    r = s.pct_change().dropna()
    rb = b.pct_change().dropna()

    out = {"ticker": ticker, "bench": bench, "last": float(s.iloc[-1]),
           "window": (px.index[0], px.index[-1])}

    # risk/return over the window and trailing 1y
    def stats(ser):
        ann_ret = (1 + ser.mean()) ** TRADING_DAYS - 1
        ann_vol = ser.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol else float("nan")
        curve = (1 + ser).cumprod()
        dd = float((curve / curve.cummax() - 1).min())
        return ann_ret, ann_vol, sharpe, dd

    out["full"] = stats(r)
    out["y1"] = stats(r.iloc[-TRADING_DAYS:]) if len(r) > TRADING_DAYS else out["full"]

    # beta & correlation vs benchmark
    aligned = np.column_stack([r.values, rb.values])
    cov = np.cov(aligned.T)
    out["beta"] = cov[0, 1] / cov[1, 1]
    out["corr"] = np.corrcoef(aligned.T)[0, 1]

    # momentum across horizons (price return)
    out["mom"] = {}
    for label, d in [("1mo", 21), ("3mo", 63), ("6mo", 126), ("12mo", 252)]:
        if len(s) > d:
            out["mom"][label] = float(s.iloc[-1] / s.iloc[-d] - 1)

    # technicals
    ma50 = s.rolling(50).mean().iloc[-1]
    ma200 = s.rolling(200).mean().iloc[-1]
    out["ma50"], out["ma200"] = float(ma50), float(ma200)
    out["vs_ma50"] = float(s.iloc[-1] / ma50 - 1)
    out["vs_ma200"] = float(s.iloc[-1] / ma200 - 1)
    out["trend"] = "up (50>200)" if ma50 > ma200 else "down (50<200)"
    out["rsi14"] = rsi(s)
    hi = float(s.iloc[-252:].max()); lo = float(s.iloc[-252:].min())
    out["hi52"], out["lo52"] = hi, lo
    out["from_hi"] = float(s.iloc[-1] / hi - 1)
    out["from_lo"] = float(s.iloc[-1] / lo - 1)

    # fundamentals (is there any cash flow?)
    out["f"] = get_fundamentals(ticker)
    return out


def print_report(o):
    f = o["f"]
    w0, w1 = o["window"]
    fr, fv, fs, fdd = o["full"]
    y = o["y1"]
    print("\n" + "=" * 60)
    print(f"  QUANT ANALYSIS — {f.name or o['ticker']} ({o['ticker']})   ${o['last']:.2f}")
    print(f"  window {str(w0)[:10]} -> {str(w1)[:10]}, vs {o['bench']}")
    print("=" * 60)
    print("  RISK / RETURN            trailing-1y      full-window")
    print(f"    Annualized return    {y[0]:>10.1%}      {fr:>10.1%}")
    print(f"    Volatility           {y[1]:>10.1%}      {fv:>10.1%}")
    print(f"    Sharpe               {y[2]:>10.2f}      {fs:>10.2f}")
    print(f"    Max drawdown         {y[3]:>10.1%}      {fdd:>10.1%}")
    print(f"\n  MARKET SENSITIVITY (vs {o['bench']})")
    print(f"    Beta                 {o['beta']:>10.2f}   ({'high' if o['beta']>1.5 else 'moderate' if o['beta']>0.9 else 'low'}-beta)")
    print(f"    Correlation          {o['corr']:>10.2f}")
    print("\n  MOMENTUM (price return)")
    for k in ("1mo", "3mo", "6mo", "12mo"):
        if k in o["mom"]:
            print(f"    {k:<5}                {o['mom'][k]:>+10.1%}")
    print("\n  TREND / TECHNICALS")
    print(f"    50d MA ${o['ma50']:.2f} | 200d MA ${o['ma200']:.2f}  -> trend {o['trend']}")
    print(f"    Price vs 50d/200d    {o['vs_ma50']:+.1%} / {o['vs_ma200']:+.1%}")
    rlabel = "overbought" if o['rsi14'] > 70 else "oversold" if o['rsi14'] < 30 else "neutral"
    print(f"    RSI(14)              {o['rsi14']:>10.1f}   ({rlabel})")
    print(f"    52-wk range          ${o['lo52']:.2f} - ${o['hi52']:.2f}")
    print(f"    From 52w high/low    {o['from_hi']:+.1%} / {o['from_lo']:+.1%}")
    print("\n  FUNDAMENTALS")
    fcf = f.fcf
    print(f"    Price {f.price} | shares {f.shares} | beta(reported) {f.beta}")
    print(f"    Free cash flow: {('$'+format(fcf/1e9,'.2f')+'B') if fcf else 'negative/none'}"
          + ("   -> not valuable via DCF; a speculative/story stock" if not fcf or fcf <= 0 else ""))
    print("=" * 60 + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("--bench", default="SPY")
    p.add_argument("--period", default="3y")
    args = p.parse_args()
    print_report(analyze(args.ticker.upper(), args.bench.upper(), args.period))


if __name__ == "__main__":
    main()
