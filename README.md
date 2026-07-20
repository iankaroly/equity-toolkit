# 📈 Equity Valuation & Portfolio Toolkit

A Python toolkit for **equity valuation and portfolio risk analysis on live market data** — discounted-cash-flow (DCF) intrinsic value, portfolio risk/return metrics, mean-variance optimization, single-name quant analysis, a macro dashboard, and crash stress-testing. Runs from the command line or an interactive Streamlit web app.

Built to answer one question with real models instead of guesswork: *is this stock worth buying, and is this portfolio well-built?*

![CI](https://github.com/USERNAME/equity-toolkit/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

| Module | What it does |
|---|---|
| **`dcf.py`** | DCF intrinsic value with a **WACC × terminal-growth sensitivity grid** and a reverse-DCF (the growth rate the current price implies). |
| **`portfolio.py`** | Weights, annualized return, volatility, **Sharpe**, max drawdown, and a full **correlation matrix** with concentration flags and rebalancing trades. |
| **`portfolio_opt.py`** | Mean-variance optimizer: **min-volatility, risk-parity, and max-Sharpe** target weights (with an honest warning that max-Sharpe overfits). |
| **`quant.py`** | Per-name **beta, momentum, RSI, moving-average trend**, drawdown, and 52-week range vs. a benchmark. |
| **`macro.py`** | Live macro dashboard: Treasury yields, **yield-curve slope, VIX, oil, gold, the dollar, S&P** — with a regime label. |
| **`stress.py`** | **Crash stress test** — the book's real historical drawdown plus beta-estimated losses under -10% / -20% / -35% market shocks, in dollars. |

All market data is pulled live from Yahoo Finance via `yfinance`. Every model also accepts fully manual inputs, so nothing hard-depends on a network call.

---

## Quickstart

```bash
git clone https://github.com/USERNAME/equity-toolkit.git
cd equity-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Command line

```bash
# DCF valuation with a sensitivity grid (auto-fetches financials)
python dcf.py AAPL --growth 0.08 --terminal 0.025 --wacc 0.09

# Portfolio risk/return + correlation matrix
python portfolio.py --holdings AAPL:50 MSFT:20 KO:100 --cash 5000

# Live macro backdrop
python macro.py

# Single-name quant read
python quant.py NVDA

# Crash stress test on a set of holdings
python stress.py --holdings AAPL:50 MSFT:20 NVDA:10
```

### Interactive web app

```bash
streamlit run app.py
```

An interactive dashboard: value any stock with live sliders, analyze a portfolio's risk and correlations, and read the macro tape — all in the browser.

---

## Example output

```
  DCF — Apple Inc. (AAPL)   market $195.00
  Fair value / share     $182.40      downside (6.5%)
  Implied growth @ price   9.8%

  WACC × TERMINAL GROWTH SENSITIVITY  (fair value / share)
              2.0%     2.5%     3.0%
   8.0%    $201     $214     $231
   9.0%    $172     $182     $194
  10.0%    $150     $158     $167
```

---

## Tech stack
Python · pandas · NumPy · SciPy · yfinance · Streamlit · pytest · GitHub Actions

## Disclaimer
This is an educational analysis tool, **not financial advice**. Models rely on assumptions (growth, discount rate, historical data) that may not hold. Do your own research.

## License
MIT
