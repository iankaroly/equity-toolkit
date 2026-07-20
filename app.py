"""
Streamlit web app for the Equity Valuation & Portfolio Toolkit.

Run locally:  streamlit run app.py
Deploy free:  push to GitHub -> streamlit.io/cloud -> point at app.py

The app wraps the same modules used on the command line. DCF and macro reports
are rendered from each tool's formatted output; the portfolio view calls
portfolio.analyze() directly to draw native metric cards and a correlation heatmap.
"""
import io
import subprocess
import sys
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Equity Toolkit", layout="wide")

# Hide Streamlit's default chrome ("Made with Streamlit" footer + hamburger menu).
st.markdown(
    "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)


def run_cli(args):
    """Run one of the toolkit scripts and capture its text report."""
    res = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=90)
    return (res.stdout or "") + (res.stderr or "")


st.title("Equity Valuation & Portfolio Toolkit")
st.caption("DCF valuation, portfolio risk, and macro — on live market data.")

tab_dcf, tab_port, tab_macro = st.tabs(["Stock Valuation (DCF)", "Portfolio Analysis", "Macro"])

# ---------------- DCF ----------------
with tab_dcf:
    st.subheader("Discounted-cash-flow intrinsic value")
    c1, c2, c3, c4 = st.columns(4)
    ticker = c1.text_input("Ticker", "AAPL").upper().strip()
    growth = c2.slider("FCF growth (%/yr)", 0.0, 20.0, 8.0, 0.5, format="%.1f%%")
    wacc = c3.slider("WACC / discount rate (%)", 5.0, 15.0, 9.0, 0.5, format="%.1f%%")
    terminal = c4.slider("Terminal growth (%)", 0.0, 4.0, 2.5, 0.5, format="%.1f%%")
    if st.button("Value it", type="primary"):
        with st.spinner(f"Fetching {ticker} and running the DCF…"):
            out = run_cli(["dcf.py", ticker, "--growth", str(growth / 100),
                           "--wacc", str(wacc / 100), "--terminal", str(terminal / 100)])
        st.code(out or "No output.", language="text")
        st.info("The sensitivity grid matters more than the point estimate — the fair value swings a lot with WACC and terminal growth.")

# ---------------- Portfolio ----------------
with tab_port:
    st.subheader("Portfolio risk & return")
    st.caption("One holding per line, as TICKER:SHARES")
    holdings_text = st.text_area("Holdings", "AAPL:50\nMSFT:20\nNVDA:10\nKO:100", height=140)
    cash = st.number_input("Cash ($)", min_value=0.0, value=5000.0, step=500.0)
    if st.button("Analyze portfolio", type="primary"):
        holdings = {}
        for line in holdings_text.splitlines():
            line = line.strip()
            if ":" in line:
                t, s = line.split(":", 1)
                try:
                    holdings[t.upper().strip()] = float(s)
                except ValueError:
                    pass
        if not holdings:
            st.error("Enter at least one holding as TICKER:SHARES.")
        else:
            with st.spinner("Pulling prices and computing metrics…"):
                from portfolio import analyze
                try:
                    res = analyze(holdings, cash=cash)
                except SystemExit as e:
                    st.error(str(e))
                    res = None
            if res:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total value", f"${res['total']:,.0f}")
                m2.metric("Annualized return", f"{res['ann_return']:+.1%}")
                m3.metric("Volatility", f"{res['ann_vol']:.1%}")
                m4.metric("Sharpe", f"{res['sharpe']:.2f}")
                m1b, m2b = st.columns(2)
                m1b.metric("Max drawdown", f"{res['max_drawdown']:.1%}")
                m2b.metric("Positions", f"{len(res['values'])}")

                weights = (pd.Series(res["weights"]).sort_values(ascending=False)
                           .rename("weight").to_frame())
                weights["value $"] = pd.Series(res["values"])
                st.markdown("**Positions**")
                st.dataframe(weights.style.format({"weight": "{:.1%}", "value $": "${:,.0f}"}),
                             use_container_width=True)

                st.markdown("**Correlation matrix** (lower = more diversified)")
                st.dataframe(res["corr"].style.format("{:.2f}").background_gradient(cmap="RdYlGn_r",
                             vmin=-1, vmax=1), use_container_width=True)

# ---------------- Macro ----------------
with tab_macro:
    st.subheader("Live macro backdrop")
    if st.button("Refresh macro", type="primary"):
        with st.spinner("Pulling yields, curve, VIX, commodities…"):
            out = run_cli(["macro.py"])
        st.code(out or "No output.", language="text")
    else:
        st.caption("Click to pull the current rates / curve / VIX / commodities snapshot.")
