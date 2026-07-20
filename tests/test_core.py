"""
Network-free unit tests for the valuation math. These run in CI without any
market-data connection, using fully manual DCF inputs.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcf import DCFInputs, intrinsic_value, wacc_from_capm  # noqa: E402


def test_wacc_from_capm():
    # rf 4.3% + beta 1.0 * erp 5% = 9.3%
    assert wacc_from_capm(1.0, rf=0.043, erp=0.05) == pytest.approx(0.093)
    # higher beta -> higher discount rate
    assert wacc_from_capm(1.5) > wacc_from_capm(0.5)


def test_intrinsic_value_positive_and_sane():
    inp = DCFInputs(fcf=100e9, shares=10e9, wacc=0.09, growth=0.08, terminal=0.025, years=10)
    out = intrinsic_value(inp)
    assert out["fair_value_per_share"] > 0
    assert out["enterprise_value"] > 0
    assert 0.0 < out["terminal_pct_of_value"] < 1.0


def test_net_cash_raises_value():
    base = DCFInputs(fcf=100e9, shares=10e9, wacc=0.09, growth=0.08, terminal=0.025)
    with_cash = DCFInputs(fcf=100e9, shares=10e9, wacc=0.09, growth=0.08,
                          terminal=0.025, net_debt=-50e9)  # net cash
    assert (intrinsic_value(with_cash)["fair_value_per_share"]
            > intrinsic_value(base)["fair_value_per_share"])


def test_higher_wacc_lowers_value():
    low = DCFInputs(fcf=100e9, shares=10e9, wacc=0.08, growth=0.08, terminal=0.025)
    high = DCFInputs(fcf=100e9, shares=10e9, wacc=0.12, growth=0.08, terminal=0.025)
    assert (intrinsic_value(high)["fair_value_per_share"]
            < intrinsic_value(low)["fair_value_per_share"])


def test_terminal_ge_wacc_rejected():
    bad = DCFInputs(fcf=100e9, shares=10e9, wacc=0.05, growth=0.08, terminal=0.06)
    with pytest.raises(ValueError):
        intrinsic_value(bad)
