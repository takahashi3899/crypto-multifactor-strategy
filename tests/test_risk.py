"""Unit tests for RiskManager."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from risk import RiskManager


# ----------------------------------------------------------------
# FIXTURES
# ----------------------------------------------------------------

@pytest.fixture
def rm():
    return RiskManager()


@pytest.fixture
def returns_df():
    np.random.seed(0)
    n, assets = 100, ["BTC", "ETH", "SOL", "BNB"]
    data = np.random.normal(0.001, 0.025, (n, len(assets)))
    df   = pd.DataFrame(data, columns=assets)
    df.index = pd.date_range("2023-01-01", periods=n, freq="D")
    return df


@pytest.fixture
def equity_curve():
    """Simple upward-trending equity curve."""
    idx = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.Series(1_000_000 * (1 + 0.001) ** np.arange(100), index=idx)


@pytest.fixture
def drawdown_curve():
    """Equity curve that has a severe -25% drawdown."""
    idx    = pd.date_range("2023-01-01", periods=100, freq="D")
    values = np.ones(100) * 1_000_000
    values[50:] *= 0.75   # 25% drop from day 50
    return pd.Series(values, index=idx)


# ----------------------------------------------------------------
# VOLATILITY TARGETING
# ----------------------------------------------------------------

class TestVolTargeting:
    def test_returns_series(self, rm, returns_df):
        weights = pd.Series(0.25, index=returns_df.columns)
        result  = rm.scale_to_vol_target(weights, returns_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(weights)

    def test_weights_capped(self, rm, returns_df):
        weights = pd.Series(0.5, index=returns_df.columns)  # above max
        result  = rm.scale_to_vol_target(weights, returns_df)
        assert (result <= rm.max_w + 1e-9).all(), "Weights should not exceed max_w"

    def test_short_history_passthrough(self, rm, returns_df):
        weights = pd.Series(0.25, index=returns_df.columns)
        result  = rm.scale_to_vol_target(weights, returns_df.iloc[:1])
        # Should return original weights unchanged
        pd.testing.assert_series_equal(result, weights)


# ----------------------------------------------------------------
# CONSTRAINT APPLICATION
# ----------------------------------------------------------------

class TestConstraints:
    def test_clips_above_max(self, rm):
        weights = pd.Series({"BTC": 0.5, "ETH": 0.1, "SOL": 0.05})
        result  = rm.apply_constraints(weights)
        assert (result.abs() <= rm.max_w + 1e-9).all()

    def test_zeros_below_min(self, rm):
        weights = pd.Series({"BTC": 0.001, "ETH": 0.2, "SOL": 0.002})
        result  = rm.apply_constraints(weights)
        assert result["BTC"] == 0.0
        assert result["SOL"] == 0.0

    def test_valid_weights_unchanged(self, rm):
        weights = pd.Series({"BTC": 0.15, "ETH": 0.15})
        result  = rm.apply_constraints(weights)
        pd.testing.assert_series_equal(result, weights)


# ----------------------------------------------------------------
# HARD STOP / DRAWDOWN
# ----------------------------------------------------------------

class TestHardStop:
    def test_no_stop_for_uptrend(self, rm, equity_curve):
        result = rm.check_hard_stop(equity_curve, threshold=-0.20)
        assert result is False

    def test_stop_triggered_for_large_dd(self, rm, drawdown_curve):
        result = rm.check_hard_stop(drawdown_curve, threshold=-0.20)
        assert result is True

    def test_too_short_curve(self, rm):
        curve  = pd.Series([1_000_000])
        result = rm.check_hard_stop(curve, threshold=-0.20)
        assert result is False

    def test_compute_drawdown_shape(self, rm, equity_curve):
        dd = rm.compute_drawdown(equity_curve)
        assert isinstance(dd, pd.Series)
        assert len(dd) == len(equity_curve)
        assert (dd <= 0).all(), "Drawdown values should be <= 0"

    def test_compute_drawdown_min(self, rm, drawdown_curve):
        dd  = rm.compute_drawdown(drawdown_curve)
        assert dd.min() <= -0.25 + 1e-3


# ----------------------------------------------------------------
# VALUE AT RISK
# ----------------------------------------------------------------

class TestVaR:
    def test_returns_float(self, rm, returns_df):
        weights = pd.Series(0.25, index=returns_df.columns)
        var     = rm.value_at_risk(weights, returns_df)
        assert isinstance(var, float)
        assert var >= 0

    def test_higher_vol_higher_var(self, rm, returns_df):
        w      = pd.Series(0.25, index=returns_df.columns)
        low_v  = returns_df * 0.1
        high_v = returns_df * 3.0
        assert rm.value_at_risk(w, low_v) < rm.value_at_risk(w, high_v)


# ----------------------------------------------------------------
# CONCENTRATION PENALTY
# ----------------------------------------------------------------

class TestConcentrationPenalty:
    def test_reduces_correlated_weights(self, rm, returns_df):
        weights = pd.Series(0.25, index=returns_df.columns)
        corr    = returns_df.corr()
        # Artificially set all correlations high
        high_corr = pd.DataFrame(
            np.full((4, 4), 0.9), index=corr.index, columns=corr.columns
        )
        np.fill_diagonal(high_corr.values, 1.0)
        result = rm.concentration_penalty(weights, high_corr)
        # Weights should be reduced due to high correlation
        assert (result <= weights + 1e-9).all()

    def test_low_corr_preserves_weights(self, rm, returns_df):
        weights  = pd.Series(0.20, index=returns_df.columns)
        low_corr = pd.DataFrame(
            np.eye(4), index=returns_df.columns, columns=returns_df.columns
        )
        result = rm.concentration_penalty(weights, low_corr)
        # Near-zero correlation -> minimal penalty
        pd.testing.assert_series_equal(
            result.round(4), weights.clip(upper=rm.max_w).round(4)
        )
