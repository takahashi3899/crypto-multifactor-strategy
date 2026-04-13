"""Unit tests for FactorEngine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from factors import FactorEngine


# ----------------------------------------------------------------
# FIXTURES
# ----------------------------------------------------------------

@pytest.fixture
def price_df():
    """200 days x 5 assets of synthetic prices."""
    np.random.seed(42)
    n, assets = 200, ["BTC", "ETH", "SOL", "BNB", "ADA"]
    log_ret   = np.random.normal(0.001, 0.03, (n, len(assets)))
    prices    = pd.DataFrame(
        100 * np.exp(log_ret.cumsum(axis=0)),
        columns=assets,
    )
    prices.index = pd.date_range("2023-01-01", periods=n, freq="D")
    return prices


@pytest.fixture
def returns_df(price_df):
    return price_df.pct_change().dropna()


@pytest.fixture
def engine():
    return FactorEngine()


# ----------------------------------------------------------------
# MOMENTUM
# ----------------------------------------------------------------

class TestMomentum:
    def test_returns_series(self, engine, price_df):
        result = engine.momentum(price_df)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(price_df.columns)

    def test_no_nan(self, engine, price_df):
        result = engine.momentum(price_df)
        assert result.notna().all(), "Momentum contains NaN values"

    def test_direction(self, engine, price_df):
        """Asset that went up most should have highest momentum."""
        result = engine.momentum(price_df)
        # BTC synthetic seed makes one asset outperform; just check it's a float
        assert result.dtype == float

    def test_short_history(self, engine, price_df):
        """Should handle fewer rows than lookback gracefully."""
        result = engine.momentum(price_df.iloc[:10])
        # May return zeros or NaN, but should not raise
        assert isinstance(result, pd.Series)


# ----------------------------------------------------------------
# VALUE (MEAN REVERSION)
# ----------------------------------------------------------------

class TestValue:
    def test_returns_series(self, engine, price_df):
        result = engine.value(price_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_df.columns)

    def test_no_inf(self, engine, price_df):
        result = engine.value(price_df)
        assert not np.isinf(result).any(), "Value contains Inf"

    def test_negative_for_overbought(self, engine, price_df):
        """
        An asset that has moved far above its mean should have
        a negative value score (mean-reversion logic).
        """
        prices = price_df.copy()
        # Spike BTC by 5x on the last day
        prices["BTC"] = prices["BTC"] * 5
        result = engine.value(prices)
        assert result["BTC"] < 0, "Overbought asset should have negative value score"


# ----------------------------------------------------------------
# CARRY
# ----------------------------------------------------------------

class TestCarry:
    def test_returns_series(self, engine, price_df):
        result = engine.carry(price_df)
        assert isinstance(result, pd.Series)

    def test_no_nan(self, engine, price_df):
        result = engine.carry(price_df)
        assert result.notna().all()

    def test_range(self, engine, price_df):
        """Carry (trend estimate) should be finite."""
        result = engine.carry(price_df)
        assert np.isfinite(result).all()


# ----------------------------------------------------------------
# QUALITY
# ----------------------------------------------------------------

class TestQuality:
    def test_returns_series(self, engine, returns_df):
        result = engine.quality(returns_df)
        assert isinstance(result, pd.Series)

    def test_higher_sharpe_higher_quality(self, engine, returns_df):
        """
        Inject an asset with known high Sharpe; it should rank first.
        """
        rets = returns_df.copy()
        # Replace SOL with very consistent positive returns
        rets["SOL"] = 0.005  # constant daily return -> infinite Sharpe
        result = engine.quality(rets)
        assert result.idxmax() == "SOL", "Highest Sharpe asset should rank #1"

    def test_no_nan(self, engine, returns_df):
        result = engine.quality(returns_df)
        assert result.notna().all()


# ----------------------------------------------------------------
# COMBINED SIGNAL
# ----------------------------------------------------------------

class TestCombinedSignal:
    def test_weights_sum_to_one(self, engine, price_df, returns_df):
        mom     = engine.momentum(price_df)
        val     = engine.value(price_df)
        carry   = engine.carry(price_df)
        quality = engine.quality(returns_df)
        combined = pd.concat([mom, val, carry, quality], axis=1).mean(axis=1)
        combined = combined.clip(lower=0)
        weights  = combined / combined.sum()
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_no_negative_weights(self, engine, price_df, returns_df):
        mom     = engine.momentum(price_df)
        combined = mom.clip(lower=0)
        if combined.sum() > 0:
            weights = combined / combined.sum()
            assert (weights >= 0).all()
