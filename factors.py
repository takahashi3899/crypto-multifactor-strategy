# =============================================================
# factors.py — 4-Factor Alpha Engine
# Factors: Momentum | Value | Carry | Quality
# =============================================================

import pandas as pd
import numpy as np


class FactorEngine:
    """
    Computes all 4 alpha factors cross-sectionally.
    All outputs are cross-sectionally ranked 0→1 per date row.
    Higher score = more attractive (long candidate).
    """

    def __init__(self, prices: pd.DataFrame, volumes: pd.DataFrame, cfg: dict):
        self.prices  = prices.copy()
        self.volumes = volumes.copy()
        self.cfg     = cfg
        self.returns = prices.pct_change()

    # ================================================================
    # FACTOR 1: MOMENTUM
    # Multi-horizon momentum with short-term reversal skip
    # ================================================================
    def compute_momentum(self) -> pd.DataFrame:
        """
        score = weighted avg of 20d, 60d, 120d price return
        Skips most recent 'mom_skip' days to avoid reversal.
        """
        skip    = self.cfg["mom_skip"]
        windows = self.cfg["mom_windows"]
        wts     = self.cfg["mom_weights"]
        composite = None

        for window, wt in zip(windows, wts):
            past_price   = self.prices.shift(window)
            recent_price = self.prices.shift(skip)
            raw_ret      = (recent_price - past_price) / (past_price + 1e-9)
            if composite is None:
                composite = raw_ret * wt
            else:
                composite = composite + raw_ret * wt

        return self._cross_rank(composite)

    # ================================================================
    # FACTOR 2: VALUE
    # Crypto-native value via NVT proxy (lower NVT = cheaper)
    # In production: replace with Token Terminal P/Fees ratio
    # ================================================================
    def compute_value(self) -> pd.DataFrame:
        """
        NVT proxy = price / rolling 30d avg volume.
        Lower NVT = more fundamentally cheap = higher value score.
        """
        vol_30d   = self.volumes.rolling(30, min_periods=10).mean()
        ntv_proxy = self.prices / (vol_30d + 1e-9)
        # Invert: high NVT = overvalued = low score
        return self._cross_rank(-ntv_proxy)

    # ================================================================
    # FACTOR 3: CARRY
    # Positive carry = upward momentum acceleration + low vol drag
    # In production: replace proxy with perpetual funding rate + staking yield
    # ================================================================
    def compute_carry(self) -> pd.DataFrame:
        """
        Carry proxy = short-term slope - medium-term slope,
        penalised by realised vol (high vol = negative carry).
        """
        short_slope  = self.prices.pct_change(10)
        medium_slope = self.prices.pct_change(30)
        acceleration = short_slope - medium_slope * 0.5

        # Vol drag penalty (higher vol = worse carry)
        vol_drag = self.returns.rolling(20).std() * np.sqrt(252)
        carry    = acceleration - vol_drag * 0.3

        return self._cross_rank(carry)

    # ================================================================
    # FACTOR 4: QUALITY (On-Chain Activity Proxy)
    # Growing network activity predicts outperformance
    # In production: replace proxy with Glassnode active addresses
    # ================================================================
    def compute_quality(self) -> pd.DataFrame:
        """
        Quality = volume growth rate (30d vs 90d) * volume consistency.
        Consistent, accelerating volume = healthy network.
        """
        vol_30d   = self.volumes.rolling(30, min_periods=10).mean()
        vol_90d   = self.volumes.rolling(90, min_periods=30).mean()
        vol_growth = (vol_30d - vol_90d) / (vol_90d + 1e-9)

        # Consistency: low CoV = stable volume = higher quality
        vol_std   = self.volumes.rolling(30).std()
        vol_cov   = vol_std / (vol_30d + 1e-9)
        consistency = 1 / (vol_cov + 1)

        quality = vol_growth * 0.7 + consistency * 0.3
        return self._cross_rank(quality)

    # ================================================================
    # COMPOSITE ALPHA SCORE
    # ================================================================
    def compute_composite(self, weights: dict = None) -> pd.DataFrame:
        """
        Weighted combination of all 4 factors.
        weights: optional override dict (keys: momentum/value/carry/quality)
        """
        w = weights or self.cfg["factor_weights"]
        mom  = self.compute_momentum() * w["momentum"]
        val  = self.compute_value()    * w["value"]
        car  = self.compute_carry()    * w["carry"]
        qua  = self.compute_quality()  * w["quality"]
        return mom + val + car + qua

    def compute_all_factors(self) -> dict:
        """Returns dict of all individual factor DataFrames."""
        return {
            "momentum": self.compute_momentum(),
            "value":    self.compute_value(),
            "carry":    self.compute_carry(),
            "quality":  self.compute_quality(),
        }

    # ================================================================
    # UTILITIES
    # ================================================================
    @staticmethod
    def _cross_rank(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional percentile rank each row (0=worst, 1=best). NaN-safe."""
        return df.rank(axis=1, pct=True, na_option='keep')
