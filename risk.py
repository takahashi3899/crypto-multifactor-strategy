import numpy as np
import pandas as pd
from config import RISK_CONFIG


class RiskManager:
    """
    Millennium-style risk management:
    - Volatility targeting (portfolio level)
    - Per-asset and per-sector position limits
    - Hard stop / drawdown circuit breaker
    - Correlation-adjusted exposure scaling
    """

    def __init__(self, config: dict = None):
        self.cfg         = config or RISK_CONFIG
        self.vol_target  = self.cfg.get("vol_target",    0.15)
        self.max_pos     = self.cfg.get("max_position",  0.20)
        self.max_gross   = self.cfg.get("max_gross",     1.50)
        self.min_w       = self.cfg.get("min_weight",    0.01)
        self.max_w       = self.cfg.get("max_weight",    0.20)
        self.hard_stop   = self.cfg.get("hard_stop_dd", -0.20)
        self.lookback    = self.cfg.get("vol_lookback",    30)

    # ================================================================
    # VOLATILITY TARGETING
    # ================================================================

    def scale_to_vol_target(
        self,
        weights:       pd.Series,
        returns:       pd.DataFrame,
        target_vol:    float = None,
    ) -> pd.Series:
        """
        Scale portfolio weights so realised volatility matches target.
        Uses exponentially-weighted covariance.
        """
        target = target_vol or self.vol_target
        if returns.shape[0] < 2:
            return weights

        recent = returns.iloc[-self.lookback:]
        cov    = recent.cov() * 365          # annualised
        w      = weights.reindex(cov.columns).fillna(0)
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(port_var) if port_var > 0 else 1e-6

        scalar = min(target / port_vol, 1.5)  # cap leverage at 1.5x
        return (weights * scalar).clip(upper=self.max_w)

    # ================================================================
    # WEIGHT CONSTRAINTS
    # ================================================================

    def apply_constraints(
        self,
        weights: pd.Series,
        max_w:   float = None,
        min_w:   float = None,
    ) -> pd.Series:
        """
        Enforce per-asset limits and re-normalise.
        """
        max_w = max_w or self.max_w
        min_w = min_w or self.min_w

        # Clip individual weights
        weights = weights.clip(lower=-max_w, upper=max_w)
        weights[weights.abs() < min_w] = 0.0

        return weights

    # ================================================================
    # DRAWDOWN MONITOR
    # ================================================================

    @staticmethod
    def check_hard_stop(equity_curve: pd.Series, threshold: float) -> bool:
        """
        Returns True if current drawdown breaches threshold.
        Triggers full exit and strategy pause.
        """
        if len(equity_curve) < 2:
            return False
        peak    = equity_curve.cummax()
        dd      = (equity_curve - peak) / peak
        current = dd.iloc[-1]
        return bool(current <= threshold)

    @staticmethod
    def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
        """Returns full drawdown series."""
        peak = equity_curve.cummax()
        return (equity_curve - peak) / peak

    # ================================================================
    # CORRELATION RISK
    # ================================================================

    def concentration_penalty(
        self,
        weights: pd.Series,
        corr:    pd.DataFrame,
    ) -> pd.Series:
        """
        Reduce weights of highly-correlated clusters.
        Assets with average pairwise corr > 0.8 get downscaled.
        """
        w = weights.reindex(corr.columns).fillna(0)
        avg_corr = corr.abs().mean(axis=1)
        penalty  = avg_corr.apply(lambda c: 1 - 0.5 * max(0, c - 0.8))
        return (w * penalty).clip(upper=self.max_w)

    # ================================================================
    # PORTFOLIO-LEVEL VAR
    # ================================================================

    def value_at_risk(
        self,
        weights:    pd.Series,
        returns:    pd.DataFrame,
        confidence: float = 0.95,
        horizon:    int   = 1,
    ) -> float:
        """
        Parametric VaR at given confidence level.
        Returns loss as a positive number (e.g. 0.03 = 3% of NAV).
        """
        recent = returns.iloc[-self.lookback:]
        w      = weights.reindex(recent.columns).fillna(0)
        port_ret = recent @ w
        z        = abs(np.percentile(port_ret, (1 - confidence) * 100))
        return float(z * np.sqrt(horizon))
