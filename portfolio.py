import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


class PortfolioOptimizer:
    """
    Institutional-grade portfolio construction.

    Supports
    --------
    1. Mean-Variance Optimization (Markowitz)
    2. Maximum Sharpe Ratio (tangency portfolio)
    3. Minimum Variance
    4. Risk Parity (equal risk contribution)
    5. Black-Litterman with factor views
    """

    def __init__(
        self,
        risk_aversion:  float = 3.0,
        max_weight:     float = 0.20,
        min_weight:     float = 0.0,
        rf_rate:        float = 0.05,   # annual risk-free rate
        lookback:       int   = 120,
    ):
        self.risk_aversion = risk_aversion
        self.max_weight    = max_weight
        self.min_weight    = min_weight
        self.rf_rate       = rf_rate / 365  # daily
        self.lookback      = lookback

    # ================================================================
    # COVARIANCE ESTIMATION
    # ================================================================

    def ledoit_wolf_shrinkage(
        self, returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Ledoit-Wolf analytical shrinkage estimator.
        Shrinks sample covariance toward scaled identity.
        Reduces estimation error in high-dimension settings.
        """
        T, N   = returns.shape
        S      = returns.cov().values
        mu_hat = np.trace(S) / N
        # Oracle shrinkage coefficient (analytical formula)
        delta2  = np.linalg.norm(S - mu_hat * np.eye(N), 'fro') ** 2
        beta2   = 0.0
        for t in range(T):
            x = returns.iloc[t].values
            beta2 += np.linalg.norm(np.outer(x, x) - S, 'fro') ** 2
        beta2 /= T ** 2
        beta_hat = min(beta2 / delta2, 1.0)
        return (1 - beta_hat) * S + beta_hat * mu_hat * np.eye(N)

    def _get_cov(self, returns: pd.DataFrame) -> np.ndarray:
        recent = returns.iloc[-self.lookback:]
        return self.ledoit_wolf_shrinkage(recent) * 365  # annualise

    # ================================================================
    # MEAN-VARIANCE OPTIMIZATION
    # ================================================================

    def max_sharpe(
        self,
        returns:  pd.DataFrame,
        mu_views: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Maximum Sharpe Ratio portfolio.
        Uses Black-Litterman expected returns if views provided.
        """
        assets   = returns.columns.tolist()
        N        = len(assets)
        cov      = self._get_cov(returns)
        mu       = self._expected_returns(returns, mu_views)
        mu_excess = mu - self.rf_rate * 365

        # Objective: minimise negative Sharpe
        def neg_sharpe(w):
            port_ret = w @ mu_excess
            port_vol = np.sqrt(w @ cov @ w)
            return -port_ret / (port_vol + 1e-9)

        w0      = np.ones(N) / N
        bounds  = [(self.min_weight, self.max_weight)] * N
        constr  = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result  = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constr,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        weights = result.x if result.success else w0
        return pd.Series(weights, index=assets)

    def min_variance(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """Global minimum variance portfolio."""
        assets  = returns.columns.tolist()
        N       = len(assets)
        cov     = self._get_cov(returns)

        def port_var(w):
            return w @ cov @ w

        w0     = np.ones(N) / N
        bounds = [(self.min_weight, self.max_weight)] * N
        constr = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result = minimize(
            port_var, w0, method="SLSQP",
            bounds=bounds, constraints=constr,
        )
        weights = result.x if result.success else w0
        return pd.Series(weights, index=assets)

    # ================================================================
    # RISK PARITY
    # ================================================================

    def risk_parity(
        self,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Equal Risk Contribution (ERC) portfolio.
        Each asset contributes equally to total portfolio variance.
        """
        assets  = returns.columns.tolist()
        N       = len(assets)
        cov     = self._get_cov(returns)
        target  = np.ones(N) / N   # equal contribution

        def risk_budget_obj(w):
            w      = np.array(w)
            port_var = w @ cov @ w
            mrc    = cov @ w       # marginal risk contribution
            rc     = w * mrc / port_var  # risk contribution
            return np.sum((rc - target) ** 2)

        w0     = np.ones(N) / N
        bounds = [(1e-4, self.max_weight)] * N
        constr = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        result = minimize(
            risk_budget_obj, w0, method="SLSQP",
            bounds=bounds, constraints=constr,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        weights = result.x if result.success else w0
        return pd.Series(weights / weights.sum(), index=assets)

    # ================================================================
    # BLACK-LITTERMAN
    # ================================================================

    def black_litterman(
        self,
        returns:     pd.DataFrame,
        views:       pd.Series,      # factor-derived alpha views {asset: expected_return}
        view_conf:   float = 0.5,    # confidence in views (0=prior only, 1=views only)
    ) -> pd.Series:
        """
        Black-Litterman model.

        Blends market equilibrium returns (implied from market caps /
        equal-weight prior) with factor-derived views.

        Parameters
        ----------
        returns   : historical returns DataFrame
        views     : Series of expected returns per asset (from factor model)
        view_conf : tau parameter controlling view confidence
        """
        assets = returns.columns.tolist()
        N      = len(assets)
        cov    = self._get_cov(returns)

        # Prior: reverse-optimised equilibrium returns
        w_eq   = np.ones(N) / N   # equal-weight market portfolio
        pi     = self.risk_aversion * cov @ w_eq   # equilibrium excess returns

        # View matrix (P) and view vector (Q)
        view_assets = [a for a in views.index if a in assets]
        if not view_assets:
            return self.max_sharpe(returns)

        k   = len(view_assets)
        P   = np.zeros((k, N))
        Q   = np.zeros(k)
        for i, asset in enumerate(view_assets):
            j       = assets.index(asset)
            P[i, j] = 1.0
            Q[i]    = views[asset]

        tau   = view_conf
        Omega = np.diag(np.diag(tau * P @ cov @ P.T))  # view uncertainty

        # BL posterior
        tau_cov     = tau * cov
        M_inv       = np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P
        mu_bl       = np.linalg.solve(
            M_inv,
            np.linalg.solve(tau_cov, pi) + P.T @ np.linalg.inv(Omega) @ Q
        )
        cov_bl      = cov + np.linalg.inv(M_inv)

        # Optimise with BL inputs
        mu_series   = pd.Series(mu_bl, index=assets)
        return self.max_sharpe(returns, mu_views=mu_series)

    # ================================================================
    # EXPECTED RETURNS
    # ================================================================

    def _expected_returns(
        self,
        returns:  pd.DataFrame,
        mu_views: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Use EWMA historical returns, blended with views if provided.
        """
        span    = self.lookback
        ewma_mu = returns.ewm(span=span).mean().iloc[-1].values * 365
        if mu_views is not None:
            v   = mu_views.reindex(returns.columns).fillna(0).values
            ewma_mu = 0.5 * ewma_mu + 0.5 * v
        return ewma_mu

    # ================================================================
    # TURNOVER-CONSTRAINED REBALANCE
    # ================================================================

    def rebalance_with_turnover_limit(
        self,
        target_weights:  pd.Series,
        current_weights: pd.Series,
        max_turnover:    float = 0.10,
    ) -> pd.Series:
        """
        Clip weight changes to max_turnover per rebalance.
        Useful for reducing transaction costs.
        """
        delta    = target_weights - current_weights.reindex(target_weights.index).fillna(0)
        total_to = delta.abs().sum()
        if total_to > max_turnover:
            scale = max_turnover / total_to
            delta = delta * scale
        new_weights = (current_weights.reindex(target_weights.index).fillna(0) + delta)
        return new_weights.clip(lower=0).pipe(lambda w: w / w.sum() if w.sum() > 0 else w)

    # ================================================================
    # DIAGNOSTICS
    # ================================================================

    @staticmethod
    def portfolio_stats(
        weights: pd.Series,
        returns: pd.DataFrame,
        rf:      float = 0.05,
    ) -> dict:
        """Compute expected portfolio stats from weights."""
        w      = weights.reindex(returns.columns).fillna(0).values
        mu     = returns.mean().values * 365
        cov    = returns.cov().values * 365
        p_ret  = float(w @ mu)
        p_vol  = float(np.sqrt(w @ cov @ w))
        sharpe = (p_ret - rf) / (p_vol + 1e-9)
        return {
            "expected_return": round(p_ret, 4),
            "expected_vol":    round(p_vol, 4),
            "expected_sharpe": round(sharpe, 4),
        }
