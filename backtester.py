import pandas as pd
import numpy as np
from typing import Optional
from config import TRADING_CONFIG


class Backtester:
    """
    Event-driven backtester with realistic transaction cost modeling.
    Supports daily rebalancing with configurable cost structure.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or TRADING_CONFIG
        self.commission = self.cfg.get("commission_bps", 10) / 10000
        self.slippage  = self.cfg.get("slippage_bps",  5)  / 10000

    # ================================================================
    # CORE BACKTEST ENGINE
    # ================================================================

    def run(
        self,
        prices:  pd.DataFrame,
        weights: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
    ) -> dict:
        """
        Run vectorised backtest.

        Parameters
        ----------
        prices  : DataFrame [dates x assets] of close prices
        weights : DataFrame [dates x assets] of target weights (sum <= 1)
        initial_capital : starting NAV in USD

        Returns
        -------
        dict with keys: equity_curve, returns, turnover, positions, stats
        """
        # Align
        prices  = prices.sort_index()
        weights = weights.sort_index().reindex(prices.index, method="ffill").fillna(0)
        weights = weights.reindex(columns=prices.columns, fill_value=0)

        daily_returns = prices.pct_change().fillna(0)

        portfolio_returns = []
        prev_weights      = pd.Series(0.0, index=prices.columns)
        nav               = initial_capital
        nav_series        = []
        turnover_series   = []

        for date in prices.index:
            w_target = weights.loc[date]
            rets     = daily_returns.loc[date]

            # Transaction costs on weight changes
            delta          = (w_target - prev_weights).abs().sum()
            cost           = delta * (self.commission + self.slippage)
            gross_return   = (w_target * rets).sum()
            net_return     = gross_return - cost

            nav           *= (1 + net_return)
            portfolio_returns.append(net_return)
            nav_series.append(nav)
            turnover_series.append(delta)

            prev_weights = w_target

        equity_curve   = pd.Series(nav_series,      index=prices.index, name="NAV")
        returns_series = pd.Series(portfolio_returns, index=prices.index, name="Returns")
        turnover_series = pd.Series(turnover_series,  index=prices.index, name="Turnover")

        stats = self._compute_stats(returns_series, equity_curve, turnover_series)

        return {
            "equity_curve": equity_curve,
            "returns":      returns_series,
            "turnover":     turnover_series,
            "positions":    weights,
            "stats":        stats,
        }

    # ================================================================
    # PERFORMANCE METRICS
    # ================================================================

    def _compute_stats(
        self,
        returns:      pd.Series,
        equity_curve: pd.Series,
        turnover:     pd.Series,
    ) -> dict:
        ann = 365  # crypto trades 24/7
        mu  = returns.mean() * ann
        vol = returns.std()  * np.sqrt(ann)

        sharpe = mu / vol if vol > 0 else 0.0
        sortino = self._sortino(returns, ann)
        calmar  = self._calmar(returns, equity_curve, ann)
        max_dd  = self._max_drawdown(equity_curve)

        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        avg_turnover = turnover.mean()

        win_rate = (returns > 0).mean()
        profit_factor = (
            returns[returns > 0].sum() / abs(returns[returns < 0].sum())
            if returns[returns < 0].sum() != 0 else np.inf
        )

        return {
            "total_return":      round(total_return,  4),
            "annualised_return": round(mu,            4),
            "annualised_vol":    round(vol,           4),
            "sharpe_ratio":      round(sharpe,        4),
            "sortino_ratio":     round(sortino,       4),
            "calmar_ratio":      round(calmar,        4),
            "max_drawdown":      round(max_dd,        4),
            "win_rate":          round(win_rate,      4),
            "profit_factor":     round(profit_factor, 4),
            "avg_daily_turnover":round(avg_turnover,  4),
        }

    @staticmethod
    def _sortino(returns: pd.Series, ann: int) -> float:
        downside = returns[returns < 0].std() * np.sqrt(ann)
        mu       = returns.mean() * ann
        return mu / downside if downside > 0 else 0.0

    @staticmethod
    def _calmar(returns: pd.Series, equity_curve: pd.Series, ann: int) -> float:
        mu     = returns.mean() * ann
        max_dd = Backtester._max_drawdown(equity_curve)
        return mu / abs(max_dd) if max_dd != 0 else 0.0

    @staticmethod
    def _max_drawdown(equity_curve: pd.Series) -> float:
        peak = equity_curve.cummax()
        dd   = (equity_curve - peak) / peak
        return float(dd.min())

    # ================================================================
    # WALK-FORWARD VALIDATION
    # ================================================================

    def walk_forward(
        self,
        prices:      pd.DataFrame,
        weights_fn,
        train_days:  int = 365,
        test_days:   int = 90,
        initial_capital: float = 1_000_000.0,
    ) -> pd.DataFrame:
        """
        Expanding-window walk-forward analysis.
        weights_fn(train_prices) -> target_weights_for_test_period
        Returns concatenated OOS equity curve.
        """
        results = []
        dates   = prices.index
        start   = train_days

        while start + test_days <= len(dates):
            train = prices.iloc[:start]
            test  = prices.iloc[start: start + test_days]

            w = weights_fn(train)
            w = w.reindex(test.index, method="ffill").fillna(0)

            res = self.run(test, w, initial_capital)
            results.append(res["equity_curve"])

            start += test_days

        return pd.concat(results) if results else pd.Series(dtype=float)

    # ================================================================
    # REPORTING
    # ================================================================

    @staticmethod
    def print_stats(stats: dict) -> None:
        print("\n" + "=" * 50)
        print("  BACKTEST PERFORMANCE SUMMARY")
        print("=" * 50)
        labels = {
            "total_return":       "Total Return",
            "annualised_return":  "Ann. Return",
            "annualised_vol":     "Ann. Volatility",
            "sharpe_ratio":       "Sharpe Ratio",
            "sortino_ratio":      "Sortino Ratio",
            "calmar_ratio":       "Calmar Ratio",
            "max_drawdown":       "Max Drawdown",
            "win_rate":           "Win Rate",
            "profit_factor":      "Profit Factor",
            "avg_daily_turnover": "Avg Daily Turnover",
        }
        for key, label in labels.items():
            val = stats.get(key, "N/A")
            if isinstance(val, float):
                if key in ("total_return", "annualised_return", "annualised_vol",
                           "max_drawdown", "win_rate", "avg_daily_turnover"):
                    print(f"  {label:<22}: {val * 100:>8.2f}%")
                else:
                    print(f"  {label:<22}: {val:>8.4f}")
        print("=" * 50 + "\n")
