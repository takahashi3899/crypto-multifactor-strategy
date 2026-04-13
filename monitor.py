import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ================================================================
# ALERT SYSTEM
# ================================================================

class Alert:
    """Immutable alert event."""
    __slots__ = ("level", "message", "ts", "data")

    def __init__(self, level: str, message: str, data: dict = None):
        self.level   = level   # INFO | WARNING | CRITICAL
        self.message = message
        self.ts      = datetime.now(timezone.utc)
        self.data    = data or {}

    def __repr__(self):
        return f"[{self.ts.strftime('%H:%M:%S')} {self.level}] {self.message}"


class AlertDispatcher:
    """
    Routes alerts to one or more sinks:
    - Console (always on)
    - Log file
    - Webhook / Slack (optional callback)
    """

    def __init__(self, webhook_fn: Callable[[Alert], None] = None):
        self._history: deque = deque(maxlen=500)
        self._webhook = webhook_fn

    def send(self, level: str, message: str, data: dict = None) -> Alert:
        alert = Alert(level, message, data)
        self._history.append(alert)
        # Console
        getattr(log, level.lower(), log.info)("ALERT: %s", message)
        # Webhook (e.g. Slack, Telegram)
        if self._webhook:
            try:
                self._webhook(alert)
            except Exception as e:
                log.error("Webhook failed: %s", e)
        return alert

    def info(self, msg: str, **kw)     -> Alert: return self.send("INFO",     msg, kw)
    def warning(self, msg: str, **kw)  -> Alert: return self.send("WARNING",  msg, kw)
    def critical(self, msg: str, **kw) -> Alert: return self.send("CRITICAL", msg, kw)

    def recent(self, n: int = 20) -> List[Alert]:
        return list(self._history)[-n:]


# ================================================================
# PERFORMANCE TRACKER
# ================================================================

class PerformanceTracker:
    """
    Real-time rolling performance metrics.
    Maintains a time series of NAV and computes live stats.
    """

    def __init__(self, initial_nav: float = 1_000_000.0):
        self._nav_history: List[tuple] = []   # (datetime, nav)
        self.initial_nav  = initial_nav
        self.peak_nav     = initial_nav

    def update(self, nav: float, ts: datetime = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        self._nav_history.append((ts, nav))
        if nav > self.peak_nav:
            self.peak_nav = nav

    @property
    def nav_series(self) -> pd.Series:
        if not self._nav_history:
            return pd.Series(dtype=float)
        idx, vals = zip(*self._nav_history)
        return pd.Series(vals, index=idx, name="NAV")

    @property
    def current_nav(self) -> float:
        return self._nav_history[-1][1] if self._nav_history else self.initial_nav

    @property
    def total_return(self) -> float:
        return (self.current_nav / self.initial_nav) - 1

    @property
    def current_drawdown(self) -> float:
        return (self.current_nav - self.peak_nav) / self.peak_nav

    def rolling_sharpe(self, window: int = 30) -> float:
        nav = self.nav_series
        if len(nav) < window + 1:
            return 0.0
        rets = nav.pct_change().dropna().iloc[-window:]
        mu   = rets.mean() * 365
        vol  = rets.std()  * np.sqrt(365)
        return float(mu / vol) if vol > 0 else 0.0

    def summary(self) -> dict:
        nav = self.nav_series
        if nav.empty:
            return {}
        rets = nav.pct_change().dropna()
        return {
            "current_nav":      round(self.current_nav, 2),
            "total_return_pct": round(self.total_return * 100, 2),
            "current_dd_pct":   round(self.current_drawdown * 100, 2),
            "rolling_sharpe_30d": round(self.rolling_sharpe(30), 3),
            "days_tracked":     len(nav),
        }


# ================================================================
# POSITION MONITOR
# ================================================================

class PositionMonitor:
    """
    Tracks live positions and raises alerts on:
    - Position concentration breach
    - Large single-asset P&L moves
    - Stale price data
    """

    def __init__(
        self,
        max_concentration: float = 0.25,   # max 25% single asset
        pnl_alert_pct:     float = 0.05,   # alert if >5% daily move
        stale_seconds:     int   = 300,    # 5 min stale threshold
    ):
        self.max_concentration = max_concentration
        self.pnl_alert_pct     = pnl_alert_pct
        self.stale_seconds     = stale_seconds
        self._last_prices:  Dict[str, float]    = {}
        self._price_times:  Dict[str, datetime] = {}
        self._prev_values:  Dict[str, float]    = {}

    def update_prices(
        self,
        prices: pd.Series,
        ts:     datetime = None,
    ) -> None:
        ts = ts or datetime.now(timezone.utc)
        for sym, price in prices.items():
            self._last_prices[sym] = float(price)
            self._price_times[sym] = ts

    def check_concentration(
        self,
        weights:    pd.Series,
        dispatcher: AlertDispatcher,
    ) -> None:
        over = weights[weights > self.max_concentration]
        for sym, w in over.items():
            dispatcher.warning(
                f"{sym} concentration {w:.1%} exceeds limit {self.max_concentration:.1%}",
                symbol=sym, weight=w,
            )

    def check_pnl_moves(
        self,
        positions:  dict,      # {sym: {qty, avg_cost, last_price, pnl_pct}}
        dispatcher: AlertDispatcher,
    ) -> None:
        for sym, pos in positions.items():
            prev = self._prev_values.get(sym)
            curr = pos["last_price"] * pos["qty"]
            if prev is not None:
                chg = abs(curr - prev) / (abs(prev) + 1e-9)
                if chg > self.pnl_alert_pct:
                    dispatcher.warning(
                        f"{sym} position value moved {chg:.1%} since last check",
                        symbol=sym, change_pct=round(chg * 100, 2),
                    )
            self._prev_values[sym] = curr

    def check_stale_prices(
        self,
        dispatcher: AlertDispatcher,
        now:        datetime = None,
    ) -> None:
        now = now or datetime.now(timezone.utc)
        for sym, ts in self._price_times.items():
            age = (now - ts).total_seconds()
            if age > self.stale_seconds:
                dispatcher.warning(
                    f"{sym} price data stale ({age:.0f}s old)",
                    symbol=sym, age_seconds=age,
                )


# ================================================================
# RISK MONITOR
# ================================================================

class RiskMonitor:
    """
    Continuously checks risk limits and fires CRITICAL alerts
    if any are breached.
    """

    def __init__(
        self,
        hard_stop_dd:   float = -0.20,   # -20% drawdown
        daily_loss_pct: float = -0.05,   # -5% intraday
        gross_exp_max:  float = 1.50,    # 150% gross
    ):
        self.hard_stop_dd   = hard_stop_dd
        self.daily_loss_pct = daily_loss_pct
        self.gross_exp_max  = gross_exp_max
        self._day_start_nav: Optional[float] = None
        self._trading_halted: bool = False

    def check_all(
        self,
        nav:        float,
        peak_nav:   float,
        weights:    pd.Series,
        dispatcher: AlertDispatcher,
    ) -> bool:
        """
        Run all checks. Returns True if trading should halt.
        """
        if self._trading_halted:
            dispatcher.critical("Trading is HALTED — manual review required.")
            return True

        # --- Hard stop ---
        dd = (nav - peak_nav) / peak_nav
        if dd <= self.hard_stop_dd:
            self._trading_halted = True
            dispatcher.critical(
                f"HARD STOP triggered! Drawdown {dd:.1%} breached {self.hard_stop_dd:.1%}",
                drawdown=dd,
            )
            return True

        # --- Daily loss ---
        if self._day_start_nav is None:
            self._day_start_nav = nav
        daily_ret = (nav - self._day_start_nav) / self._day_start_nav
        if daily_ret <= self.daily_loss_pct:
            dispatcher.critical(
                f"Daily loss {daily_ret:.1%} breached limit {self.daily_loss_pct:.1%}",
                daily_ret=daily_ret,
            )
            return True

        # --- Gross exposure ---
        gross = weights.abs().sum()
        if gross > self.gross_exp_max:
            dispatcher.warning(
                f"Gross exposure {gross:.2f}x exceeds max {self.gross_exp_max:.2f}x",
                gross_exposure=gross,
            )

        return False

    def reset_daily(self, nav: float) -> None:
        """Call at start of each trading day."""
        self._day_start_nav = nav


# ================================================================
# MASTER MONITOR (orchestrates all monitors)
# ================================================================

class SystemMonitor:
    """
    Top-level monitor. Wires together:
    - PerformanceTracker
    - PositionMonitor
    - RiskMonitor
    - AlertDispatcher
    """

    def __init__(
        self,
        initial_nav:   float = 1_000_000.0,
        webhook_fn:    Callable = None,
    ):
        self.perf      = PerformanceTracker(initial_nav)
        self.positions = PositionMonitor()
        self.risk      = RiskMonitor()
        self.alerts    = AlertDispatcher(webhook_fn)

    def tick(
        self,
        nav:      float,
        weights:  pd.Series,
        prices:   pd.Series,
        snapshot: dict,
    ) -> bool:
        """
        Called every update cycle (e.g. each minute or each bar).
        Returns True if trading should halt.
        """
        self.perf.update(nav)
        self.positions.update_prices(prices)
        self.positions.check_concentration(weights, self.alerts)
        self.positions.check_pnl_moves(snapshot.get("positions", {}), self.alerts)
        self.positions.check_stale_prices(self.alerts)
        halt = self.risk.check_all(
            nav, self.perf.peak_nav, weights, self.alerts
        )
        return halt

    def print_dashboard(self) -> None:
        summary = self.perf.summary()
        print("\n" + "=" * 55)
        print("  LIVE SYSTEM MONITOR")
        print("=" * 55)
        for k, v in summary.items():
            print(f"  {k:<28}: {v}")
        recent = self.alerts.recent(5)
        if recent:
            print("\n  Recent Alerts:")
            for a in recent:
                print(f"    {a}")
        print("=" * 55 + "\n")
