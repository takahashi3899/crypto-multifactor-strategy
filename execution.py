import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class Order:
    symbol:    str
    side:      str          # "buy" | "sell"
    qty:       float        # base-asset quantity
    order_type:str = "market"
    price:     float = 0.0  # limit price (0 = market)
    status:    str  = "pending"   # pending | filled | cancelled
    fill_price:float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return (
            f"Order({self.side.upper()} {self.qty:.6f} {self.symbol} "
            f"@ {self.fill_price or self.price or 'MKT'} [{self.status}])"
        )


@dataclass
class Position:
    symbol:    str
    qty:       float = 0.0
    avg_cost:  float = 0.0
    last_price:float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.last_price

    @property
    def unrealised_pnl(self) -> float:
        return self.qty * (self.last_price - self.avg_cost)

    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.last_price - self.avg_cost) / self.avg_cost


# ================================================================
# SLIPPAGE MODELS
# ================================================================

class SlippageModel:
    """
    Realistic fill-price estimation.

    Models
    ------
    fixed_bps   : constant slippage in basis points
    sqrt_impact : square-root market impact (Kyle / Almgren-Chriss)
    """

    @staticmethod
    def fixed_bps(price: float, bps: float = 5.0) -> float:
        """Return fill price after fixed BPS slippage."""
        return price * (1 + bps / 10_000)

    @staticmethod
    def sqrt_impact(
        price:       float,
        qty:         float,
        adv:         float,    # average daily volume in USD
        volatility:  float,    # daily vol (e.g. 0.03)
        eta:         float = 0.1,
    ) -> float:
        """
        Square-root impact model:
            impact = eta * vol * sqrt(qty_usd / adv)
        """
        qty_usd = qty * price
        impact  = eta * volatility * np.sqrt(qty_usd / (adv + 1e-9))
        return price * (1 + impact)

    @staticmethod
    def twap_schedule(
        total_qty:   float,
        n_slices:    int   = 10,
        interval_s:  float = 60.0,
    ) -> List[dict]:
        """
        Simple TWAP: split total_qty into n equal slices.
        Returns list of {qty, delay_s} dicts.
        """
        slice_qty = total_qty / n_slices
        return [
            {"qty": slice_qty, "delay_s": i * interval_s}
            for i in range(n_slices)
        ]


# ================================================================
# ORDER SIZER
# ================================================================

class OrderSizer:
    """
    Converts target portfolio weights into concrete order quantities.

    Handles
    -------
    - Net-change sizing (only trade the delta)
    - Minimum notional filter
    - Lot-size rounding (exchange precision)
    - Max single-order size cap
    """

    def __init__(
        self,
        min_notional:  float = 10.0,    # USD
        max_order_pct: float = 0.02,    # max 2% NAV per order
        lot_precision: int   = 6,       # decimal places
    ):
        self.min_notional  = min_notional
        self.max_order_pct = max_order_pct
        self.lot_precision = lot_precision

    def compute_orders(
        self,
        target_weights:  pd.Series,     # {symbol: weight}
        current_weights: pd.Series,     # {symbol: weight}
        prices:          pd.Series,     # {symbol: last_price}
        nav:             float,
    ) -> List[Order]:
        """
        Generate list of Orders needed to move from current to target.
        """
        orders   = []
        all_syms = target_weights.index.union(current_weights.index)

        for sym in all_syms:
            w_target  = float(target_weights.get(sym, 0))
            w_current = float(current_weights.get(sym, 0))
            delta_w   = w_target - w_current

            if abs(delta_w) < 1e-6:
                continue

            price = float(prices.get(sym, 0))
            if price <= 0:
                log.warning("No price for %s — skipping", sym)
                continue

            notional = abs(delta_w) * nav
            if notional < self.min_notional:
                continue

            # Cap per-order notional
            max_notional = self.max_order_pct * nav
            notional     = min(notional, max_notional)

            qty  = round(notional / price, self.lot_precision)
            side = "buy" if delta_w > 0 else "sell"

            orders.append(Order(symbol=sym, side=side, qty=qty, price=price))

        return orders


# ================================================================
# REBALANCE SCHEDULER
# ================================================================

class RebalanceScheduler:
    """
    Decides WHEN to rebalance based on configurable triggers:
    - Time-based (daily, weekly)
    - Drift-based (weights deviate > threshold)
    - Signal-based (factor score changes significantly)
    """

    def __init__(
        self,
        frequency:       str   = "daily",   # "daily" | "weekly"
        drift_threshold: float = 0.05,      # 5% weight drift
        signal_threshold:float = 0.20,      # 20% signal change
    ):
        self.frequency        = frequency
        self.drift_threshold  = drift_threshold
        self.signal_threshold = signal_threshold
        self._last_rebalance: Optional[datetime] = None
        self._last_signal:    Optional[pd.Series] = None

    def should_rebalance(
        self,
        current_weights: pd.Series,
        target_weights:  pd.Series,
        current_signal:  Optional[pd.Series] = None,
        now:             Optional[datetime]   = None,
    ) -> tuple[bool, str]:
        """
        Returns (True, reason) if rebalance is warranted.
        """
        now = now or datetime.now(timezone.utc)

        # --- Time trigger ---
        if self._last_rebalance is None:
            return True, "initial"

        elapsed = (now - self._last_rebalance).total_seconds()
        if self.frequency == "daily"  and elapsed >= 86_400:
            return True, "daily_schedule"
        if self.frequency == "weekly" and elapsed >= 7 * 86_400:
            return True, "weekly_schedule"

        # --- Drift trigger ---
        drift = (target_weights - current_weights.reindex(target_weights.index).fillna(0)).abs().sum()
        if drift > self.drift_threshold:
            return True, f"drift={drift:.2%}"

        # --- Signal change trigger ---
        if current_signal is not None and self._last_signal is not None:
            sig_chg = (
                current_signal
                .reindex(self._last_signal.index)
                .fillna(0)
                .sub(self._last_signal)
                .abs()
                .mean()
            )
            if sig_chg > self.signal_threshold:
                return True, f"signal_change={sig_chg:.2%}"

        return False, "no_trigger"

    def mark_rebalanced(
        self,
        signal: Optional[pd.Series] = None,
        now:    Optional[datetime]   = None,
    ) -> None:
        self._last_rebalance = now or datetime.now(timezone.utc)
        self._last_signal    = signal


# ================================================================
# PAPER EXECUTION ENGINE
# ================================================================

class PaperExecutionEngine:
    """
    Simulated execution for backtesting / paper trading.
    Fills orders instantly at last price + slippage.
    Tracks positions and NAV.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_bps:  float = 10.0,
        slippage_bps:    float = 5.0,
    ):
        self.nav        = initial_capital
        self.cash       = initial_capital
        self.commission = commission_bps / 10_000
        self.slippage   = slippage_bps   / 10_000
        self.positions: Dict[str, Position] = {}
        self.order_log: List[Order]          = []

    def execute(self, order: Order, market_price: float) -> Order:
        """Fill a single order and update positions."""
        fill = SlippageModel.fixed_bps(
            market_price,
            bps=(self.slippage * 10_000)
            if order.side == "buy"
            else -(self.slippage * 10_000),
        )
        cost   = order.qty * fill
        fee    = cost * self.commission

        if order.side == "buy":
            if self.cash < cost + fee:
                log.warning("Insufficient cash for %s", order)
                order.status = "cancelled"
                return order
            self.cash -= (cost + fee)
            pos = self.positions.setdefault(
                order.symbol, Position(order.symbol)
            )
            new_qty      = pos.qty + order.qty
            pos.avg_cost = (
                (pos.avg_cost * pos.qty + fill * order.qty) / new_qty
                if new_qty > 0 else 0
            )
            pos.qty          = new_qty
            pos.last_price   = fill

        elif order.side == "sell":
            pos = self.positions.get(order.symbol)
            if pos is None or pos.qty < order.qty:
                log.warning("Cannot sell %s — insufficient position", order)
                order.status = "cancelled"
                return order
            pos.qty    -= order.qty
            self.cash  += (cost - fee)
            pos.last_price = fill
            if pos.qty < 1e-9:
                del self.positions[order.symbol]

        order.fill_price = fill
        order.status     = "filled"
        self.order_log.append(order)
        return order

    def update_prices(self, prices: pd.Series) -> None:
        """Update last_price for all positions."""
        for sym, pos in self.positions.items():
            if sym in prices.index:
                pos.last_price = float(prices[sym])

    @property
    def portfolio_value(self) -> float:
        """Total NAV = cash + market value of positions."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    def snapshot(self) -> dict:
        return {
            "nav":       round(self.portfolio_value, 2),
            "cash":      round(self.cash, 2),
            "positions": {
                sym: {
                    "qty":       pos.qty,
                    "avg_cost":  pos.avg_cost,
                    "last_price":pos.last_price,
                    "pnl_pct":   round(pos.pnl_pct * 100, 2),
                }
                for sym, pos in self.positions.items()
            },
        }
