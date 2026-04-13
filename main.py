#!/usr/bin/env python3
"""
Crypto Multi-Factor Strategy — Main Entry Point

Mimics a Millennium-style pod structure:
  1. Fetch OHLCV data via ccxt
  2. Compute 4 alpha factors: Momentum, Value, Carry, Quality
  3. Combine into composite signal
  4. Risk-overlay: vol targeting, position limits, hard-stop check
  5. Run walk-forward backtest
  6. Print performance summary

Usage:
    python main.py [--mode backtest|live] [--symbols BTC ETH SOL BNB ...]
"""

import argparse
import logging
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from config  import UNIVERSE, TRADING_CONFIG, RISK_CONFIG, DATA_CONFIG
from data    import DataFetcher
from factors import FactorEngine
from risk    import RiskManager
from backtester import Backtester

# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
# ------------------------------------------------------------------ #


def build_weights(
    prices:  pd.DataFrame,
    factor:  FactorEngine,
    risk:    RiskManager,
) -> pd.DataFrame:
    """
    Roll through the price history and produce a daily weight DataFrame.
    Looks back min_history days before producing first signal.
    """
    min_history = TRADING_CONFIG.get("min_history", 60)
    weights_list = []

    for i in range(len(prices)):
        if i < min_history:
            weights_list.append(pd.Series(0.0, index=prices.columns))
            continue

        window = prices.iloc[:i]
        returns = window.pct_change().dropna()

        # --- 1. Compute factors ---
        mom     = factor.momentum(window)
        val     = factor.value(window)
        carry   = factor.carry(window)
        quality = factor.quality(returns)

        # --- 2. Combine signals (equal-weight blend) ---
        composite = pd.concat([mom, val, carry, quality], axis=1).mean(axis=1)
        composite = composite.dropna()

        if composite.empty:
            weights_list.append(pd.Series(0.0, index=prices.columns))
            continue

        # --- 3. Cross-sectional z-score -> long-only weights ---
        z = (composite - composite.mean()) / (composite.std() + 1e-9)
        z = z.clip(lower=0)            # long-only
        raw_w = z / (z.sum() + 1e-9)  # normalise

        # --- 4. Risk constraints ---
        w = risk.apply_constraints(raw_w)
        w = risk.scale_to_vol_target(w, returns)

        weights_list.append(w.reindex(prices.columns).fillna(0))

    return pd.DataFrame(weights_list, index=prices.index)


def run_backtest(args) -> None:
    symbols = args.symbols or UNIVERSE[:8]   # default top-8
    log.info("Universe: %s", symbols)

    # --- Fetch data ---
    fetcher  = DataFetcher()
    timeframe = DATA_CONFIG.get("timeframe", "1d")
    limit     = DATA_CONFIG.get("limit",     500)

    log.info("Fetching OHLCV data (%s bars, %s)...", limit, timeframe)
    prices = fetcher.fetch_multi(symbols, timeframe=timeframe, limit=limit)

    if prices.empty:
        log.error("No price data fetched. Check network / exchange config.")
        sys.exit(1)

    log.info("Price matrix: %s rows x %s cols", *prices.shape)

    # --- Build engine objects ---
    factor_engine = FactorEngine(config=TRADING_CONFIG)
    risk_manager  = RiskManager(config=RISK_CONFIG)
    backtester    = Backtester(config=TRADING_CONFIG)

    # --- Generate weights ---
    log.info("Computing factor signals and building weight matrix...")
    weights = build_weights(prices, factor_engine, risk_manager)

    # --- Run backtest ---
    log.info("Running vectorised backtest...")
    results = backtester.run(prices, weights)

    # --- Hard-stop check (informational) ---
    threshold = RISK_CONFIG.get("hard_stop_dd", -0.20)
    if RiskManager.check_hard_stop(results["equity_curve"], threshold):
        log.warning(
            "Hard stop would have triggered! Max DD breached %.0f%%.",
            threshold * 100,
        )

    # --- Print results ---
    Backtester.print_stats(results["stats"])

    # --- Walk-forward validation ---
    if args.wf:
        log.info("Running walk-forward validation (train=365d, test=90d)...")

        def wf_weights_fn(train_prices: pd.DataFrame) -> pd.DataFrame:
            return build_weights(train_prices, factor_engine, risk_manager)

        oos_equity = backtester.walk_forward(
            prices,
            wf_weights_fn,
            train_days=365,
            test_days=90,
        )
        if not oos_equity.empty:
            oos_returns = oos_equity.pct_change().dropna()
            oos_stats   = backtester._compute_stats(
                oos_returns, oos_equity,
                pd.Series(0, index=oos_equity.index),
            )
            print("\n--- Walk-Forward OOS Performance ---")
            Backtester.print_stats(oos_stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto Multi-Factor Strategy"
    )
    parser.add_argument(
        "--mode", choices=["backtest", "live"], default="backtest",
        help="Run mode (default: backtest)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Asset symbols, e.g. BTC/USDT ETH/USDT",
    )
    parser.add_argument(
        "--wf", action="store_true",
        help="Also run walk-forward validation",
    )
    args = parser.parse_args()

    if args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "live":
        log.warning("Live trading mode is not yet implemented.")
        sys.exit(0)


if __name__ == "__main__":
    main()
