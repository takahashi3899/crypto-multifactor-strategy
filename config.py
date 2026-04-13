# =============================================================
# config.py — Central Configuration for Multi-Factor Strategy
# =============================================================

CONFIG = {
    # ── Universe ──────────────────────────────────────────────
    "universe": [
        "bitcoin", "ethereum", "solana", "binancecoin", "avalanche-2",
        "chainlink", "polkadot", "near", "injective-protocol", "sui",
        "aptos", "arbitrum", "optimism", "uniswap", "aave",
        "the-graph", "render-token", "fetch-ai", "worldcoin-wld", "sei-network"
    ],

    # ── Backtest window ───────────────────────────────────────
    "start_date": "2022-01-01",
    "end_date":   "2026-04-01",

    # ── Rebalance ─────────────────────────────────────────────
    "rebalance_freq": "W-MON",  # weekly on Monday

    # ── Factor weights (must sum to 1.0) ─────────────────────
    "factor_weights": {
        "momentum": 0.35,
        "value":    0.25,
        "carry":    0.25,
        "quality":  0.15,
    },

    # ── Portfolio construction ────────────────────────────────
    "n_longs":       5,      # top N assets to go long
    "n_shorts":      5,      # bottom N to short (0 = long-only)
    "long_only":     True,   # True = spot only, no shorting

    # ── Risk management ───────────────────────────────────────
    "target_vol_annual":   0.15,  # 15% annualised portfolio vol target
    "max_single_weight":   0.25,  # max 25% in any one asset
    "min_single_weight":   0.02,  # min weight threshold
    "hard_stop_drawdown": -0.10,  # -10% portfolio DD -> exit all
    "vol_scale_threshold": 2.0,   # realized vol > 2x baseline -> cut 50%

    # ── Transaction costs ─────────────────────────────────────
    "fee_per_trade": 0.001,  # 0.1% taker fee
    "slippage":      0.001,  # 0.1% slippage

    # ── Momentum params ───────────────────────────────────────
    "mom_windows": [20, 60, 120],
    "mom_weights": [0.5, 0.3, 0.2],
    "mom_skip":    5,  # skip last 5d (reversal filter)

    # ── Volatility windows ────────────────────────────────────
    "vol_window_short": 30,
    "vol_window_long":  90,

    # ── Regime filter ─────────────────────────────────────────
    "regime_ma_window": 200,  # BTC 200-day SMA
}
