# =============================================================
# data.py — Data Layer: fetch prices/volumes or use synthetic
# =============================================================

import pandas as pd
import numpy as np


def fetch_price_data(coins: list, start: str, end: str):
    """
    Fetch OHLCV data from CoinGecko.
    Returns (prices_df, volumes_df) with DatetimeIndex and coin_id columns.
    Falls back to synthetic data if CoinGecko is unavailable.
    """
    try:
        from pycoingecko import CoinGeckoAPI
        cg = CoinGeckoAPI()
        all_prices = {}
        all_volumes = {}
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts   = int(pd.Timestamp(end).timestamp())

        for coin in coins:
            try:
                data = cg.get_coin_market_chart_range_by_id(
                    id=coin, vs_currency='usd',
                    from_timestamp=start_ts, to_timestamp=end_ts
                )
                all_prices[coin]  = pd.Series(
                    {pd.Timestamp(x[0], unit='ms'): x[1] for x in data['prices']},
                    name=coin
                )
                all_volumes[coin] = pd.Series(
                    {pd.Timestamp(x[0], unit='ms'): x[1] for x in data['total_volumes']},
                    name=coin
                )
                print(f"  fetched {coin}")
            except Exception as e:
                print(f"  skipped {coin}: {e}")

        prices_df  = pd.DataFrame(all_prices).resample('D').last().ffill()
        volumes_df = pd.DataFrame(all_volumes).resample('D').last().ffill()
        return prices_df.astype(float), volumes_df.astype(float)

    except ImportError:
        print("[data] pycoingecko not found — using synthetic data")
        return generate_synthetic_data(coins, start, end)


def generate_synthetic_data(coins: list, start: str, end: str):
    """
    Generates realistic synthetic crypto price/volume data for offline testing.
    Uses correlated GBM with fat-tailed returns and bull/bear regimes.
    """
    dates   = pd.date_range(start, end, freq='D')
    n       = len(dates)
    n_coins = len(coins)
    np.random.seed(42)

    # Correlation matrix: all correlated to 'BTC' ~0.6, cross ~0.3
    corr = np.full((n_coins, n_coins), 0.3)
    np.fill_diagonal(corr, 1.0)
    corr[0, :] = corr[:, 0] = 0.6
    corr[0, 0] = 1.0
    L = np.linalg.cholesky(corr)

    # Fat-tailed daily returns (t-distribution df=4)
    raw        = np.random.standard_t(df=4, size=(n, n_coins)) * 0.035
    correlated = raw @ L.T

    # Bull/bear regimes
    regime               = np.ones(n) * 0.0008
    bear_start           = int(n * 0.25)
    bear_end             = int(n * 0.45)
    regime[bear_start:bear_end] = -0.0004
    correlated          += regime[:, None]

    # Starting prices (approximate 2022 levels)
    start_prices = np.array([
        45000, 3000, 150, 400, 80, 20, 15, 8, 12, 1.5,
        10, 1.2, 2.5, 8, 90, 0.2, 8, 1.5, 2, 0.5
    ])[:n_coins]

    prices_df = pd.DataFrame(
        index=dates, columns=coins,
        data=start_prices * np.exp(np.cumsum(correlated, axis=0))
    ).astype(float)

    vol_noise  = np.abs(np.random.normal(1.0, 0.3, (n, n_coins)))
    volumes_df = pd.DataFrame(
        index=dates, columns=coins,
        data=prices_df.values * start_prices * 1e6 * vol_noise
    ).astype(float)

    return prices_df, volumes_df
