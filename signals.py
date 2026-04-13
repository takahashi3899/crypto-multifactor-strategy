import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Tuple


class RegimeDetector:
    """
    Market regime classifier used to gate signal strength.

    Regimes
    -------
    1  : Bull  — trend-following signals amplified
    0  : Neutral
    -1 : Bear  — reduce gross exposure, invert mean-reversion

    Methods
    -------
    hmm_regime       : 2-state Gaussian HMM on rolling vol
    trend_filter     : Kalman-style HP-filter trend direction
    composite_regime : majority-vote across detectors
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    # ----------------------------------------------------------------
    # TREND FILTER  (Hodrick-Prescott proxy via Savitzky-Golay)
    # ----------------------------------------------------------------

    def trend_filter(self, prices: pd.Series, window: int = 21) -> pd.Series:
        """
        Smooth price with Savitzky-Golay filter.
        Returns +1 if smoothed slope > 0, -1 otherwise.
        """
        if len(prices) < window + 2:
            return pd.Series(0, index=prices.index)
        win = window if window % 2 == 1 else window + 1
        smoothed = savgol_filter(prices.values, win, polyorder=2)
        slope    = np.gradient(smoothed)
        regime   = np.sign(slope)
        return pd.Series(regime, index=prices.index)

    # ----------------------------------------------------------------
    # VOLATILITY REGIME  (low vol = bull, high vol = bear)
    # ----------------------------------------------------------------

    def vol_regime(
        self,
        returns:    pd.Series,
        short_win:  int = 10,
        long_win:   int = 60,
    ) -> pd.Series:
        """
        Compare short-term vol vs long-term vol.
        short < long * 0.8  -> bull (1)
        short > long * 1.2  -> bear (-1)
        else                -> neutral (0)
        """
        short_vol = returns.rolling(short_win).std()
        long_vol  = returns.rolling(long_win).std()
        regime    = pd.Series(0, index=returns.index)
        regime[short_vol < long_vol * 0.8] =  1
        regime[short_vol > long_vol * 1.2] = -1
        return regime

    # ----------------------------------------------------------------
    # MOMENTUM REGIME  (200-day MA filter)
    # ----------------------------------------------------------------

    def ma_regime(
        self,
        prices: pd.Series,
        fast:   int = 50,
        slow:   int = 200,
    ) -> pd.Series:
        """
        Classic dual-MA crossover regime.
        price > MA200 and MA50 > MA200 -> bull (1)
        price < MA200 and MA50 < MA200 -> bear (-1)
        """
        ma_fast = prices.rolling(fast).mean()
        ma_slow = prices.rolling(slow).mean()
        regime  = pd.Series(0, index=prices.index)
        regime[(prices > ma_slow) & (ma_fast > ma_slow)] =  1
        regime[(prices < ma_slow) & (ma_fast < ma_slow)] = -1
        return regime

    # ----------------------------------------------------------------
    # COMPOSITE REGIME  (majority vote)
    # ----------------------------------------------------------------

    def composite_regime(
        self,
        prices:  pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Combine trend_filter + vol_regime + ma_regime.
        Returns -1 / 0 / +1 via majority vote.
        """
        r1 = self.trend_filter(prices)
        r2 = self.vol_regime(returns)
        r3 = self.ma_regime(prices)
        combined = r1.add(r2, fill_value=0).add(r3, fill_value=0)
        return np.sign(combined).fillna(0).astype(int)

    # ----------------------------------------------------------------
    # EXPOSURE SCALAR  (map regime -> position scale)
    # ----------------------------------------------------------------

    @staticmethod
    def regime_scalar(regime: int) -> float:
        """
        Map regime to gross exposure multiplier.
        Bull  -> full exposure (1.0)
        Neutral -> reduced  (0.6)
        Bear  -> minimal    (0.2)
        """
        return {1: 1.0, 0: 0.6, -1: 0.2}.get(int(regime), 0.6)


class SignalAggregator:
    """
    Combines raw factor scores with regime overlay to produce
    final trade signals.

    Pipeline
    --------
    raw_factors -> z-score normalise -> IC-weight blend
                -> regime gate -> final signal [-1, 1]
    """

    def __init__(self, ic_halflife: int = 60):
        """
        ic_halflife : decay window (days) for rolling IC estimation
        """
        self.ic_halflife = ic_halflife
        self._ic_history: dict = {}  # factor_name -> list of IC values

    # ----------------------------------------------------------------
    # Z-SCORE NORMALISATION
    # ----------------------------------------------------------------

    @staticmethod
    def cross_sectional_zscore(factor: pd.Series) -> pd.Series:
        """
        Cross-sectional z-score across assets at a single point in time.
        Winsorised at +-3 sigma.
        """
        mu  = factor.mean()
        std = factor.std()
        if std < 1e-9:
            return pd.Series(0.0, index=factor.index)
        z = (factor - mu) / std
        return z.clip(-3, 3)

    # ----------------------------------------------------------------
    # INFORMATION COEFFICIENT WEIGHTING
    # ----------------------------------------------------------------

    def update_ic(
        self,
        factor_name: str,
        factor_vals: pd.Series,
        fwd_returns: pd.Series,
    ) -> float:
        """
        Compute rank IC (Spearman) between factor and forward returns.
        Maintain exponentially-weighted rolling average.
        """
        aligned = factor_vals.align(fwd_returns, join="inner")
        f, r    = aligned
        if len(f) < 5:
            return 0.0
        ic = float(f.rank().corr(r.rank()))
        history = self._ic_history.setdefault(factor_name, [])
        history.append(ic)
        # exponential decay
        weights = np.array(
            [0.5 ** (len(history) - 1 - i / self.ic_halflife)
             for i in range(len(history))]
        )
        weights /= weights.sum()
        return float(np.dot(weights, history))

    def get_ic_weight(self, factor_name: str) -> float:
        """Return latest IC-based weight for a factor (floored at 0)."""
        history = self._ic_history.get(factor_name, [])
        if not history:
            return 1.0   # equal weight until IC is established
        return max(history[-1], 0.0)

    # ----------------------------------------------------------------
    # COMPOSITE SIGNAL
    # ----------------------------------------------------------------

    def combine(
        self,
        factors:  dict,          # {name: pd.Series}
        regime:   int   = 0,
        fwd_ret:  pd.Series = None,
    ) -> pd.Series:
        """
        Blend multiple factor series into a single composite signal.

        Parameters
        ----------
        factors  : dict of {factor_name: raw_factor_series}
        regime   : current market regime (-1, 0, 1)
        fwd_ret  : optional forward returns for IC update

        Returns
        -------
        composite signal Series, normalised to [-1, 1]
        """
        if not factors:
            return pd.Series(dtype=float)

        weighted_signals = []
        for name, raw in factors.items():
            z  = self.cross_sectional_zscore(raw)
            # Update IC if forward returns provided
            if fwd_ret is not None:
                ic_w = self.update_ic(name, raw, fwd_ret)
            else:
                ic_w = self.get_ic_weight(name)
            weighted_signals.append(z * max(ic_w, 0.1))  # floor weight at 0.1

        composite = pd.concat(weighted_signals, axis=1).mean(axis=1)

        # Apply regime scalar
        scalar = RegimeDetector.regime_scalar(regime)
        composite = composite * scalar

        # Re-normalise to [-1, 1]
        abs_max = composite.abs().max()
        if abs_max > 1e-9:
            composite = composite / abs_max

        return composite.fillna(0)

    # ----------------------------------------------------------------
    # LONG-ONLY RANK SIGNAL
    # ----------------------------------------------------------------

    @staticmethod
    def rank_to_weights(
        signal:    pd.Series,
        top_n:     int   = None,
        long_only: bool  = True,
    ) -> pd.Series:
        """
        Convert composite signal to investable weights.
        Selects top_n assets by signal strength.
        """
        if signal.empty:
            return signal

        if long_only:
            signal = signal.clip(lower=0)

        if top_n and top_n < len(signal):
            threshold = signal.nlargest(top_n).min()
            signal    = signal.where(signal >= threshold, 0)

        total = signal.sum()
        if total > 1e-9:
            return signal / total
        return pd.Series(1 / len(signal), index=signal.index)
