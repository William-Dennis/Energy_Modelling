"""Price volatility regime strategy.

Classifies the market into a HIGH or LOW volatility regime based on the
7-day rolling volatility (``rolling_vol_7d`` derived feature).  In high
volatility regimes the strategy trades with the recent momentum (last change
sign); in low volatility regimes it mean-reverts.

Uses the median ``rolling_vol_7d`` from training as the regime threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DEFAULT_VOL = 5.0  # fallback if feature absent


class VolatilityRegimeMLStrategy(BacktestStrategy):
    """Regime strategy: momentum in high-vol, mean-reversion in low-vol.

    *Different from* the existing VolatilityRegimeStrategy which uses a
    fixed threshold — this one learns the median vol threshold from training
    data and also learns the mean price change in each regime.
    """

    def __init__(self) -> None:
        self._vol_threshold: float = _DEFAULT_VOL
        self._high_vol_mean: float = 0.0
        self._low_vol_mean: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if "rolling_vol_7d" not in train_data.columns:
            self.skip_buffer = 0.0
            return
        vol = train_data["rolling_vol_7d"].fillna(_DEFAULT_VOL)
        self._vol_threshold = float(np.median(vol))
        changes = train_data["price_change_eur_mwh"]
        high_mask = vol > self._vol_threshold
        self._high_vol_mean = float(changes[high_mask].mean()) if high_mask.any() else 0.0
        self._low_vol_mean = float(changes[~high_mask].mean()) if (~high_mask).any() else 0.0
        self.skip_buffer = abs(self._high_vol_mean - self._low_vol_mean) * 0.2

    def forecast(self, state: BacktestState) -> float:
        vol = float(state.features.get("rolling_vol_7d", _DEFAULT_VOL))
        mean_change = self._high_vol_mean if vol > self._vol_threshold else self._low_vol_mean
        return state.last_settlement_price + mean_change

    def reset(self) -> None:
        pass
