"""Volatility regime strategy: momentum in low-vol, mean-reversion in high-vol.

Hypothesis: High intra-day price volatility (measured by price_std) signals
market stress and uncertainty. After high-volatility days, prices tend to
mean-revert. After low-volatility days, prices tend to continue trending.
This reflects the well-documented volatility-clustering and mean-reversion
dynamics in electricity spot markets.

Signal (high vol = price_std > P75 of training):
    high vol + price went up   → short (mean-revert)
    high vol + price went down → long  (mean-revert)
    low  vol + price went up   → long  (momentum)
    low  vol + price went down → short (momentum)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_STD_COL = "price_std"
_CHANGE_COL = "price_change_eur_mwh"


class VolatilityRegimeStrategy(BacktestStrategy):
    """Regime-switching strategy: momentum below P75 vol, mean-reversion above.

    The volatility threshold is the 75th percentile of ``price_std`` from
    training data.  Direction is derived from yesterday's ``price_change_eur_mwh``.
    """

    def __init__(self) -> None:
        self._vol_threshold: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._vol_threshold = float(np.percentile(train_data[_STD_COL], 75))
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data[_CHANGE_COL].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._vol_threshold is None:
            raise RuntimeError("VolatilityRegimeStrategy.forecast() called before fit()")
        vol = float(state.features[_STD_COL])
        change = float(state.features[_CHANGE_COL])
        high_vol = vol > self._vol_threshold
        price_up = change >= 0.0
        # high-vol: mean-revert; low-vol: momentum
        if high_vol:
            direction = -1 if price_up else 1
        else:
            direction = 1 if price_up else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
