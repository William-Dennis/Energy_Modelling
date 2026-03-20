"""Renewable penetration regime classifier strategy.

Segments the market into HIGH and LOW renewable penetration regimes (using
``renewable_penetration_pct``) and learns the mean price change in each
regime.  High renewable penetration → lower prices (merit-order effect).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DEFAULT_REN = 20.0  # fallback penetration pct


class RenewableRegimeStrategy(BacktestStrategy):
    """Price-change signal conditioned on renewable penetration regime.

    Threshold = median renewable penetration from training.
    High regime → mean change for high-ren days (typically negative).
    Low regime → mean change for low-ren days.
    """

    def __init__(self) -> None:
        self._threshold: float = _DEFAULT_REN
        self._high_ren_mean: float = 0.0
        self._low_ren_mean: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if "renewable_penetration_pct" not in train_data.columns:
            self.skip_buffer = 0.0
            return
        ren = train_data["renewable_penetration_pct"].fillna(_DEFAULT_REN)
        self._threshold = float(np.median(ren))
        changes = train_data["price_change_eur_mwh"]
        high_mask = ren > self._threshold
        self._high_ren_mean = float(changes[high_mask].mean()) if high_mask.any() else 0.0
        self._low_ren_mean = float(changes[~high_mask].mean()) if (~high_mask).any() else 0.0
        gap = abs(self._high_ren_mean - self._low_ren_mean)
        self.skip_buffer = gap * 0.2

    def forecast(self, state: BacktestState) -> float:
        ren = float(state.features.get("renewable_penetration_pct", _DEFAULT_REN))
        if ren > self._threshold:
            return state.last_settlement_price + self._high_ren_mean
        return state.last_settlement_price + self._low_ren_mean

    def reset(self) -> None:
        pass
