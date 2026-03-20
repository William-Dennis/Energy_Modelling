"""Boosted signal ensemble: amplify when spread/fuel signals align with ML.

Combines a spread signal (DEFRSpread) with the GradientBoosting classifier.
When both agree, the signal is amplified; when they disagree, skip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.de_fr_spread import DEFRSpreadStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy


class BoostedSpreadMLStrategy(BacktestStrategy):
    """Trade only when DE-FR spread and GBM classifier agree on direction."""

    def __init__(self) -> None:
        self._spread = DEFRSpreadStrategy()
        self._gbm = GradientBoostingStrategy()

    def fit(self, train_data: pd.DataFrame) -> None:
        self._spread.fit(train_data)
        self._gbm.fit(train_data)
        self.skip_buffer = float(np.mean([self._spread.skip_buffer, self._gbm.skip_buffer]))

    def _direction(self, strategy: BacktestStrategy, state: BacktestState) -> float:
        f = float(strategy.forecast(state))
        diff = f - state.last_settlement_price
        if abs(diff) <= strategy.skip_buffer:
            return 0.0
        return 1.0 if diff > 0 else -1.0

    def forecast(self, state: BacktestState) -> float:
        d_spread = self._direction(self._spread, state)
        d_gbm = self._direction(self._gbm, state)
        if d_spread == d_gbm and d_spread != 0.0:
            return state.last_settlement_price + d_spread
        return state.last_settlement_price  # disagreement → skip

    def reset(self) -> None:
        self._spread.reset()
        self._gbm.reset()
