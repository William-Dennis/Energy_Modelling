"""Blended day-of-week + composite signal strategy.

Combines the two strongest individual strategies:
  - DayOfWeekStrategy: exploits the structural weekly settlement-price cycle
    (Mon/Tue reliably high, Fri/Sat/Sun reliably low).
  - CompositeSignalStrategy: weighted z-score of the top EDA features
    (load forecast, wind, fossil generation).

Each strategy contributes its predicted price *delta* from the last
settlement price with equal weight.  When both signals point in the same
direction the blended forecast is amplified; when they disagree they
partially cancel, naturally reducing position size / conviction.

Signal:
    dow_delta       = DayOfWeek predicted change
    composite_delta = CompositeSignal predicted change
    forecast        = last_settlement_price + 0.5 * dow_delta + 0.5 * composite_delta
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.composite_signal import CompositeSignalStrategy
from strategies.day_of_week import DayOfWeekStrategy


class DowCompositeStrategy(BacktestStrategy):
    """Equal-weight blend of DayOfWeekStrategy and CompositeSignalStrategy.

    Both sub-strategies are fitted on the same training window.  At forecast
    time, each sub-strategy's predicted price delta is averaged to produce
    the blended forecast.
    """

    def __init__(self, dow_weight: float = 0.5, composite_weight: float = 0.5) -> None:
        """
        Parameters
        ----------
        dow_weight:
            Weight applied to the DayOfWeek price delta (default 0.5).
        composite_weight:
            Weight applied to the CompositeSignal price delta (default 0.5).
        """
        self._dow = DayOfWeekStrategy()
        self._composite = CompositeSignalStrategy()
        self._dow_weight = dow_weight
        self._composite_weight = composite_weight

    def fit(self, train_data: pd.DataFrame) -> None:
        self._dow.fit(train_data)
        self._composite.fit(train_data)

    def forecast(self, state: BacktestState) -> float:
        last = state.last_settlement_price
        dow_delta = self._dow.forecast(state) - last
        composite_delta = self._composite.forecast(state) - last
        blended_delta = self._dow_weight * dow_delta + self._composite_weight * composite_delta
        return last + blended_delta

    def reset(self) -> None:
        self._dow.reset()
        self._composite.reset()
