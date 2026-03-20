"""Dual regime ensemble: weekday vs weekend specialist sub-ensembles.

On weekdays (dow 0–4): uses ML classifiers that leverage market features.
On weekends (dow 5–6): uses calendar/regime strategies that capture
weekend-specific dynamics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.monday_effect import MondayEffectStrategy
from strategies.month_seasonal import MonthSeasonalStrategy
from strategies.quarter_seasonal import QuarterSeasonalStrategy
from strategies.random_forest_direction import RandomForestStrategy


class WeekdayWeekendEnsembleStrategy(BacktestStrategy):
    """ML classifiers on weekdays; calendar strategies on weekends."""

    def __init__(self) -> None:
        self._weekday_members: list[BacktestStrategy] = []
        self._weekend_members: list[BacktestStrategy] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._weekday_members = [
            LogisticDirectionStrategy(),
            RandomForestStrategy(),
            GradientBoostingStrategy(),
        ]
        self._weekend_members = [
            MonthSeasonalStrategy(),
            QuarterSeasonalStrategy(),
            MondayEffectStrategy(),
        ]
        for m in self._weekday_members + self._weekend_members:
            m.fit(train_data)
        buffers = [m.skip_buffer for m in self._weekday_members + self._weekend_members]
        self.skip_buffer = float(np.median(buffers)) if buffers else 0.0

    def _vote(self, members: list[BacktestStrategy], state: BacktestState) -> float:
        total = 0.0
        for m in members:
            f = float(m.forecast(state))
            diff = f - state.last_settlement_price
            if abs(diff) > m.skip_buffer:
                total += 1.0 if diff > 0 else -1.0
        return total

    def forecast(self, state: BacktestState) -> float:
        dow = int(state.features.get("dow_int", state.delivery_date.weekday()))
        members = self._weekend_members if dow >= 5 else self._weekday_members
        vote = self._vote(members, state)
        if vote > 0:
            return state.last_settlement_price + 1.0
        if vote < 0:
            return state.last_settlement_price - 1.0
        return state.last_settlement_price

    def reset(self) -> None:
        for m in self._weekday_members + self._weekend_members:
            m.reset()
