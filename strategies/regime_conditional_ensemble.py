"""Regime-conditional ensemble strategy.

Selects between two sub-ensembles depending on the volatility regime:
- HIGH vol (rolling_vol_7d > median): uses ML classifiers (higher adaptability)
- LOW vol (rolling_vol_7d <= median): uses rule-based strategies (more stable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy
from strategies.load_forecast import LoadForecastStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.random_forest_direction import RandomForestStrategy
from strategies.wind_forecast import WindForecastStrategy

_DEFAULT_VOL = 5.0


class RegimeConditionalEnsembleStrategy(BacktestStrategy):
    """ML ensemble in high-vol, rule-based ensemble in low-vol regimes."""

    def __init__(self) -> None:
        self._vol_threshold: float = _DEFAULT_VOL
        self._ml_members: list[BacktestStrategy] = []
        self._rule_members: list[BacktestStrategy] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        if "rolling_vol_7d" in train_data.columns:
            self._vol_threshold = float(
                np.median(train_data["rolling_vol_7d"].fillna(_DEFAULT_VOL))
            )

        self._ml_members = [
            LogisticDirectionStrategy(),
            RandomForestStrategy(),
            GradientBoostingStrategy(),
        ]
        self._rule_members = [
            WindForecastStrategy(),
            GasTrendStrategy(),
            LoadForecastStrategy(),
        ]
        for m in self._ml_members + self._rule_members:
            m.fit(train_data)

        buffers = [m.skip_buffer for m in self._ml_members + self._rule_members]
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
        vol = float(state.features.get("rolling_vol_7d", _DEFAULT_VOL))
        members = self._ml_members if vol > self._vol_threshold else self._rule_members
        vote = self._vote(members, state)
        if vote > 0:
            return state.last_settlement_price + 1.0
        if vote < 0:
            return state.last_settlement_price - 1.0
        return state.last_settlement_price

    def reset(self) -> None:
        for m in self._ml_members + self._rule_members:
            m.reset()
