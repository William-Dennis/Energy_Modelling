"""Gas–Carbon joint trend strategy.

Combines the 3-day gas price trend (``gas_trend_3d``) and the 3-day carbon
price trend (``carbon_trend_3d``) into a single signal.  Both rising = long
(higher fuel costs → higher electricity prices); both falling = short; mixed
= no trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DEFAULT = 0.0


class GasCarbonJointTrendStrategy(BacktestStrategy):
    """Go long when both gas and carbon trends are positive, short when both negative."""

    def __init__(self) -> None:
        self._gas_threshold: float = 0.0
        self._carbon_threshold: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if "gas_trend_3d" not in train_data.columns or "carbon_trend_3d" not in train_data.columns:
            self.skip_buffer = 0.0
            return
        gas = train_data["gas_trend_3d"].fillna(0.0)
        carbon = train_data["carbon_trend_3d"].fillna(0.0)
        self._gas_threshold = float(gas.abs().median()) * 0.2
        self._carbon_threshold = float(carbon.abs().median()) * 0.2
        # skip_buffer: median absolute price change
        self.skip_buffer = float(np.median(np.abs(train_data["price_change_eur_mwh"]))) * 0.3

    def forecast(self, state: BacktestState) -> float:
        gas = float(state.features.get("gas_trend_3d", _DEFAULT))
        carbon = float(state.features.get("carbon_trend_3d", _DEFAULT))
        if gas > self._gas_threshold and carbon > self._carbon_threshold:
            return state.last_settlement_price + 1.0  # both bullish
        if gas < -self._gas_threshold and carbon < -self._carbon_threshold:
            return state.last_settlement_price - 1.0  # both bearish
        return state.last_settlement_price  # mixed signal — no trade

    def reset(self) -> None:
        pass
