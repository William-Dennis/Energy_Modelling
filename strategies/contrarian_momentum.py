"""Contrarian momentum strategy.

Combines two opposing approaches: trend-following when the trend is
strong, and contrarian (mean-reversion) when the trend is weak.

Uses the 3-day gas trend as the momentum indicator.  When the gas
trend is in the top or bottom quartile (strong trend), follow it.
When the gas trend is in the middle 50% (weak trend), go against
the last price change direction (contrarian).

Signal:
    strong gas trend (|trend| > p75): follow trend direction
    weak gas trend: contrarian on last price change
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_GAS_TREND_COL = "gas_trend_3d"
_CHANGE_COL = "price_change_eur_mwh"


class ContrarianMomentumStrategy(BacktestStrategy):
    """Follow strong trends; fade weak trends.

    Strong gas trend -> momentum-follow.
    Weak gas trend -> contrarian on recent price change.
    """

    def __init__(self) -> None:
        self._p75_abs_trend: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _GAS_TREND_COL in train_data.columns:
            self._p75_abs_trend = float(train_data[_GAS_TREND_COL].abs().quantile(0.75))
        else:
            self._p75_abs_trend = float("inf")

        if _CHANGE_COL in train_data.columns:
            mac = float(train_data[_CHANGE_COL].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data[_CHANGE_COL].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._p75_abs_trend is None:
            raise RuntimeError("ContrarianMomentumStrategy.forecast() called before fit()")

        gas_trend = float(state.features.get(_GAS_TREND_COL, 0.0))
        change = float(state.features.get(_CHANGE_COL, 0.0))

        if abs(gas_trend) > self._p75_abs_trend:
            # Strong trend: follow it
            direction = 1 if gas_trend > 0 else -1
        else:
            # Weak trend: go contrarian on last price change
            if change == 0:
                return state.last_settlement_price  # neutral
            direction = -1 if change > 0 else 1

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
