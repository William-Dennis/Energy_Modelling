"""Load forecast level strategy for the challenge dashboard.

Exploits the strongest single-feature predictor found in EDA (Phase 3, H3):
load_forecast_mw_mean has +0.234 correlation with price direction. Higher
demand forecast → higher clearing price → price more likely to rise.

Signal:
    load_forecast_mw_mean >= median(training) → long  (+1)
    load_forecast_mw_mean <  median(training) → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_LOAD_COL = "load_forecast_mw_mean"


class LoadForecastStrategy(BacktestStrategy):
    """Go long when load forecast is high, short when low.

    The threshold is the median of load_forecast_mw_mean from the
    training data.
    """

    def __init__(self) -> None:
        self._threshold: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._threshold = float(train_data[_LOAD_COL].median())
        if "price_change_eur_mwh" in train_data.columns:
            self._mean_abs_change = float(train_data["price_change_eur_mwh"].abs().mean())
            if self._mean_abs_change <= 0:
                self._mean_abs_change = 1.0

    def act(self, state: BacktestState) -> int | None:
        if self._threshold is None:
            msg = "LoadForecastStrategy.act() called before fit()"
            raise RuntimeError(msg)
        load = float(state.features[_LOAD_COL])
        return 1 if load >= self._threshold else -1

    def forecast(self, state: BacktestState) -> float:
        direction = self.act(state)
        if direction is None:
            return state.last_settlement_price
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
