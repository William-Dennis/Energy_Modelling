"""Wind forecast contrarian strategy for the challenge dashboard.

Exploits the structural relationship between renewable supply and electricity
prices (Phase 3, H2): high wind forecasts push clearing prices down via the
merit order effect.

Signal:
    combined_wind = forecast_wind_offshore_mw_mean + forecast_wind_onshore_mw_mean
    combined_wind >= median(training) → short (-1)
    combined_wind <  median(training) → long  (+1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_OFFSHORE_COL = "forecast_wind_offshore_mw_mean"
_ONSHORE_COL = "forecast_wind_onshore_mw_mean"


class WindForecastStrategy(BacktestStrategy):
    """Go short when wind forecast is high, long when low.

    The threshold is the median of combined offshore + onshore wind
    forecasts from the training data.
    """

    def __init__(self) -> None:
        self._threshold: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        combined = train_data[_OFFSHORE_COL] + train_data[_ONSHORE_COL]
        self._threshold = float(combined.median())
        if "price_change_eur_mwh" in train_data.columns:
            self._mean_abs_change = float(train_data["price_change_eur_mwh"].abs().mean())
            if self._mean_abs_change <= 0:
                self._mean_abs_change = 1.0

    def act(self, state: BacktestState) -> int | None:
        if self._threshold is None:
            msg = "WindForecastStrategy.act() called before fit()"
            raise RuntimeError(msg)
        combined = float(state.features[_OFFSHORE_COL]) + float(state.features[_ONSHORE_COL])
        return -1 if combined >= self._threshold else 1

    def forecast(self, state: BacktestState) -> float:
        direction = self.act(state)
        if direction is None:
            return state.last_settlement_price
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
