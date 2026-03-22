"""Offshore wind anomaly strategy.

Compares actual offshore wind generation (lagged) with the offshore wind
forecast.  When actual generation significantly exceeds the forecast,
there was a supply surplus yesterday that may depress prices today.
When actual fell short of the forecast, the supply shortfall supports
higher prices.

The threshold for "significant" deviation is the median absolute
forecast error observed during training.

Signal:
    actual - forecast >=  threshold -> short (-1)  [surplus supply]
    actual - forecast <= -threshold -> long  (+1)  [supply shortfall]
    otherwise                       -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_ACTUAL_COL = "gen_wind_offshore_mw_mean"
_FORECAST_COL = "forecast_wind_offshore_mw_mean"


class OffshoreWindAnomalyStrategy(BacktestStrategy):
    """Trade on the gap between actual and forecast offshore wind.

    Surplus generation (actual >> forecast) signals downward pressure;
    shortfall (actual << forecast) signals upward pressure.
    """

    def __init__(self) -> None:
        self._median_abs_error: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _ACTUAL_COL in train_data.columns and _FORECAST_COL in train_data.columns:
            error = train_data[_ACTUAL_COL] - train_data[_FORECAST_COL]
            self._median_abs_error = float(error.abs().median())
        else:
            self._median_abs_error = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_abs_error is None:
            raise RuntimeError("OffshoreWindAnomalyStrategy.forecast() called before fit()")

        actual = float(state.features.get(_ACTUAL_COL, 0.0))
        fcast = float(state.features.get(_FORECAST_COL, 0.0))
        anomaly = actual - fcast

        threshold = max(self._median_abs_error, 1.0)

        if anomaly >= threshold:
            direction = -1  # surplus supply -> lower prices
        elif anomaly <= -threshold:
            direction = 1  # supply shortfall -> higher prices
        else:
            return state.last_settlement_price  # no signal

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
