"""Radiation-solar strategy using shortwave radiation as a price signal.

High shortwave radiation yesterday implies strong solar generation today,
which pushes supply up and prices down.  Low radiation implies weaker solar,
supporting higher prices.

The threshold is the median radiation observed during training.

Signal:
    radiation >= median(training) -> short (-1)  [more solar -> lower prices]
    radiation <  median(training) -> long  (+1)  [less solar -> higher prices]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_RAD_COL = "weather_shortwave_radiation_wm2_mean"


class RadiationSolarStrategy(BacktestStrategy):
    """Trade based on yesterday's shortwave radiation level.

    Above-median radiation forecasts lower prices (solar surplus);
    below-median radiation forecasts higher prices (solar deficit).
    """

    def __init__(self) -> None:
        self._median_radiation: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _RAD_COL in train_data.columns:
            self._median_radiation = float(train_data[_RAD_COL].median())
        else:
            self._median_radiation = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_radiation is None:
            raise RuntimeError("RadiationSolarStrategy.forecast() called before fit()")

        radiation = float(state.features.get(_RAD_COL, 0.0))

        # High radiation -> surplus solar -> prices fall -> short
        direction = -1 if radiation >= self._median_radiation else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
