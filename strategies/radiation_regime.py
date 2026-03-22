"""Radiation regime strategy.

Classifies days into radiation regimes:
- High radiation (> p75): strong solar output -> bearish
- Low radiation (< p25): weak solar output -> bullish
- Medium radiation: neutral

Uses ``weather_shortwave_radiation_wm2_mean``.  Different from
RadiationSolar (which uses a binary median split); this strategy
uses a three-regime classification with a wider neutral band.

Signal:
    radiation > p75(training) -> short (-1)  [strong solar surplus]
    radiation < p25(training) -> long  (+1)  [weak solar deficit]
    p25 <= radiation <= p75   -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_RAD_COL = "weather_shortwave_radiation_wm2_mean"


class RadiationRegimeStrategy(BacktestStrategy):
    """Trade based on radiation quartile regimes.

    Top quartile -> solar surplus -> short.
    Bottom quartile -> solar deficit -> long.
    Middle 50% -> neutral.
    """

    def __init__(self) -> None:
        self._p25_rad: float | None = None
        self._p75_rad: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _RAD_COL in train_data.columns:
            self._p25_rad = float(train_data[_RAD_COL].quantile(0.25))
            self._p75_rad = float(train_data[_RAD_COL].quantile(0.75))
        else:
            self._p25_rad = 0.0
            self._p75_rad = float("inf")

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._p25_rad is None or self._p75_rad is None:
            raise RuntimeError("RadiationRegimeStrategy.forecast() called before fit()")

        radiation = float(state.features.get(_RAD_COL, 0.0))

        if radiation > self._p75_rad:
            direction = -1  # strong solar -> short
        elif radiation < self._p25_rad:
            direction = 1  # weak solar -> long
        else:
            return state.last_settlement_price  # neutral

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
