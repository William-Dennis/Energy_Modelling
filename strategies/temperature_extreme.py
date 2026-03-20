"""Temperature extreme strategy using non-linear demand response.

Hypothesis: Extreme temperatures (cold in winter, hot in summer) drive
demand spikes that push clearing prices up. This is a non-linear signal —
moderate temperatures have no effect, only extremes matter.

Signal:
    temp < P10 (extreme cold) OR temp > P90 (extreme heat) → long  (+1)
    P10 <= temp <= P90 (moderate)                          → short (-1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_TEMP_COL = "weather_temperature_2m_degc_mean"


class TemperatureExtremeStrategy(BacktestStrategy):
    """Long on extreme temperatures, short on moderate temperatures.

    Extreme cold increases heating demand; extreme heat increases cooling
    demand. Both push total load above baseline and elevate prices.
    The moderate band is a noise zone — price direction is uncertain.
    """

    def __init__(self) -> None:
        self._p10: float | None = None
        self._p90: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        temps = train_data[_TEMP_COL]
        self._p10 = float(np.percentile(temps, 10))
        self._p90 = float(np.percentile(temps, 90))
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._p10 is None or self._p90 is None:
            raise RuntimeError("TemperatureExtremeStrategy.forecast() called before fit()")
        temp = float(state.features[_TEMP_COL])
        direction = 1 if (temp < self._p10 or temp > self._p90) else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
