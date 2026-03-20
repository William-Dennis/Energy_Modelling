"""Solar forecast contrarian strategy.

Exploits the merit-order effect: high solar forecast suppresses midday
clearing prices (zero marginal cost supply displaces gas/coal generation).
Mirrors WindForecastStrategy but uses solar as the signal.

Signal:
    forecast_solar_mw_mean >= median(training) → short (-1)
    forecast_solar_mw_mean <  median(training) → long  (+1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_SOLAR_COL = "forecast_solar_mw_mean"


class SolarForecastStrategy(BacktestStrategy):
    """Go short when solar forecast is high, long when low.

    The threshold is the median of the solar forecast from training data.
    Solar has a strong seasonal profile (summer/winter) that is orthogonal
    to the wind signal, providing diversified merit-order coverage.
    """

    def __init__(self) -> None:
        self._threshold: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._threshold = float(train_data[_SOLAR_COL].median())
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._threshold is None:
            raise RuntimeError("SolarForecastStrategy.forecast() called before fit()")
        solar = float(state.features[_SOLAR_COL])
        direction = -1 if solar >= self._threshold else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
