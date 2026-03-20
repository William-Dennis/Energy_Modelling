"""Wind forecast error signal strategy.

wind_forecast_error = (today's wind forecast) − (yesterday's actual wind generation).

A positive error means today's wind forecast is higher than yesterday's
actual output → more supply expected → bearish (prices should fall).
A negative error means wind is expected to underperform yesterday → bullish.

Signal:
    wind_forecast_error > 0 → short (-1)   [more wind supply expected]
    wind_forecast_error ≤ 0 → long  (+1)   [less wind supply expected]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "wind_forecast_error"


class WindForecastErrorStrategy(BacktestStrategy):
    """Short when today's wind forecast exceeds yesterday's actual (supply glut)."""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("WindForecastErrorStrategy.forecast() called before fit()")
        error = float(state.features[_COL])
        direction = -1 if error > 0.0 else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
