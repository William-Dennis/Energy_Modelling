"""Renewables surplus strategy — extreme merit-order signal.

Hypothesis: When combined wind + solar forecasts are very high, the
merit-order effect is so strong that prices fall significantly. Conversely,
a renewables drought forces expensive fossil dispatch and pushes prices up.
This is a non-linear effect — only extreme values provide strong signal.

Signal:
    combined > P80 (renewables flood)   → short (-1)
    combined < P20 (renewables drought) → long  (+1)
    P20 <= combined <= P80 (moderate)   → skip  (None)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_OFFSHORE_COL = "forecast_wind_offshore_mw_mean"
_ONSHORE_COL = "forecast_wind_onshore_mw_mean"
_SOLAR_COL = "forecast_solar_mw_mean"


class RenewablesSurplusStrategy(BacktestStrategy):
    """High-selectivity strategy: only trade in extreme renewables regimes.

    P80 and P20 thresholds are computed on combined offshore + onshore wind
    + solar forecast from training data.  The middle 60% of days is skipped.
    """

    def __init__(self) -> None:
        self._p20: float | None = None
        self._p80: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        combined = train_data[_OFFSHORE_COL] + train_data[_ONSHORE_COL] + train_data[_SOLAR_COL]
        self._p20 = float(np.percentile(combined, 20))
        self._p80 = float(np.percentile(combined, 80))
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def act(self, state: BacktestState) -> int | None:
        """Override act() to return None cleanly in the moderate band."""
        if self._p20 is None or self._p80 is None:
            raise RuntimeError("RenewablesSurplusStrategy.act() called before fit()")
        combined = (
            float(state.features[_OFFSHORE_COL])
            + float(state.features[_ONSHORE_COL])
            + float(state.features[_SOLAR_COL])
        )
        if combined > self._p80:
            return -1
        if combined < self._p20:
            return 1
        return None

    def forecast(self, state: BacktestState) -> float:
        """Forecast is not used directly; act() overrides the decision."""
        if self._p20 is None or self._p80 is None:
            raise RuntimeError("RenewablesSurplusStrategy.forecast() called before fit()")
        combined = (
            float(state.features[_OFFSHORE_COL])
            + float(state.features[_ONSHORE_COL])
            + float(state.features[_SOLAR_COL])
        )
        if combined > self._p80:
            direction = -1
        elif combined < self._p20:
            direction = 1
        else:
            return state.last_settlement_price  # neutral → act() returns None
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
