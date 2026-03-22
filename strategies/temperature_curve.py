"""Temperature curve strategy using quadratic temperature-demand model.

Models the U-shaped relationship between temperature and price changes:
extreme cold drives heating demand up; extreme heat drives cooling demand up;
moderate temperatures have neutral or slightly negative price impact.

During fit(), fits a simple quadratic (degree-2 polynomial) on temperature
vs price change from training data. During forecast(), evaluates the
quadratic to predict the expected price change.

Source: Phase 10g identified that Temperature Extreme exists but uses a
simple threshold; no existing strategy models the non-linear
temperature-demand relationship.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_TEMP_COL = "weather_temperature_2m_degc_mean"


class TemperatureCurveStrategy(BacktestStrategy):
    """Quadratic temperature-to-price-change model.

    Captures the U-shaped relationship: both cold and hot extremes push
    prices up, moderate temperatures are neutral.
    """

    def __init__(self) -> None:
        self._coeffs: np.ndarray | None = None  # [a, b, c] for a*x^2 + b*x + c
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        temps = train_data[_TEMP_COL].values.astype(float)
        if "price_change_eur_mwh" in train_data.columns:
            changes = train_data["price_change_eur_mwh"].values.astype(float)
            mac = float(np.nanmean(np.abs(changes)))
            self._mean_abs_change = mac if mac > 0 else 1.0
        else:
            changes = np.zeros_like(temps)

        # Remove NaN pairs
        mask = np.isfinite(temps) & np.isfinite(changes)
        temps_clean = temps[mask]
        changes_clean = changes[mask]

        if len(temps_clean) >= 3:
            self._coeffs = np.polyfit(temps_clean, changes_clean, 2)
        else:
            # Fallback: no signal
            self._coeffs = np.array([0.0, 0.0, 0.0])

        self.skip_buffer = self._mean_abs_change * 0.3

    def forecast(self, state: BacktestState) -> float:
        if self._coeffs is None:
            raise RuntimeError("TemperatureCurveStrategy.forecast() called before fit()")
        temp = float(state.features[_TEMP_COL])
        predicted_change = float(np.polyval(self._coeffs, temp))

        # Clip to prevent extreme forecasts
        max_change = self._mean_abs_change * 3.0
        predicted_change = max(-max_change, min(max_change, predicted_change))

        return state.last_settlement_price + predicted_change

    def reset(self) -> None:
        pass
