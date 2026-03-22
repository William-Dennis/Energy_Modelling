"""Self-correcting forecast-error strategy.

Tracks the running error between the strategy's own previous forecast
and the realised last_settlement_price.  If the strategy has been
consistently over-forecasting (positive cumulative error), it adjusts
downward.  If under-forecasting, it adjusts upward.  This creates a
negative-feedback loop that adapts the forecast to recent market
behaviour.

The first forecast on any day with no history is simply the current
last_settlement_price (neutral).  The correction magnitude is scaled
by the mean absolute price change from training.

Signal:
    cumulative_error > 0  -> over-forecasting -> adjust down (short)
    cumulative_error < 0  -> under-forecasting -> adjust up (long)
    cumulative_error == 0 -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class ForecastPriceErrorStrategy(BacktestStrategy):
    """Self-correcting strategy that adapts based on its own forecast errors.

    Tracks cumulative forecast error and applies a proportional
    correction to reduce systematic bias.
    """

    def __init__(self) -> None:
        self._mean_abs_change: float = 1.0
        self._last_forecast: float | None = None
        self._cum_error: float = 0.0
        self._error_decay: float = 0.8  # exponential decay on error

    def fit(self, train_data: pd.DataFrame) -> None:
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.3

        # Reset tracking state at fit time
        self._last_forecast = None
        self._cum_error = 0.0

    def forecast(self, state: BacktestState) -> float:
        ref = state.last_settlement_price

        # Update cumulative error from previous forecast
        if self._last_forecast is not None:
            error = self._last_forecast - ref
            # Exponentially decaying cumulative error
            self._cum_error = self._error_decay * self._cum_error + error

        # Determine correction direction from cumulative error
        if abs(self._cum_error) < 1e-9:
            # No error accumulated yet -> neutral
            self._last_forecast = ref
            return ref

        # Normalise error: clamp correction to [-1, +1] of mac
        correction_sign = -1.0 if self._cum_error > 0 else 1.0
        correction_magnitude = min(abs(self._cum_error) / self._mean_abs_change, 1.0)

        forecast_price = ref + correction_sign * correction_magnitude * self._mean_abs_change
        self._last_forecast = forecast_price
        return forecast_price

    def reset(self) -> None:
        pass
