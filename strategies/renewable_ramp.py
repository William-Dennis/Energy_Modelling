"""Renewable ramp strategy.

Measures the day-over-day change in total renewable generation
(wind onshore + wind offshore + solar).  A large increase in
renewables represents a supply ramp-up that pushes prices down.
A large decrease (ramp-down) reduces supply and pushes prices up.

Uses the price history to estimate yesterday's renewable generation
vs the current day.

Signal:
    renewable_today > renewable_yesterday * (1 + threshold) -> short
    renewable_today < renewable_yesterday * (1 - threshold) -> long
    otherwise -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_WIND_ON_COL = "gen_wind_onshore_mw_mean"
_WIND_OFF_COL = "gen_wind_offshore_mw_mean"
_SOLAR_COL = "gen_solar_mw_mean"
_RAMP_THRESHOLD = 0.10  # 10% change triggers a signal


class RenewableRampStrategy(BacktestStrategy):
    """Trade based on day-over-day renewable generation change.

    Large ramp-up in renewables -> supply surplus -> short.
    Large ramp-down -> supply deficit -> long.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._mean_abs_change: float = 1.0

    def _total_renewable(self, row: pd.Series) -> float:
        wind_on = float(row.get(_WIND_ON_COL, 0.0))
        wind_off = float(row.get(_WIND_OFF_COL, 0.0))
        solar = float(row.get(_SOLAR_COL, 0.0))
        return wind_on + wind_off + solar

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("RenewableRampStrategy.forecast() called before fit()")

        current_renewable = self._total_renewable(state.features)
        history = state.history

        # Get yesterday's renewable from history
        if len(history) >= 1:
            last_row = history.iloc[-1]
            prev_renewable = self._total_renewable(last_row)
        else:
            return state.last_settlement_price  # neutral

        if prev_renewable <= 0:
            return state.last_settlement_price  # neutral

        change_pct = (current_renewable - prev_renewable) / prev_renewable

        if change_pct > _RAMP_THRESHOLD:
            direction = -1  # ramp-up -> surplus -> short
        elif change_pct < -_RAMP_THRESHOLD:
            direction = 1  # ramp-down -> deficit -> long
        else:
            return state.last_settlement_price  # neutral

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
