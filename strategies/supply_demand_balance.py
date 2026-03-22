"""Supply-demand balance strategy.

Combines supply-side (total generation) and demand-side (load forecast)
signals into a balance score.  When forecast demand exceeds recent
generation, expect scarcity and higher prices.  When generation exceeds
forecast demand, expect surplus and lower prices.

Uses ``forecast_load_mw_mean`` for demand expectations and actual
generation columns for supply.

Signal:
    forecast_load > total_gen * (1 + buffer) -> long  (+1)  [scarcity]
    forecast_load < total_gen * (1 - buffer) -> short (-1)  [surplus]
    otherwise -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_LOAD_FCST_COL = "forecast_load_mw_mean"
_FOSSIL_COL = "total_fossil_mw"
_WIND_ON_COL = "gen_wind_onshore_mw_mean"
_WIND_OFF_COL = "gen_wind_offshore_mw_mean"
_SOLAR_COL = "gen_solar_mw_mean"
_NUCLEAR_COL = "gen_nuclear_mw_mean"
_BALANCE_BUFFER = 0.05  # 5% buffer for neutral zone


class SupplyDemandBalanceStrategy(BacktestStrategy):
    """Trade based on the supply-demand balance ratio.

    Forecast demand vs actual generation determines direction.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._mean_abs_change: float = 1.0

    def _total_gen(self, row: pd.Series) -> float:
        fossil = float(row.get(_FOSSIL_COL, 0.0))
        nuclear = float(row.get(_NUCLEAR_COL, 0.0))
        wind_on = float(row.get(_WIND_ON_COL, 0.0))
        wind_off = float(row.get(_WIND_OFF_COL, 0.0))
        solar = float(row.get(_SOLAR_COL, 0.0))
        return fossil + nuclear + wind_on + wind_off + solar

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("SupplyDemandBalanceStrategy.forecast() called before fit()")

        load_fcst = float(state.features.get(_LOAD_FCST_COL, 0.0))
        gen = self._total_gen(state.features)

        if gen <= 0:
            return state.last_settlement_price  # neutral

        ratio = load_fcst / gen

        if ratio > 1.0 + _BALANCE_BUFFER:
            direction = 1  # scarcity -> long
        elif ratio < 1.0 - _BALANCE_BUFFER:
            direction = -1  # surplus -> short
        else:
            return state.last_settlement_price  # neutral

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
