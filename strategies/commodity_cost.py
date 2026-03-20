"""Commodity cost strategy based on gas and carbon price levels.

Hypothesis: Rising gas and carbon prices increase the marginal cost of
gas-fired generation (the price-setting technology in most hours).
Higher fuel costs → higher clearing prices → long signal.

The fuel cost index combines gas price (heat-rate weighted) and carbon
price (emission-factor weighted) to approximate CCGT variable cost.

Signal:
    fuel_index >= median(training) → long  (+1)
    fuel_index <  median(training) → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_GAS_COL = "gas_price_usd_mean"
_CARBON_COL = "carbon_price_usd_mean"

# CCGT heat rate (MWh_th / MWh_e) and CO2 emission factor (tCO2 / MWh_e)
_GAS_HEAT_RATE = 7.5
_EMISSION_FACTOR = 0.37


def _fuel_index(gas: float, carbon: float) -> float:
    return gas * _GAS_HEAT_RATE + carbon * _EMISSION_FACTOR


class CommodityCostStrategy(BacktestStrategy):
    """Go long when the CCGT fuel cost index is above its training median.

    The index approximates the variable cost of a combined-cycle gas turbine:
    fuel_index = gas_price × 7.5 + carbon_price × 0.37
    """

    def __init__(self) -> None:
        self._threshold: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        idx = train_data[_GAS_COL] * _GAS_HEAT_RATE + train_data[_CARBON_COL] * _EMISSION_FACTOR
        self._threshold = float(idx.median())
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._threshold is None:
            raise RuntimeError("CommodityCostStrategy.forecast() called before fit()")
        gas = float(state.features[_GAS_COL])
        carbon = float(state.features[_CARBON_COL])
        idx = _fuel_index(gas, carbon)
        direction = 1 if idx >= self._threshold else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
