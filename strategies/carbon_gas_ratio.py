"""Carbon-to-gas ratio regime strategy.

The ratio of carbon price to gas price signals which fuel is the
marginal cost driver.  A high ratio means carbon costs dominate,
favouring a switch from coal to gas and raising overall generation
costs (bullish for electricity).  A low ratio means gas costs dominate
and coal is relatively cheaper (bearish for electricity as cheaper coal
generation enters the merit order).

Signal:
    ratio > median(training) -> long  (+1)  [carbon expensive -> costs rise]
    ratio <= median(training) -> short (-1)  [gas expensive -> coal cheaper]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_CARBON_COL = "carbon_price_usd_mean"
_GAS_COL = "gas_price_usd_mean"


class CarbonGasRatioStrategy(BacktestStrategy):
    """Trade based on the carbon/gas price ratio.

    High carbon/gas ratio signals carbon-intensive generation is
    expensive -> electricity prices rise.  Low ratio signals gas-driven
    marginal costs dominate -> cheaper coal undercuts -> prices fall.
    """

    def __init__(self) -> None:
        self._median_ratio: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _CARBON_COL in train_data.columns and _GAS_COL in train_data.columns:
            gas = train_data[_GAS_COL].replace(0, float("nan"))
            ratio = train_data[_CARBON_COL] / gas
            self._median_ratio = float(ratio.median())
        else:
            self._median_ratio = 1.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_ratio is None:
            raise RuntimeError("CarbonGasRatioStrategy.forecast() called before fit()")

        carbon = float(state.features.get(_CARBON_COL, 0.0))
        gas = float(state.features.get(_GAS_COL, 0.0))
        ratio = carbon / gas if gas != 0 else self._median_ratio

        # High ratio -> carbon expensive -> electricity prices rise -> long
        direction = 1 if ratio > self._median_ratio else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
