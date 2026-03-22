"""Spark-spread strategy.

The spark spread measures the gap between electricity price and the
fuel cost of gas-fired generation:

    spark_spread = electricity_price - gas_price * heat_rate

When the spark spread is above its historical median, gas plants are
very profitable and more generation will come online, pushing prices
down.  When below median, gas plants are unprofitable and may shut
down, reducing supply and pushing prices up.

A typical CCGT heat rate is ~7.0 MWh_th/MWh_el (simplified).  We use
USD gas price directly as a proxy since the conversion factor is
absorbed into the learned median.

Signal:
    spark > median(training) -> short (-1)  [gas gen profitable -> prices fall]
    spark <= median(training) -> long  (+1)  [gas gen marginal -> prices rise]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_ELEC_COL = "price_mean"
_GAS_COL = "gas_price_usd_mean"
_HEAT_RATE = 7.0  # approximate CCGT heat rate


class SparkSpreadStrategy(BacktestStrategy):
    """Trade based on the gas-to-electricity spark spread.

    High spark spread (electricity expensive vs gas) means profitable
    gas generation -> more supply -> prices fall.  Low spark spread
    means less gas supply -> prices rise.
    """

    def __init__(self) -> None:
        self._median_spark: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _ELEC_COL in train_data.columns and _GAS_COL in train_data.columns:
            spark = train_data[_ELEC_COL] - train_data[_GAS_COL] * _HEAT_RATE
            self._median_spark = float(spark.median())
        else:
            self._median_spark = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_spark is None:
            raise RuntimeError("SparkSpreadStrategy.forecast() called before fit()")

        elec = float(state.features.get(_ELEC_COL, 0.0))
        gas = float(state.features.get(_GAS_COL, 0.0))
        spark = elec - gas * _HEAT_RATE

        direction = -1 if spark > self._median_spark else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
