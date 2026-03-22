"""Load-generation gap strategy.

Compares total electricity load to total available generation
(fossil + renewable).  A positive gap (load > generation) indicates
supply scarcity and upward price pressure.  A negative gap (generation
surplus) means downward price pressure.

The threshold is the median gap observed during training.

Signal:
    gap > median(training) -> long  (+1)  [scarcity -> prices rise]
    gap <= median(training) -> short (-1)  [surplus -> prices fall]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_LOAD_COL = "load_actual_mw_mean"
_FOSSIL_COL = "total_fossil_mw"
_WIND_ON_COL = "gen_wind_onshore_mw_mean"
_WIND_OFF_COL = "gen_wind_offshore_mw_mean"
_SOLAR_COL = "gen_solar_mw_mean"


class LoadGenerationGapStrategy(BacktestStrategy):
    """Trade based on the load vs generation gap.

    When load exceeds generation (scarcity), expect higher prices.
    When generation exceeds load (surplus), expect lower prices.
    """

    def __init__(self) -> None:
        self._median_gap: float | None = None
        self._mean_abs_change: float = 1.0

    def _compute_gap(self, row: pd.Series) -> float:
        load = float(row.get(_LOAD_COL, 0.0))
        fossil = float(row.get(_FOSSIL_COL, 0.0))
        wind_on = float(row.get(_WIND_ON_COL, 0.0))
        wind_off = float(row.get(_WIND_OFF_COL, 0.0))
        solar = float(row.get(_SOLAR_COL, 0.0))
        generation = fossil + wind_on + wind_off + solar
        return load - generation

    def fit(self, train_data: pd.DataFrame) -> None:
        required = [_LOAD_COL]
        if all(c in train_data.columns for c in required):
            gaps = train_data.apply(self._compute_gap, axis=1)
            self._median_gap = float(gaps.median())
        else:
            self._median_gap = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_gap is None:
            raise RuntimeError("LoadGenerationGapStrategy.forecast() called before fit()")

        gap = self._compute_gap(state.features)

        # Large gap -> scarcity -> prices rise -> long
        direction = 1 if gap > self._median_gap else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
