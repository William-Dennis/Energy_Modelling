"""Fossil dispatch contrarian strategy for the challenge dashboard.

Exploits the negative correlation between yesterday's fossil generation and
today's price direction (Phase 3, H6): high fossil dispatch signals expensive
generation conditions that tend to mean-revert.

Signal:
    combined_fossil = gas + hard_coal + lignite (all D-1 realised)
    combined_fossil >= median(training) → short (-1)
    combined_fossil <  median(training) → long  (+1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_GAS_COL = "gen_fossil_gas_mw_mean"
_COAL_COL = "gen_fossil_hard_coal_mw_mean"
_LIGNITE_COL = "gen_fossil_brown_coal_lignite_mw_mean"


class FossilDispatchStrategy(BacktestStrategy):
    """Go short when yesterday's fossil generation was high, long when low.

    The threshold is the median of combined fossil generation from the
    training data.
    """

    def __init__(self) -> None:
        self._threshold: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        combined = train_data[_GAS_COL] + train_data[_COAL_COL] + train_data[_LIGNITE_COL]
        self._threshold = float(combined.median())
        if "price_change_eur_mwh" in train_data.columns:
            self._mean_abs_change = float(train_data["price_change_eur_mwh"].abs().mean())
            if self._mean_abs_change <= 0:
                self._mean_abs_change = 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._threshold is None:
            msg = "FossilDispatchStrategy.forecast() called before fit()"
            raise RuntimeError(msg)
        combined = (
            float(state.features[_GAS_COL])
            + float(state.features[_COAL_COL])
            + float(state.features[_LIGNITE_COL])
        )
        direction = -1 if combined >= self._threshold else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
