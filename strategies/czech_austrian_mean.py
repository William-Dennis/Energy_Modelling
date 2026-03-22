"""Czech-Austrian mean spread strategy.

Uses the average of CZ and AT prices versus DE as a regional
convergence signal.  When DE is significantly more expensive than the
CZ-AT average, cross-border coupling drives DE prices down.  When DE is
cheap relative to both neighbours, expect upward pressure.

Signal:
    de_price - avg(cz, at) > median(training) -> short (-1)
    de_price - avg(cz, at) <= median(training) -> long  (+1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DE_COL = "price_mean"
_CZ_COL = "price_cz_eur_mwh_mean"
_AT_COL = "price_at_eur_mwh_mean"


class CzechAustrianMeanStrategy(BacktestStrategy):
    """Trade based on DE vs average CZ+AT price spread.

    When DE is expensive relative to CZ-AT mean (spread > median),
    expect convergence (DE falls).  When DE is cheap, expect it to rise.
    """

    def __init__(self) -> None:
        self._median_spread: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        cols = [_DE_COL, _CZ_COL, _AT_COL]
        if all(c in train_data.columns for c in cols):
            neighbour_avg = (train_data[_CZ_COL] + train_data[_AT_COL]) / 2.0
            spread = train_data[_DE_COL] - neighbour_avg
            self._median_spread = float(spread.median())
        else:
            self._median_spread = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_spread is None:
            raise RuntimeError("CzechAustrianMeanStrategy.forecast() called before fit()")

        de_price = float(state.features.get(_DE_COL, 0.0))
        cz_price = float(state.features.get(_CZ_COL, 0.0))
        at_price = float(state.features.get(_AT_COL, 0.0))
        neighbour_avg = (cz_price + at_price) / 2.0
        spread = de_price - neighbour_avg

        direction = -1 if spread > self._median_spread else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
