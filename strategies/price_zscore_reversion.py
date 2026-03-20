"""Price z-score mean-reversion strategy.

Uses a 20-day rolling z-score of the DE-LU settlement price.
An elevated z-score signals the price is statistically high relative to
recent history and likely to revert downward; a depressed z-score
signals upward reversion.

Signal:
    z > +1  → short (-1)   [price elevated, expect fall]
    z < -1  → long  (+1)   [price depressed, expect rise]
    |z| ≤ 1 → skip  (None) [within normal band]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "price_zscore_20d"
_BAND = 1.0


class PriceZScoreReversionStrategy(BacktestStrategy):
    """Mean-reversion on 20-day price z-score. Skip when |z| ≤ 1."""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def act(self, state: BacktestState) -> int | None:
        if not self._fitted:
            raise RuntimeError("PriceZScoreReversionStrategy.act() called before fit()")
        z = float(state.features[_COL])
        if z > _BAND:
            return -1
        if z < -_BAND:
            return 1
        return None

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("PriceZScoreReversionStrategy.forecast() called before fit()")
        z = float(state.features[_COL])
        if z > _BAND:
            direction = -1
        elif z < -_BAND:
            direction = 1
        else:
            return state.last_settlement_price
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
