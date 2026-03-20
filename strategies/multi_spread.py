"""Multi-market average spread strategy.

Uses the average spread between DE-LU and all six neighbouring markets
(FR, NL, AT, CZ, PL, DK1). When DE is cheap relative to all neighbours,
convergence pressure is strongest.

de_avg_neighbour_spread = price_mean − mean(all 6 neighbour prices)

Signal:
    de_avg_neighbour_spread <= 0  (DE cheap vs avg neighbours) → long  (+1)
    de_avg_neighbour_spread >  0  (DE dear vs avg neighbours)  → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "de_avg_neighbour_spread"


class MultiSpreadStrategy(BacktestStrategy):
    """Long when DE is cheaper than the average of all 6 neighbours."""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("MultiSpreadStrategy.forecast() called before fit()")
        spread = float(state.features[_COL])
        direction = 1 if spread <= 0.0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
