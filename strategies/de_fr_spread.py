"""DE-FR cross-border spread strategy.

When DE-LU prices were lower than French prices yesterday (negative
de_fr_spread), European market coupling pulls DE prices upward today.
When DE was more expensive than FR, DE prices face downward convergence.

de_fr_spread = price_mean − price_fr_eur_mwh_mean  (yesterday, lagged)

Signal:
    de_fr_spread <= 0  (DE cheap vs FR) → long  (+1)
    de_fr_spread >  0  (DE dear vs FR)  → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "de_fr_spread"


class DEFRSpreadStrategy(BacktestStrategy):
    """Long when DE is cheaper than France yesterday (spread ≤ 0), short when dear."""

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
            raise RuntimeError("DEFRSpreadStrategy.forecast() called before fit()")
        spread = float(state.features[_COL])
        direction = 1 if spread <= 0.0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
