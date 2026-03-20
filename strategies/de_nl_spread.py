"""DE-NL cross-border spread strategy.

When DE-LU prices were lower than Dutch prices yesterday (negative
de_nl_spread), European market coupling pulls DE prices upward today.

de_nl_spread = price_mean − price_nl_eur_mwh_mean  (yesterday, lagged)

Signal:
    de_nl_spread <= 0  (DE cheap vs NL) → long  (+1)
    de_nl_spread >  0  (DE dear vs NL)  → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "de_nl_spread"


class DENLSpreadStrategy(BacktestStrategy):
    """Long when DE is cheaper than Netherlands yesterday (spread ≤ 0)."""

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
            raise RuntimeError("DENLSpreadStrategy.forecast() called before fit()")
        spread = float(state.features[_COL])
        direction = 1 if spread <= 0.0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
