"""Always-short baseline for the challenge dashboard."""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class AlwaysShortStrategy(BacktestStrategy):
    """Always go short. Useful as a symmetry check against always-long."""

    def __init__(self) -> None:
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if "price_change_eur_mwh" in train_data.columns:
            self._mean_abs_change = float(train_data["price_change_eur_mwh"].abs().mean())
            if self._mean_abs_change <= 0:
                self._mean_abs_change = 1.0

    def forecast(self, state: BacktestState) -> float:
        return state.last_settlement_price - self._mean_abs_change

    def reset(self) -> None:
        pass
