"""Always-short baseline for the challenge dashboard."""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class AlwaysShortStrategy(BacktestStrategy):
    """Always go short. Useful as a symmetry check against always-long."""

    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def act(self, state: BacktestState) -> int | None:
        return -1

    def reset(self) -> None:
        pass
