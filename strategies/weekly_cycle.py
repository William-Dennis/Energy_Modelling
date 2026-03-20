"""Weekly cycle exploitation strategy for the challenge dashboard.

Exploits the significant positive autocorrelation at lag 7 (ACF = +0.297)
found in EDA (Phase 3, H5): price changes on the same day of the week tend
to repeat direction. This reflects the structural weekly demand cycle in
European electricity markets.

Signal:
    change_7d_ago = price_change from 7 days before delivery_date
    change_7d_ago > 0  → long  (+1)
    change_7d_ago < 0  → short (-1)
    change_7d_ago == 0 → skip  (None)
    insufficient history → skip (None)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class WeeklyCycleStrategy(BacktestStrategy):
    """Follow the same day-of-week's direction from last week.

    No fitting required — the signal is purely from recent history.
    """

    def fit(self, train_data: pd.DataFrame) -> None:
        pass

    def act(self, state: BacktestState) -> int | None:
        history = state.history
        if history.empty or len(history) < 7:
            return None

        change_7d_ago = float(history["price_change_eur_mwh"].iloc[-7])

        if change_7d_ago > 0:
            return 1
        if change_7d_ago < 0:
            return -1
        return None

    def forecast(self, state: BacktestState) -> float:
        history = state.history
        if history.empty or len(history) < 7:
            return state.last_settlement_price

        change_7d_ago = float(history["price_change_eur_mwh"].iloc[-7])
        return state.last_settlement_price + change_7d_ago

    def reset(self) -> None:
        pass
