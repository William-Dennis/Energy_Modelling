"""Lag-2 mean reversion strategy for the challenge dashboard.

Exploits the significant negative autocorrelation at lag 2 (ACF = -0.277)
found in EDA (Phase 3, H4): after a large price move, the move 2 days later
tends to reverse.

Signal:
    change_2d_ago = history's price_change 2 days before delivery_date
    |change_2d_ago| < threshold → skip (None)
    change_2d_ago >  threshold → short (-1)  (mean-revert large up)
    change_2d_ago < -threshold → long  (+1)  (mean-revert large down)

Threshold: median of |price_change_eur_mwh| from training data.
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class Lag2ReversionStrategy(BacktestStrategy):
    """Fade large price moves from 2 days ago.

    The threshold is the median absolute price change from training,
    ensuring we only trade when the lag-2 move was meaningful.
    """

    def fit(self, train_data: pd.DataFrame) -> None:
        abs_changes = train_data["price_change_eur_mwh"].abs()
        self.skip_buffer = float(abs_changes.median())

    def forecast(self, state: BacktestState) -> float:
        history = state.history
        if history.empty or len(history) < 2:
            return state.last_settlement_price

        change_2d_ago = float(history["price_change_eur_mwh"].iloc[-2])
        # Mean-reversion: forecast that the price will revert the lag-2 move
        return state.last_settlement_price - change_2d_ago

    def reset(self) -> None:
        pass
