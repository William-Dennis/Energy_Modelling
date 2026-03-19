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

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class Lag2ReversionStrategy(BacktestStrategy):
    """Fade large price moves from 2 days ago.

    The threshold is the median absolute price change from training,
    ensuring we only trade when the lag-2 move was meaningful.
    """

    def __init__(self) -> None:
        self._threshold: float | None = None

    def fit(self, train_data: pd.DataFrame) -> None:
        abs_changes = train_data["price_change_eur_mwh"].abs()
        self._threshold = float(abs_changes.median())

    def act(self, state: BacktestState) -> int | None:
        if self._threshold is None:
            msg = "Lag2ReversionStrategy.act() called before fit()"
            raise RuntimeError(msg)

        history = state.history
        if history.empty or len(history) < 2:
            return None

        # The second-to-last row in history is 2 days before delivery_date
        change_2d_ago = float(history["price_change_eur_mwh"].iloc[-2])

        if abs(change_2d_ago) < self._threshold:
            return None
        return -1 if change_2d_ago > 0 else 1

    def reset(self) -> None:
        pass
