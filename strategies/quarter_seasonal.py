"""Quarter-of-year seasonal strategy.

Coarser-grained than MonthSeasonalStrategy, using calendar quarters (Q1–Q4)
to capture seasonal patterns in energy prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DEFAULT_SIGNAL = 0.0


class QuarterSeasonalStrategy(BacktestStrategy):
    """Forecast based on mean quarterly price-change signal (Q1–Q4)."""

    def __init__(self) -> None:
        self._quarterly_mean: dict[int, float] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        dates = pd.to_datetime(train_data["delivery_date"])
        quarters = dates.dt.quarter.values  # numpy array to avoid index issues
        changes = train_data["price_change_eur_mwh"].values
        quarterly: dict[int, list[float]] = {}
        for q, c in zip(quarters, changes):
            quarterly.setdefault(int(q), []).append(float(c))
        self._quarterly_mean = {q: float(np.mean(vals)) for q, vals in quarterly.items()}
        abs_means = np.abs(list(self._quarterly_mean.values()))
        self.skip_buffer = float(np.median(abs_means)) * 0.3 if len(abs_means) else 0.0

    def forecast(self, state: BacktestState) -> float:
        quarter = (state.delivery_date.month - 1) // 3 + 1
        mean_change = self._quarterly_mean.get(quarter, _DEFAULT_SIGNAL)
        return state.last_settlement_price + mean_change

    def reset(self) -> None:
        pass
