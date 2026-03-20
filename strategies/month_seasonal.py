"""Month-of-year seasonal strategy.

Energy prices show strong monthly seasonality (winter peaks, summer troughs).
This strategy learns the mean price change for each calendar month from
training data and uses that as the forecast signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DEFAULT_SIGNAL = 0.0


class MonthSeasonalStrategy(BacktestStrategy):
    """Forecast direction based on the mean monthly price-change signal.

    Training: compute mean ``price_change_eur_mwh`` per calendar month (1–12).
    Inference: look up the month of ``delivery_date`` and return
    ``last_settlement_price + monthly_mean``.
    """

    def __init__(self) -> None:
        self._monthly_mean: dict[int, float] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        dates = pd.to_datetime(train_data["delivery_date"])
        months = dates.dt.month.values  # numpy array to avoid index mismatch
        changes = train_data["price_change_eur_mwh"].values
        monthly: dict[int, list[float]] = {}
        for m, c in zip(months, changes):
            monthly.setdefault(int(m), []).append(float(c))
        self._monthly_mean = {m: float(np.mean(vals)) for m, vals in monthly.items()}
        abs_means = np.abs(list(self._monthly_mean.values()))
        self.skip_buffer = float(np.median(abs_means)) * 0.3 if len(abs_means) else 0.0

    def forecast(self, state: BacktestState) -> float:
        month = state.delivery_date.month
        mean_change = self._monthly_mean.get(month, _DEFAULT_SIGNAL)
        return state.last_settlement_price + mean_change

    def reset(self) -> None:
        pass
