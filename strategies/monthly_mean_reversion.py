"""Monthly mean-reversion strategy.

Computes the average price for each calendar month during training.
When the current price is above the historical monthly average, expect
mean-reversion downward.  When below, expect reversion upward.

This captures strong seasonal patterns: winter months tend to have
higher prices (heating demand), summer months lower (solar surplus).

Signal:
    price > monthly_avg -> short (-1)  [overbought for this month]
    price <= monthly_avg -> long  (+1)  [underbought for this month]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_PRICE_COL = "price_mean"


class MonthlyMeanReversionStrategy(BacktestStrategy):
    """Trade based on deviation from historical monthly average price.

    When price is above the average for this calendar month, expect it
    to fall.  When below, expect it to rise.
    """

    def __init__(self) -> None:
        self._monthly_avg: dict[int, float] | None = None
        self._overall_avg: float = 50.0
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _PRICE_COL in train_data.columns and hasattr(train_data.index, "month"):
            monthly = train_data.groupby(train_data.index.month)[_PRICE_COL].mean()
            self._monthly_avg = monthly.to_dict()
            self._overall_avg = float(train_data[_PRICE_COL].mean())
        elif _PRICE_COL in train_data.columns:
            self._monthly_avg = {}
            self._overall_avg = float(train_data[_PRICE_COL].mean())
        else:
            self._monthly_avg = {}
            self._overall_avg = 50.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._monthly_avg is None:
            raise RuntimeError("MonthlyMeanReversionStrategy.forecast() called before fit()")

        month = state.delivery_date.month
        avg = self._monthly_avg.get(month, self._overall_avg)
        current = state.last_settlement_price

        direction = -1 if current > avg else 1
        return current + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
