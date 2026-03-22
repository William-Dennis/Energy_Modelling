"""Weekend mean-reversion strategy.

Electricity prices typically drop on weekends due to lower industrial
demand.  This strategy expects weekend prices to revert toward a
lower weekend average and weekday prices to revert toward a higher
weekday average.

Uses the ``is_weekend`` feature and separate weekday/weekend training
averages.

Signal:
    Weekend: price > weekend_avg -> short, price <= weekend_avg -> long
    Weekday: price > weekday_avg -> short, price <= weekday_avg -> long
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_PRICE_COL = "price_mean"
_WEEKEND_COL = "is_weekend"


class WeekendMeanReversionStrategy(BacktestStrategy):
    """Trade based on weekday/weekend price mean-reversion.

    Prices tend to fall on weekends and rise on weekdays.  This
    strategy expects reversion toward the respective average.
    """

    def __init__(self) -> None:
        self._weekday_avg: float | None = None
        self._weekend_avg: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _PRICE_COL in train_data.columns and _WEEKEND_COL in train_data.columns:
            weekend_mask = train_data[_WEEKEND_COL].astype(bool)
            self._weekend_avg = float(train_data.loc[weekend_mask, _PRICE_COL].mean())
            self._weekday_avg = float(train_data.loc[~weekend_mask, _PRICE_COL].mean())
            # Handle NaN from empty groups
            if pd.isna(self._weekend_avg):
                self._weekend_avg = float(train_data[_PRICE_COL].mean())
            if pd.isna(self._weekday_avg):
                self._weekday_avg = float(train_data[_PRICE_COL].mean())
        elif _PRICE_COL in train_data.columns:
            avg = float(train_data[_PRICE_COL].mean())
            self._weekday_avg = avg
            self._weekend_avg = avg
        else:
            self._weekday_avg = 50.0
            self._weekend_avg = 50.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._weekday_avg is None or self._weekend_avg is None:
            raise RuntimeError("WeekendMeanReversionStrategy.forecast() called before fit()")

        is_weekend = bool(state.features.get(_WEEKEND_COL, 0))
        avg = self._weekend_avg if is_weekend else self._weekday_avg
        current = state.last_settlement_price

        direction = -1 if current > avg else 1
        return current + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
