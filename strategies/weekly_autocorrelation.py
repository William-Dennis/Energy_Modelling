"""Weekly autocorrelation strategy.

Uses the price level from 7 days ago as a predictor for today's price.
Electricity prices exhibit strong weekly seasonality (same day of week
tends to have similar demand patterns).  The strategy computes the
average 7-day price change during training and applies that as a
directional signal.

Unlike WeeklyCycle (which trades day-of-week direction patterns), this
strategy uses the actual price level from 7 days ago relative to the
current price.

Signal:
    current_price > price_7d_ago + median_weekly_change -> short (overbought)
    current_price <= price_7d_ago + median_weekly_change -> long  (underbought)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_PRICE_COL = "price_mean"


class WeeklyAutocorrelationStrategy(BacktestStrategy):
    """Trade based on 7-day price autocorrelation.

    If the current price is above where the 7-day lag pattern predicts,
    expect mean-reversion down.  If below, expect reversion up.
    """

    def __init__(self) -> None:
        self._median_weekly_change: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _PRICE_COL in train_data.columns and len(train_data) >= 8:
            prices = train_data[_PRICE_COL]
            weekly_change = prices.values[7:] - prices.values[:-7]
            self._median_weekly_change = float(pd.Series(weekly_change).median())
        else:
            self._median_weekly_change = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_weekly_change is None:
            raise RuntimeError("WeeklyAutocorrelationStrategy.forecast() called before fit()")

        current_price = state.last_settlement_price
        history = state.history

        # Try to get the price from 7 days ago
        if _PRICE_COL in history.columns and len(history) >= 7:
            price_7d_ago = float(history[_PRICE_COL].iloc[-7])
        else:
            # Fall back to neutral
            return current_price

        # Expected price based on 7-day lag pattern
        expected = price_7d_ago + self._median_weekly_change

        # Current above expected -> overbought -> short
        # Current below expected -> underbought -> long
        direction = -1 if current_price > expected else 1
        return current_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
