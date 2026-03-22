"""Intraday range strategy using price_max and price_min.

High intraday range (price_max - price_min) signals a volatile day.
After volatile days, prices tend to mean-revert toward the midpoint.
After calm days (narrow range), trend-following is more effective.

Signal:
    range >= p75(training) -> mean-reversion: forecast midpoint of max/min
    range <= p25(training) -> trend-follow:   forecast in the direction of
                              yesterday's close relative to midpoint
    otherwise              -> neutral (forecast = last_settlement_price)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_MAX_COL = "price_max"
_MIN_COL = "price_min"
_RANGE_COL = "price_range"


class IntradayRangeStrategy(BacktestStrategy):
    """Trade based on yesterday's intraday price range.

    Wide range -> mean-reversion toward midpoint.
    Narrow range -> trend-follow in the direction of close vs midpoint.
    """

    def __init__(self) -> None:
        self._p25_range: float | None = None
        self._p75_range: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _RANGE_COL in train_data.columns:
            self._p25_range = float(train_data[_RANGE_COL].quantile(0.25))
            self._p75_range = float(train_data[_RANGE_COL].quantile(0.75))
        else:
            self._p25_range = 0.0
            self._p75_range = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.35

    def forecast(self, state: BacktestState) -> float:
        if self._p25_range is None or self._p75_range is None:
            raise RuntimeError("IntradayRangeStrategy.forecast() called before fit()")

        price_max = float(state.features.get(_MAX_COL, 0.0))
        price_min = float(state.features.get(_MIN_COL, 0.0))
        price_range = float(state.features.get(_RANGE_COL, 0.0))
        midpoint = (price_max + price_min) / 2.0

        if price_range >= self._p75_range:
            # Volatile day -> mean-revert toward midpoint
            # Forecast nudges toward the midpoint relative to settlement
            if midpoint > state.last_settlement_price:
                direction = 1
            elif midpoint < state.last_settlement_price:
                direction = -1
            else:
                return state.last_settlement_price
            return state.last_settlement_price + direction * self._mean_abs_change

        if price_range <= self._p25_range:
            # Calm day -> trend-follow
            # If settlement was above midpoint, expect continuation up
            if state.last_settlement_price > midpoint:
                direction = 1
            elif state.last_settlement_price < midpoint:
                direction = -1
            else:
                return state.last_settlement_price
            return state.last_settlement_price + direction * self._mean_abs_change

        # Middle range -> no strong signal
        return state.last_settlement_price

    def reset(self) -> None:
        pass
