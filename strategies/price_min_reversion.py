"""Price minimum mean-reversion strategy.

When yesterday's intraday minimum price was very low, it indicates extreme
supply surplus during off-peak hours. These extreme lows tend to mean-revert.
A low minimum suggests the market is oversupplied; prices typically recover.

price_min: yesterday's minimum hourly price (lagged, already available)
Correlation with direction: −0.143.

Signal:
    price_min < median(training) → long  (+1)   [low min → expect recovery]
    price_min >= median(training) → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "price_min"


class PriceMinReversionStrategy(BacktestStrategy):
    """Long when yesterday's price minimum was below median (mean-reversion)."""

    def __init__(self) -> None:
        self._threshold: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._threshold = float(train_data[_COL].median())
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._threshold is None:
            raise RuntimeError("PriceMinReversionStrategy.forecast() called before fit()")
        price_min = float(state.features[_COL])
        direction = 1 if price_min < self._threshold else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
