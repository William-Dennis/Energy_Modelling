"""Net demand threshold strategy.

Net demand = load forecast − (wind offshore + wind onshore + solar).
This is the residual demand that dispatchable (price-setting) generators
must cover. Higher net demand forces more expensive units onto the stack,
pushing clearing prices up.

Signal:
    net_demand_mw >= median(training) → long  (+1)
    net_demand_mw <  median(training) → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "net_demand_mw"


class NetDemandStrategy(BacktestStrategy):
    """Go long when net demand is above median, short when below.

    Net demand is the strongest single derived signal (~+0.30 correlation
    with price direction) because it directly measures how much expensive
    fossil generation the market must call.
    """

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
            raise RuntimeError("NetDemandStrategy.forecast() called before fit()")
        val = float(state.features[_COL])
        direction = 1 if val >= self._threshold else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
