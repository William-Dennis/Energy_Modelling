"""Renewables penetration strategy.

renewable_penetration_pct = (wind + solar forecast) / load_forecast.

High renewable penetration suppresses prices via the merit-order effect —
zero-marginal-cost renewables displace fossil generation, reducing the
clearing price. This is related to net_demand but expressed as a ratio.

Signal:
    penetration >= median(training) → short (-1)  [high renewable share]
    penetration <  median(training) → long  (+1)  [low renewable share]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "renewable_penetration_pct"


class RenewablesPenetrationStrategy(BacktestStrategy):
    """Short when renewable share of load is above median (merit-order bearish)."""

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
            raise RuntimeError("RenewablesPenetrationStrategy.forecast() called before fit()")
        pct = float(state.features[_COL])
        direction = -1 if pct >= self._threshold else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
