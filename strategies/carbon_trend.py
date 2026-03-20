"""Carbon price trend strategy.

Uses the 3-day momentum of carbon (ETS) prices to forecast direction.
Rising carbon costs increase the marginal cost of fossil generation,
especially coal and gas, pushing electricity prices up.

Signal:
    carbon_trend_3d > 0 → long  (+1)   [carbon rising → electricity rising]
    carbon_trend_3d ≤ 0 → short (-1)   [carbon flat/falling]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "carbon_trend_3d"


class CarbonTrendStrategy(BacktestStrategy):
    """Long when carbon price momentum is positive over 3 days, short otherwise."""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("CarbonTrendStrategy.forecast() called before fit()")
        trend = float(state.features[_COL])
        direction = 1 if trend > 0.0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
