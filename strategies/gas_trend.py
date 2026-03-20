"""Gas price trend strategy.

Uses the 3-day momentum of gas prices to forecast direction.
Rising gas costs increase the marginal cost of gas-fired generation,
pushing electricity prices up.

Signal:
    gas_trend_3d > 0 → long  (+1)   [gas rising → electricity rising]
    gas_trend_3d ≤ 0 → short (-1)   [gas flat/falling]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "gas_trend_3d"


class GasTrendStrategy(BacktestStrategy):
    """Long when gas price momentum is positive over 3 days, short otherwise."""

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
            raise RuntimeError("GasTrendStrategy.forecast() called before fit()")
        trend = float(state.features[_COL])
        direction = 1 if trend > 0.0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
