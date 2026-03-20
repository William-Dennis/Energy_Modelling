"""Combined fuel index trend strategy.

Uses both gas and carbon 3-day trends to form a composite fuel cost
momentum signal. Both inputs push electricity prices in the same
direction via the marginal cost of gas-fired generation.

Signal:
    gas_trend_3d + carbon_trend_3d > 0 → long  (+1)
    gas_trend_3d + carbon_trend_3d ≤ 0 → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_GAS_COL = "gas_trend_3d"
_CARBON_COL = "carbon_trend_3d"


class FuelIndexTrendStrategy(BacktestStrategy):
    """Long when combined gas + carbon trend is positive, short otherwise.

    Combines two correlated but independent trend signals for a slightly
    stronger overall commodity momentum signal (~+0.08 combined correlation).
    """

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
            raise RuntimeError("FuelIndexTrendStrategy.forecast() called before fit()")
        combined = float(state.features[_GAS_COL]) + float(state.features[_CARBON_COL])
        direction = 1 if combined > 0.0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
