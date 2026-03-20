"""Load surprise signal strategy.

load_surprise = (today's load forecast) − (yesterday's actual load).

A positive surprise means today is expected to have higher demand than
yesterday's actual → more demand → bullish.
A negative surprise means demand expected to fall → bearish.

Signal:
    load_surprise > 0 → long  (+1)   [demand rising]
    load_surprise ≤ 0 → short (-1)   [demand flat or falling]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "load_surprise"


class LoadSurpriseStrategy(BacktestStrategy):
    """Long when today's load forecast exceeds yesterday's actual demand."""

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
            raise RuntimeError("LoadSurpriseStrategy.forecast() called before fit()")
        surprise = float(state.features[_COL])
        direction = 1 if surprise > 0.0 else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
