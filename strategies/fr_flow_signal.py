"""FR net import flow signal strategy.

When DE is a heavy exporter to France (negative net import), it reflects
a surplus supply day. The next day, supply/demand tends to rebalance,
pushing prices back toward the mean.

flow_fr_net_import_mw_mean: negative = DE exporting to FR
Correlation with direction: −0.099.

Signal:
    flow_fr < median(training) → long  (+1)   [heavy FR export → expect rise]
    flow_fr >= median(training) → short (-1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "flow_fr_net_import_mw_mean"


class FRFlowSignalStrategy(BacktestStrategy):
    """Long when FR export was heavy yesterday (supply surplus → price recovery)."""

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
            raise RuntimeError("FRFlowSignalStrategy.forecast() called before fit()")
        flow = float(state.features[_COL])
        direction = 1 if flow < self._threshold else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
