"""NL net import flow signal strategy.

When DE is a heavy exporter to the Netherlands (negative net import = large
export), it reflects a day where DE had surplus supply, which suppresses
prices. The next day, lower production or higher demand tends to push prices
back up.

flow_nl_net_import_mw_mean: negative = DE exporting to NL
Correlation with direction: −0.192 (strongest flow signal).

Signal:
    flow_nl < median(training) → long  (+1)   [heavy NL export → expect rise]
    flow_nl >= median(training) → short (-1)  [light export or import]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_COL = "flow_nl_net_import_mw_mean"


class NLFlowSignalStrategy(BacktestStrategy):
    """Long when NL export was heavy yesterday (strong supply → price recovery)."""

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
            raise RuntimeError("NLFlowSignalStrategy.forecast() called before fit()")
        flow = float(state.features[_COL])
        direction = 1 if flow < self._threshold else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
