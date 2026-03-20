"""Net demand momentum strategy.

``net_demand_mw`` = load – wind – solar.  When net demand is rising (current
value > rolling mean) the residual thermal generation requirement increases,
pushing prices up.  This strategy goes long when net demand is above its
rolling mean and short when below.

The rolling mean window is calibrated as the median of available history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DEFAULT_ND = 30_000.0  # MWh fallback


class NetDemandMomentumStrategy(BacktestStrategy):
    """Long when net demand is above its training mean; short when below.

    The threshold is the mean ``net_demand_mw`` from training data.
    """

    def __init__(self) -> None:
        self._mean_net_demand: float = _DEFAULT_ND
        self._high_mean: float = 0.0
        self._low_mean: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if "net_demand_mw" not in train_data.columns:
            self.skip_buffer = 0.0
            return
        nd = train_data["net_demand_mw"].fillna(_DEFAULT_ND)
        self._mean_net_demand = float(nd.mean())
        changes = train_data["price_change_eur_mwh"]
        high_mask = nd > self._mean_net_demand
        self._high_mean = float(changes[high_mask].mean()) if high_mask.any() else 0.0
        self._low_mean = float(changes[~high_mask].mean()) if (~high_mask).any() else 0.0
        self.skip_buffer = float(np.median(np.abs(changes))) * 0.3

    def forecast(self, state: BacktestState) -> float:
        nd = float(state.features.get("net_demand_mw", _DEFAULT_ND))
        if nd > self._mean_net_demand:
            return state.last_settlement_price + self._high_mean
        return state.last_settlement_price + self._low_mean

    def reset(self) -> None:
        pass
