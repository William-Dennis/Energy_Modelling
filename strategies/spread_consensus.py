"""Spread consensus strategy.

Combines signals from multiple cross-border spreads (DE-FR, DE-NL,
DE-PL, DE-DK1) into a consensus view.  When all (or most) spreads
indicate DE is expensive, the consensus is short.  When most spreads
indicate DE is cheap, the consensus is long.

Signal:
    count(spread > median) >= 3 -> short (-1)  [DE expensive everywhere]
    count(spread <= median) >= 3 -> long  (+1)  [DE cheap everywhere]
    otherwise -> neutral
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DE_COL = "price_mean"
_NEIGHBOURS = {
    "FR": "price_fr_eur_mwh_mean",
    "NL": "price_nl_eur_mwh_mean",
    "PL": "price_pl_eur_mwh_mean",
    "DK1": "price_dk_1_eur_mwh_mean",
}


class SpreadConsensusStrategy(BacktestStrategy):
    """Trade based on consensus of multiple cross-border spreads.

    When DE is expensive vs most neighbours, expect convergence down.
    When DE is cheap vs most neighbours, expect reversion up.
    """

    def __init__(self) -> None:
        self._median_spreads: dict[str, float] | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._median_spreads = {}
        for name, col in _NEIGHBOURS.items():
            if _DE_COL in train_data.columns and col in train_data.columns:
                spread = train_data[_DE_COL] - train_data[col]
                self._median_spreads[name] = float(spread.median())

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_spreads is None:
            raise RuntimeError("SpreadConsensusStrategy.forecast() called before fit()")

        de_price = float(state.features.get(_DE_COL, 0.0))

        expensive_votes = 0
        cheap_votes = 0
        total = 0

        for name, col in _NEIGHBOURS.items():
            if name not in self._median_spreads:
                continue
            neighbour_price = float(state.features.get(col, 0.0))
            spread = de_price - neighbour_price
            median = self._median_spreads[name]
            total += 1
            if spread > median:
                expensive_votes += 1
            else:
                cheap_votes += 1

        threshold = max(total - 1, 2)  # need strong consensus

        if expensive_votes >= threshold:
            direction = -1  # DE expensive -> convergence down
        elif cheap_votes >= threshold:
            direction = 1  # DE cheap -> reversion up
        else:
            return state.last_settlement_price  # no consensus

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
