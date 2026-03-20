"""Cross-border price spread strategy using French and Dutch electricity prices.

Hypothesis: When neighbouring markets (FR, NL) were recently more expensive
than DE-LU, European market coupling will pull DE prices upward (convergence).
When neighbours were cheaper, DE prices face downward pressure.

The spread is defined as: avg(FR_price, NL_price) - DE_price (yesterday's
lagged values are already available at decision time).

Signal:
    spread >= median(training) → long  (+1)   [neighbours expensive → DE rises]
    spread <  median(training) → short (-1)   [neighbours cheap → DE falls]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_PRICE_COL = "price_mean"
_FR_COL = "price_fr_eur_mwh_mean"
_NL_COL = "price_nl_eur_mwh_mean"


class CrossBorderSpreadStrategy(BacktestStrategy):
    """Trade based on the DE vs FR/NL price spread.

    Positive spread (neighbours > DE) signals upward pressure on DE prices.
    Threshold is the median spread from training data.
    """

    def __init__(self) -> None:
        self._median_spread: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        neighbour_avg = (train_data[_FR_COL] + train_data[_NL_COL]) / 2.0
        spread = neighbour_avg - train_data[_PRICE_COL]
        self._median_spread = float(spread.median())
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._median_spread is None:
            raise RuntimeError("CrossBorderSpreadStrategy.forecast() called before fit()")
        fr = float(state.features[_FR_COL])
        nl = float(state.features[_NL_COL])
        de = float(state.features[_PRICE_COL])
        spread = (fr + nl) / 2.0 - de
        direction = 1 if spread >= self._median_spread else -1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
