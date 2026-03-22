"""Denmark spread convergence strategy.

Uses the DE-DK1 price spread to forecast DE price movements.  When
Germany is much more expensive than Denmark, cross-border flows create
downward pressure on DE prices (convergence).  When Germany is cheap
relative to DK1, expect DE prices to rise.

The spread is defined as: DE_price - DK1_price (positive = DE is
expensive).  The training median spread is used as the neutral
threshold.

Signal:
    spread > median(training) -> short (-1)  [DE too expensive -> falls]
    spread <= median(training) -> long (+1)  [DE relatively cheap -> rises]
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DE_COL = "price_mean"
_DK_COL = "price_dk_1_eur_mwh_mean"


class DenmarkSpreadStrategy(BacktestStrategy):
    """Trade based on the DE-DK1 price spread.

    When DE is expensive relative to DK1 (spread > median), expect
    convergence (DE falls).  When DE is cheap, expect it to rise.
    """

    def __init__(self) -> None:
        self._median_spread: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _DE_COL in train_data.columns and _DK_COL in train_data.columns:
            spread = train_data[_DE_COL] - train_data[_DK_COL]
            self._median_spread = float(spread.median())
        else:
            self._median_spread = 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_spread is None:
            raise RuntimeError("DenmarkSpreadStrategy.forecast() called before fit()")

        de_price = float(state.features.get(_DE_COL, 0.0))
        dk_price = float(state.features.get(_DK_COL, 0.0))
        spread = de_price - dk_price

        # DE expensive vs DK1 -> convergence -> DE falls -> short
        direction = -1 if spread > self._median_spread else 1
        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
