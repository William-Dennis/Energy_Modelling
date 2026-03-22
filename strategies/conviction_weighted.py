"""Conviction-weighted strategy.

Uses the magnitude of the price z-score as a confidence measure.
When the z-score is extreme (high absolute value), the strategy has
high conviction and takes a full-size position.  When the z-score is
near zero, the strategy has low conviction and reduces position size.

The direction is always mean-reversion: high z-score -> short,
low z-score -> long.  The magnitude of the forecast offset scales
with conviction.

Signal:
    zscore > 0: short with magnitude proportional to |zscore|
    zscore <= 0: long with magnitude proportional to |zscore|
    |zscore| < 0.5: neutral (low conviction)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_ZSCORE_COL = "price_zscore_20d"
_MIN_CONVICTION = 0.5  # minimum |zscore| to take a position


class ConvictionWeightedStrategy(BacktestStrategy):
    """Mean-reversion with conviction scaling based on z-score magnitude.

    Larger z-score deviations -> larger position size.
    Near-zero z-score -> neutral (no conviction).
    """

    def __init__(self) -> None:
        self._mean_abs_change: float = 1.0
        self._fitted: bool = False

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.3

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("ConvictionWeightedStrategy.forecast() called before fit()")

        zscore = float(state.features.get(_ZSCORE_COL, 0.0))

        if abs(zscore) < _MIN_CONVICTION:
            return state.last_settlement_price  # low conviction -> neutral

        # Direction: mean-revert against z-score
        direction = -1 if zscore > 0 else 1

        # Scale magnitude: cap at 2x mean_abs_change for extreme z-scores
        conviction = min(abs(zscore), 2.0)
        offset = conviction * self._mean_abs_change

        return state.last_settlement_price + direction * offset

    def reset(self) -> None:
        pass
