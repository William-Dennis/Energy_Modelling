"""Nuclear event strategy — binary spike detector.

Detects sharp drops in nuclear generation relative to a rolling 7-day mean
computed from training data. When nuclear output drops by more than 15% from
the rolling mean, forecasts a price increase.

Unlike NuclearAvailabilityStrategy which uses static mean +/- 1 std thresholds,
this strategy uses a percentage-drop model that is more sensitive to relative
changes and avoids triggering on structurally-zero post-shutdown data.

Signal:
    nuclear < rolling_mean * 0.85  -> long  (+1)  [supply shortfall]
    nuclear > rolling_mean * 1.15  -> short (-1)  [supply surplus]
    otherwise                      -> neutral (forecast = entry)

Source: Phase 10g candidate #8. Rare but high-impact supply-side signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_NUCLEAR_COL = "gen_nuclear_mw_mean"
_DROP_THRESHOLD = 0.15  # 15% deviation from rolling mean


class NuclearEventStrategy(BacktestStrategy):
    """Long on nuclear supply shortfall, short on surplus. Uses 15% relative
    deviation from the training-set rolling mean.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._rolling_mean: float = 0.0
        self._mean_abs_change: float = 1.0
        self._min_generation: float = 0.0

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        vals = train_data[_NUCLEAR_COL].values.astype(float)
        clean = vals[np.isfinite(vals)]

        if len(clean) >= 7:
            # Use the last 7-day window as the reference mean
            self._rolling_mean = float(np.mean(clean[-7:]))
        elif len(clean) > 0:
            self._rolling_mean = float(np.mean(clean))
        else:
            self._rolling_mean = 0.0

        # Minimum meaningful generation to avoid triggering on post-shutdown zeros
        self._min_generation = float(np.percentile(clean, 10)) if len(clean) > 0 else 0.0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

        self.skip_buffer = self._mean_abs_change * 0.5

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("NuclearEventStrategy.forecast() called before fit()")

        nuclear = float(state.features[_NUCLEAR_COL])

        # If rolling mean is near zero (post-shutdown), no meaningful signal
        if self._rolling_mean < 100.0:
            return state.last_settlement_price

        lower = self._rolling_mean * (1 - _DROP_THRESHOLD)
        upper = self._rolling_mean * (1 + _DROP_THRESHOLD)

        if nuclear < lower:
            direction = 1  # supply shortfall -> price up
        elif nuclear > upper:
            direction = -1  # supply surplus -> price down
        else:
            return state.last_settlement_price  # no event

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
