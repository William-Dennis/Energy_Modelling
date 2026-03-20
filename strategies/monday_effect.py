"""Public holiday / Monday effect strategy.

Mondays following a weekend (and days after public holidays) often see
elevated energy prices due to restart of industrial load.  This strategy goes
long on Mondays and short on Fridays (pre-weekend load reduction).

Uses the ``dow_int`` derived feature (0=Mon, 6=Sun) if available, otherwise
reads ``delivery_date.weekday()``.
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

# Mean price changes by day-of-week, learned from training
_DEFAULT = 0.0


class MondayEffectStrategy(BacktestStrategy):
    """Long Monday, short Friday; neutral otherwise.

    The per-DOW mean price changes are learned from training data to set
    ``skip_buffer``.  The act() override returns the DOW-specific signal.
    """

    def __init__(self) -> None:
        self._dow_mean: dict[int, float] = {}

    def fit(self, train_data: pd.DataFrame) -> None:
        dates = pd.to_datetime(train_data["delivery_date"])
        dow = dates.dt.dayofweek  # 0=Mon … 6=Sun
        changes = train_data["price_change_eur_mwh"]
        self._dow_mean = changes.groupby(dow).mean().to_dict()
        # skip_buffer: half the median absolute DOW mean
        import numpy as np

        abs_means = [abs(v) for v in self._dow_mean.values()]
        self.skip_buffer = float(np.median(abs_means)) * 0.5 if abs_means else 0.0

    def forecast(self, state: BacktestState) -> float:
        dow = int(state.features.get("dow_int", state.delivery_date.weekday()))
        mean_change = self._dow_mean.get(dow, _DEFAULT)
        return state.last_settlement_price + mean_change

    def reset(self) -> None:
        pass
