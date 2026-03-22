"""Volatility breakout strategy.

When recent price volatility spikes above its historical norm (using
rolling_vol_7d), a breakout is underway.  The strategy then follows
the recent price direction: if the last move was up, continue long;
if down, continue short.  During low-volatility regimes, stay neutral.

Signal:
    vol_7d > p75(training) AND last_change > 0 -> long  (+1)
    vol_7d > p75(training) AND last_change < 0 -> short (-1)
    vol_7d <= p75(training) -> neutral (0)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_VOL_COL = "rolling_vol_7d"
_CHANGE_COL = "price_change_eur_mwh"


class VolatilityBreakoutStrategy(BacktestStrategy):
    """Trade momentum breakouts triggered by high volatility.

    High volatility + positive change -> long (breakout up).
    High volatility + negative change -> short (breakout down).
    Normal volatility -> neutral (no signal).
    """

    def __init__(self) -> None:
        self._p75_vol: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _VOL_COL in train_data.columns:
            self._p75_vol = float(train_data[_VOL_COL].quantile(0.75))
        else:
            self._p75_vol = float("inf")  # never triggers without data

        if _CHANGE_COL in train_data.columns:
            mac = float(train_data[_CHANGE_COL].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data[_CHANGE_COL].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._p75_vol is None:
            raise RuntimeError("VolatilityBreakoutStrategy.forecast() called before fit()")

        vol = float(state.features.get(_VOL_COL, 0.0))
        change = float(state.features.get(_CHANGE_COL, 0.0))

        if vol > self._p75_vol:
            direction = 1 if change > 0 else -1
            return state.last_settlement_price + direction * self._mean_abs_change

        return state.last_settlement_price  # neutral

    def reset(self) -> None:
        pass
