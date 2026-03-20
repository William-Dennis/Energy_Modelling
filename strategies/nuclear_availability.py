"""Nuclear availability strategy — event-driven supply shortfall signal.

Hypothesis: Nuclear provides baseload supply. Sudden drops in nuclear
generation (outages, maintenance) tighten supply and push prices up.
This is visible as yesterday's nuclear output falling more than 1 standard
deviation below the 14-day rolling mean fitted from training data.

Signal:
    nuclear < mean - 1*std  → long  (+1)  [supply shortfall]
    nuclear > mean + 1*std  → short (-1)  [supply surplus]
    otherwise               → skip  (None) [normal range]

Note: German nuclear was structurally zero from April 2023 following the
final shutdown. This strategy will produce mostly skips on post-2023 data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_NUCLEAR_COL = "gen_nuclear_mw_mean"


class NuclearAvailabilityStrategy(BacktestStrategy):
    """Event-driven strategy: long on nuclear outage, short on surplus.

    Thresholds are the mean ± 1 standard deviation of nuclear generation
    from the training data.  When std == 0 (structurally constant), the
    strategy always skips to avoid spurious signals.
    """

    def __init__(self) -> None:
        self._mean: float | None = None
        self._std: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        vals = train_data[_NUCLEAR_COL]
        self._mean = float(vals.mean())
        self._std = float(vals.std(ddof=0))
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0

    def forecast(self, state: BacktestState) -> float:
        if self._mean is None or self._std is None:
            raise RuntimeError("NuclearAvailabilityStrategy.forecast() called before fit()")
        nuclear = float(state.features[_NUCLEAR_COL])
        if self._std == 0.0:
            # No variation → no signal; act() will return None via skip_buffer trick
            return state.last_settlement_price
        if nuclear < self._mean - self._std:
            direction = 1
        elif nuclear > self._mean + self._std:
            direction = -1
        else:
            return state.last_settlement_price  # triggers None in act()
        return state.last_settlement_price + direction * self._mean_abs_change

    def act(self, state: BacktestState) -> int | None:
        """Override act() to return None cleanly in the neutral zone."""
        if self._mean is None or self._std is None:
            raise RuntimeError("NuclearAvailabilityStrategy.act() called before fit()")
        nuclear = float(state.features[_NUCLEAR_COL])
        if self._std == 0.0:
            return None
        if nuclear < self._mean - self._std:
            return 1
        if nuclear > self._mean + self._std:
            return -1
        return None

    def reset(self) -> None:
        pass
