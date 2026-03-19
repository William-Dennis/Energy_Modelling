"""Perfect foresight strategy for convergence analysis (Phase 7).

This strategy is NOT a legitimate competitor -- it cheats by looking at
the real settlement price before making its decision.  It exists solely
as a theoretical tool for analyzing the market engine's convergence
properties.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class PerfectForesightStrategy(BacktestStrategy):
    """Always predicts the correct direction by peeking at the settlement price.

    Constructed with a lookup of delivery_date → settlement_price for the
    evaluation period.  During ``act()``, compares the real settlement to
    ``last_settlement_price`` and returns +1 if price goes up, -1 if down.

    This strategy is for analysis only — it has zero prediction error and
    serves as an upper bound on strategy performance.

    Parameters
    ----------
    settlement_lookup:
        Mapping from delivery date to real settlement price for all
        evaluation dates.  Typically built from ``daily_public.csv``.
    """

    def __init__(self, settlement_lookup: dict[date, float]) -> None:
        self._settlement_lookup = settlement_lookup

    def act(self, state: BacktestState) -> int | None:
        """Return the correct direction by looking up the real settlement."""
        real_price = self._settlement_lookup.get(state.delivery_date)
        if real_price is None:
            return 1  # Default to long if unknown date
        if real_price > state.last_settlement_price:
            return 1
        return -1

    def reset(self) -> None:
        """No-op: nothing to reset."""
