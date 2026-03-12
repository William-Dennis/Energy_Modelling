"""Perfect-foresight strategy: theoretical upper-bound on PnL.

Description
-----------
Uses advance knowledge of the true settlement price for each delivery day
to always signal the maximum-profit direction.  Not realisable in practice
-- it requires information only available after the day-ahead auction
clears.  Its sole purpose is to provide a **theoretical upper bound** on
what any causal strategy could earn in the same market environment.

Decision rule
-------------
For each delivery day D with true settlement price P_DA and prior
settlement price P_{D-1} (the fixed entry price set by the runner):

* If P_DA > P_{D-1}  → signal **long**  (+1)  -- profit is maximised
* If P_DA < P_{D-1}  → signal **short** (-1)  -- profit is maximised
* If P_DA == P_{D-1} → skip (return None)      -- zero PnL either way

PnL formula
-----------
PnL = |today's DA settlement − yesterday's DA settlement| × 1 MW × 24 h
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from energy_modelling.market_simulation.types import DayState, Signal
from energy_modelling.strategy.base import Strategy


class PerfectForesightStrategy(Strategy):
    """Always signal the maximum-profit direction using future price knowledge.

    Parameters
    ----------
    settlement_prices:
        A mapping from delivery date → true settlement price (EUR/MWh).
        This is the daily mean of the 24 hourly DA prices.  It must cover
        every delivery date the strategy will be asked to trade.

    Notes
    -----
    The ``settlement_prices`` mapping should be built from
    :func:`~energy_modelling.market_simulation.data.compute_daily_settlement`
    *after* the backtest period is known, and passed in at construction time.
    This keeps the ``act()`` interface identical to all other strategies while
    making the look-ahead explicit and auditable at the call site.
    """

    def __init__(self, settlement_prices: dict[date, float] | pd.Series) -> None:
        if isinstance(settlement_prices, pd.Series):
            self._settlements: dict[date, float] = settlement_prices.to_dict()
        else:
            self._settlements = dict(settlement_prices)

    def act(self, state: DayState) -> Signal | None:
        """Signal the maximum-profit direction using the true settlement price.

        Parameters
        ----------
        state:
            Observable market state for the delivery day.

        Returns
        -------
        Signal | None
            Direction ``+1`` if P_DA > entry price, ``-1`` if P_DA < entry
            price, or *None* if P_DA == entry price (no edge).

        Raises
        ------
        KeyError
            If the delivery date is not present in ``settlement_prices``.
        """
        true_settlement = self._settlements[state.delivery_date]
        # Compare against the prior day's settlement -- the entry price the
        # runner will use.  This is the same value exposed via the state.
        entry_price = state.last_settlement_price

        if true_settlement > entry_price:
            direction = 1
        elif true_settlement < entry_price:
            direction = -1
        else:
            # Exactly flat -- no edge, skip the day.
            return None

        return Signal(
            delivery_date=state.delivery_date,
            direction=direction,
        )
