"""Baseline strategy: naive copy of the most recent settlement price.

Goes long 1 MW every day using yesterday's average DA price as the
entry price.  This is equivalent to the "random walk" hypothesis --
the best forecast is the last observation.

Daily PnL = (today's settlement - yesterday's settlement) * 1 MW * 24 h
"""

from __future__ import annotations

from energy_modelling.market_simulation.types import DayState, Trade
from energy_modelling.strategy.base import Strategy


class NaiveCopyStrategy(Strategy):
    """Always go long 1 MW at the last known settlement price.

    The entry price equals the most recent day's average DA price.
    This tests the null hypothesis that day-ahead prices follow a
    random walk.
    """

    def act(self, state: DayState) -> Trade | None:
        """Enter a long 1 MW position at the last settlement price.

        Parameters
        ----------
        state:
            Observable market state for the delivery day.

        Returns
        -------
        Trade
            A long 1 MW trade with entry price equal to
            ``state.last_settlement_price``.
        """
        return Trade(
            delivery_date=state.delivery_date,
            entry_price=state.last_settlement_price,
            position_mw=1.0,
            hours=24,
        )
