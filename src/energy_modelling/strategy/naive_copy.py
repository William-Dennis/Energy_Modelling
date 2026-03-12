"""Naive Copy strategy -- always long at the prior day's settlement price.

Description
-----------
Goes long 1 MW every day, betting that today's DA settlement will be at
least as high as yesterday's.  Serves as the random-walk baseline: the
best forecast of tomorrow's price is today's price.

Decision rule
-------------
* Every day → signal **long** (+1), unconditionally.

PnL formula
-----------
PnL = (today's DA settlement − yesterday's DA settlement) × 1 MW × 24 h
"""

from __future__ import annotations

from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import DayState, Signal
from energy_modelling.strategy.base import Strategy


class NaiveCopyStrategy(Strategy):
    """Always signal long (+1) every trading day.

    Tests the null hypothesis that day-ahead prices follow a random walk:
    the expected move from yesterday's settlement to today's is zero, so
    there is no systematic edge in either direction.

    Parameters
    ----------
    market:
        Accepted for interface uniformity with strategies that require
        market data at construction time; not used.  Optional -- may be
        omitted in unit tests.
    """

    def __init__(self, market: MarketEnvironment | None = None) -> None:  # noqa: ARG002
        pass

    def act(self, state: DayState) -> Signal | None:
        """Signal long for every delivery day without exception.

        Parameters
        ----------
        state:
            Observable market state for the delivery day.

        Returns
        -------
        Signal
            Direction ``+1`` (long) every day.
        """
        return Signal(
            delivery_date=state.delivery_date,
            direction=1,
        )
