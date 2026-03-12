"""Abstract base class for trading strategies.

All strategies must implement the :meth:`act` method which maps
observable market state to a trading decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from energy_modelling.market_simulation.types import DayState, Trade


class Strategy(ABC):
    """Abstract interface for a day-ahead power futures strategy.

    Subclasses implement :meth:`act` to map market observations to
    trading decisions.  The framework calls :meth:`reset` before
    each backtest run.
    """

    @abstractmethod
    def act(self, state: DayState) -> Trade | None:
        """Decide on a trade for the given delivery day.

        Parameters
        ----------
        state:
            Observable market state for the delivery day.

        Returns
        -------
        Trade | None
            A trade to execute, or *None* to skip this day.
        """

    def reset(self) -> None:  # noqa: B027
        """Reset any internal state between backtest runs."""
