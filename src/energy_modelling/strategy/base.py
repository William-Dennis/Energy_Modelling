"""Abstract base class for trading strategies.

All strategies must implement the :meth:`act` method which maps
observable market state to a direction signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from energy_modelling.market_simulation.types import DayState, Signal


class Strategy(ABC):
    """Abstract interface for a day-ahead power futures strategy.

    Subclasses implement :meth:`act` to map market observations to a
    :class:`~energy_modelling.market_simulation.types.Signal` indicating
    trade direction.  The :class:`~energy_modelling.strategy.runner.BacktestRunner`
    constructs the actual :class:`~energy_modelling.market_simulation.types.Trade`
    using the signal direction, the market's fixed entry price (prior day's DA
    settlement), and the default 1 MW quantity.

    This separation means strategies can never influence entry price or
    position size -- they only decide *which way* to trade each day.

    The framework calls :meth:`reset` before each backtest run.
    """

    @abstractmethod
    def act(self, state: DayState) -> Signal | None:
        """Decide on a trade direction for the given delivery day.

        Parameters
        ----------
        state:
            Observable market state for the delivery day.

        Returns
        -------
        Signal | None
            A direction signal to execute, or *None* to skip this day.
        """

    def reset(self) -> None:  # noqa: B027
        """Reset any internal state between backtest runs."""
