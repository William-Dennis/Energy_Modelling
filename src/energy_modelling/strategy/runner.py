"""Backtesting runner for strategy evaluation.

Wires together a :class:`MarketEnvironment` and a :class:`Strategy`,
iterating over delivery days and collecting settlement results.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import Settlement
from energy_modelling.strategy.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated results from a backtest run.

    Parameters
    ----------
    settlements:
        List of all settlement results in chronological order.
    daily_pnl:
        Series of daily PnL values indexed by delivery date.
    cumulative_pnl:
        Cumulative sum of daily PnL.
    """

    settlements: list[Settlement]
    daily_pnl: pd.Series
    cumulative_pnl: pd.Series


class BacktestRunner:
    """Runs a strategy against a market environment and collects results.

    Parameters
    ----------
    market:
        The simulated market environment.
    strategy:
        The trading strategy to evaluate.
    """

    def __init__(self, market: MarketEnvironment, strategy: Strategy) -> None:
        self._market = market
        self._strategy = strategy

    def run(self) -> BacktestResult:
        """Execute the backtest over all delivery days.

        Resets the strategy, then iterates over every delivery day in
        the market, calling ``strategy.act()`` and settling each trade.

        Returns
        -------
        BacktestResult
            The aggregated backtest results.
        """
        self._strategy.reset()

        settlements: list[Settlement] = []
        pnl_values: list[float] = []
        pnl_dates: list[object] = []

        for delivery_date in self._market.delivery_dates:
            state = self._market.get_state(delivery_date)
            trade = self._strategy.act(state)

            if trade is None:
                continue

            result = self._market.settle(trade)
            settlements.append(result)
            pnl_values.append(result.pnl)
            pnl_dates.append(delivery_date)

        daily_pnl = pd.Series(pnl_values, index=pnl_dates, name="pnl", dtype=float)
        cumulative_pnl = daily_pnl.cumsum()

        return BacktestResult(
            settlements=settlements,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
        )
