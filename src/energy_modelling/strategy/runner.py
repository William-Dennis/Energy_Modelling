"""Backtesting runner for strategy evaluation.

Wires together a :class:`MarketEnvironment` and a :class:`Strategy`,
iterating over delivery days and collecting settlement results.

The runner is the single point where :class:`Trade` objects are created.
It combines the direction :class:`Signal` returned by the strategy with
the market's fixed entry price (prior day's DA settlement price from
:attr:`DayState.last_settlement_price`) and the default position size of
1 MW.  This enforces two key invariants:

1. **Entry price is always the prior day's DA settlement** -- strategies
   cannot influence what price they trade at.
2. **Quantity is always 1 MW** -- strategies express conviction via
   direction only, not position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import Settlement, Trade
from energy_modelling.strategy.base import Strategy

_DEFAULT_QUANTITY_MW: float = 1.0
_DEFAULT_HOURS: int = 24


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
        the market.  For each day:

        1. Builds the observable :class:`DayState`.
        2. Asks the strategy for a direction :class:`Signal`.
        3. Constructs a :class:`Trade` from the signal, fixing the entry
           price to ``state.last_settlement_price`` and the quantity to
           :data:`_DEFAULT_QUANTITY_MW` (1 MW).
        4. Settles the trade against the realised day-ahead prices.

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
            signal = self._strategy.act(state)

            if signal is None:
                continue

            # The runner is the sole constructor of Trade objects.
            # Entry price is fixed by the market; quantity is fixed at 1 MW.
            trade = Trade(
                delivery_date=signal.delivery_date,
                entry_price=state.last_settlement_price,
                position_mw=float(signal.direction) * _DEFAULT_QUANTITY_MW,
                hours=_DEFAULT_HOURS,
            )

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
