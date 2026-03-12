"""Core data structures for the market simulation.

Defines the immutable value types exchanged between the market environment,
strategies, and the backtesting runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Signal:
    """Trading signal produced by a strategy.

    Strategies emit a direction signal only -- they have no control over
    entry price or position size.  The :class:`BacktestRunner` constructs
    a :class:`Trade` from this signal by combining it with the market's
    fixed entry price (prior day's DA settlement) and the default
    quantity of 1 MW.

    Parameters
    ----------
    delivery_date:
        The calendar date of the delivery day the signal applies to.
    direction:
        ``+1`` for a long position, ``-1`` for a short position.

    Raises
    ------
    ValueError
        If *direction* is not ``+1`` or ``-1``.
    """

    delivery_date: date
    direction: int  # +1 = long, -1 = short

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            msg = f"direction must be +1 or -1, got {self.direction!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class DayState:
    """Observable market state for a given delivery day.

    All information is available *before* the day-ahead auction closes
    (i.e. data up to and including D-1 23:00 UTC).

    Parameters
    ----------
    delivery_date:
        The calendar date of the delivery day.
    last_settlement_price:
        The settlement price (average hourly DA price) of the most
        recently completed delivery day.
    features:
        A single-row DataFrame of daily-aggregated features available
        at decision time (lagged to prevent look-ahead).
    neighbor_prices:
        Most recent day-ahead prices from neighbouring bidding zones.
    """

    delivery_date: date
    last_settlement_price: float
    features: pd.DataFrame
    neighbor_prices: dict[str, float]


@dataclass(frozen=True)
class Trade:
    """A single trade on the German Base Day Power Future.

    Parameters
    ----------
    delivery_date:
        Calendar date of the delivery day the contract settles against.
    entry_price:
        The futures price locked in at trade time (F_0).
    position_mw:
        Signed position size in MW.  Positive = long, negative = short.
    hours:
        Number of delivery hours.  24 for base-load, 12 for peak.
    """

    delivery_date: date
    entry_price: float
    position_mw: float = 1.0
    hours: int = 24


@dataclass(frozen=True)
class Settlement:
    """Settlement result for a completed trade.

    Parameters
    ----------
    trade:
        The original trade that was entered.
    settlement_price:
        The realised day-ahead settlement price (average of 24 hourly
        prices on the delivery day).
    pnl:
        Profit-and-loss in EUR: ``(P_DA - F_0) * Q * H``.
    """

    trade: Trade
    settlement_price: float
    pnl: float
