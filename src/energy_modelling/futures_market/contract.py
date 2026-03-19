"""German Base Day Power Future contract calculations.

Implements the settlement price computation and payoff formula for the
EEX-style financially settled day-ahead power future.
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.futures_market.types import Trade


def compute_settlement_price(hourly_prices: pd.Series) -> float:
    """Compute the settlement price as the mean of hourly DA prices.

    Parameters
    ----------
    hourly_prices:
        Series of hourly day-ahead prices (EUR/MWh) for a single
        delivery day.  Must contain at least one non-NaN value.

    Returns
    -------
    float
        The arithmetic mean of the hourly prices.

    Raises
    ------
    ValueError
        If *hourly_prices* is empty or all NaN.
    """
    clean = hourly_prices.dropna()
    if clean.empty:
        msg = "hourly_prices is empty or all NaN"
        raise ValueError(msg)
    return float(clean.mean())


def compute_pnl(trade: Trade, settlement_price: float) -> float:
    """Compute the payoff for a settled trade.

    Uses the standard futures payoff formula::

        PnL = (P_DA - F_0) * Q * H

    Parameters
    ----------
    trade:
        The trade whose PnL is being computed.
    settlement_price:
        The realised day-ahead settlement price (P_DA).

    Returns
    -------
    float
        Profit-and-loss in EUR.
    """
    return (settlement_price - trade.entry_price) * trade.position_mw * trade.hours
