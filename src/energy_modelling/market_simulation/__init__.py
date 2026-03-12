"""Market simulation sub-package for German Base Day Power Futures.

Provides a market environment that iterates over delivery days, enforces
information cutoffs, and computes settlement prices and PnL for trades
on the DE-LU day-ahead auction index.
"""

from energy_modelling.market_simulation.contract import compute_pnl, compute_settlement_price
from energy_modelling.market_simulation.data import (
    build_daily_features,
    compute_daily_settlement,
    load_dataset,
)
from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import DayState, Settlement, Signal, Trade

__all__ = [
    "DayState",
    "MarketEnvironment",
    "Settlement",
    "Signal",
    "Trade",
    "build_daily_features",
    "compute_daily_settlement",
    "compute_pnl",
    "compute_settlement_price",
    "load_dataset",
]
