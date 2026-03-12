"""Strategy testing framework for day-ahead power futures.

Provides an abstract :class:`Strategy` base class, a backtesting runner,
performance analysis utilities, and baseline strategies.
"""

from energy_modelling.market_simulation.types import Signal
from energy_modelling.strategy.analysis import compute_metrics
from energy_modelling.strategy.base import Strategy
from energy_modelling.strategy.naive_copy import NaiveCopyStrategy
from energy_modelling.strategy.perfect_foresight import PerfectForesightStrategy
from energy_modelling.strategy.runner import BacktestResult, BacktestRunner

__all__ = [
    "BacktestResult",
    "BacktestRunner",
    "NaiveCopyStrategy",
    "PerfectForesightStrategy",
    "Signal",
    "Strategy",
    "compute_metrics",
]
