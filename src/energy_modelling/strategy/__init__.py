"""Strategy testing framework for day-ahead power futures.

Provides an abstract :class:`Strategy` base class, a backtesting runner,
performance analysis utilities, and a baseline naive-copy strategy.
"""

from energy_modelling.strategy.analysis import compute_metrics
from energy_modelling.strategy.base import Strategy
from energy_modelling.strategy.naive_copy import NaiveCopyStrategy
from energy_modelling.strategy.runner import BacktestResult, BacktestRunner

__all__ = [
    "BacktestResult",
    "BacktestRunner",
    "NaiveCopyStrategy",
    "Strategy",
    "compute_metrics",
]
