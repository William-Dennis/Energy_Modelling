"""Core types for the student hackathon challenge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class BacktestState:
    """Observable state for one challenge decision.

    Parameters
    ----------
    delivery_date:
        The delivery date the strategy is deciding for.
    last_settlement_price:
        The most recent realised daily settlement available at decision time.
    features:
        Feature row for the current decision date. This excludes unknown target
        columns such as the current day's settlement price.
    history:
        All previously observed daily rows strictly before ``delivery_date``.
        Historical labels are allowed here because they are already realised.
    """

    delivery_date: date
    last_settlement_price: float
    features: pd.Series
    history: pd.DataFrame


class BacktestStrategy(ABC):
    """Minimal interface students implement for the hackathon challenge."""

    def fit(self, train_data: pd.DataFrame) -> None:  # noqa: B027
        """Fit the strategy on public training data."""

    @abstractmethod
    def act(self, state: BacktestState) -> int | None:
        """Return ``+1`` for long, ``-1`` for short, or ``None`` to skip."""

    def reset(self) -> None:  # noqa: B027
        """Reset any internal state before a new evaluation run."""
