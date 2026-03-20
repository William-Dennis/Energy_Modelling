"""Ensemble/meta strategy base class.

Provides shared infrastructure for strategies that combine signals from
multiple child strategies.  Child strategies are listed by class in
``_MEMBERS``.  The base class handles:

- ``fit()``: instantiating and fitting each member strategy
- ``_get_member_forecasts()``: collecting forecast prices from all members
- ``_get_member_directions()``: converting forecasts → {-1, 0, +1}
- ``forecast()``: abstract — subclass must implement the aggregation logic

Usage
-----
```python
class MajorityVoteStrategy(_EnsembleBase):
    _MEMBERS = [AlwaysLongStrategy, WindForecastStrategy, ...]

    def forecast(self, state):
        dirs = self._get_member_directions(state)
        vote = sum(dirs)
        return state.last_settlement_price + (1 if vote > 0 else -1 if vote < 0 else 0)
```
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class _EnsembleBase(BacktestStrategy):
    """Abstract base for ensemble strategies.

    Subclasses declare ``_MEMBERS`` as a list of strategy *classes* (not
    instances).  ``fit()`` instantiates each class and calls its ``fit()``.
    """

    _MEMBERS: list[type[BacktestStrategy]] = []

    def __init__(self) -> None:
        self._fitted_members: list[BacktestStrategy] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted_members = []
        for cls in self._MEMBERS:
            member = cls()
            member.fit(train_data)
            self._fitted_members.append(member)
        # Default skip_buffer: median of member skip_buffers
        buffers = [m.skip_buffer for m in self._fitted_members]
        self.skip_buffer = float(np.median(buffers)) if buffers else 0.0

    def _get_member_forecasts(self, state: BacktestState) -> list[float]:
        """Return raw forecast prices from all fitted members."""
        return [float(m.forecast(state)) for m in self._fitted_members]

    def _get_member_directions(self, state: BacktestState) -> list[float]:
        """Return {-1, 0, +1} direction signals from all fitted members."""
        dirs = []
        for m in self._fitted_members:
            forecast = float(m.forecast(state))
            diff = forecast - state.last_settlement_price
            if abs(diff) <= m.skip_buffer:
                dirs.append(0.0)
            else:
                dirs.append(1.0 if diff > 0 else -1.0)
        return dirs

    @abstractmethod
    def forecast(self, state: BacktestState) -> float:
        """Aggregate member signals into a single price forecast."""

    def reset(self) -> None:
        for m in self._fitted_members:
            m.reset()
