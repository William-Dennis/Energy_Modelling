"""Diversity ensemble: maximize signal diversity.

Selects one rule-based, one ML regression, and one ML classifier to
represent three orthogonal signal sources.  Uses simple majority vote.
"""

from __future__ import annotations

from energy_modelling.backtest.types import BacktestState
from strategies.ensemble_base import _EnsembleBase
from strategies.composite_signal import CompositeSignalStrategy
from strategies.ridge_regression import RidgeRegressionStrategy
from strategies.random_forest_direction import RandomForestStrategy


class DiversityEnsembleStrategy(_EnsembleBase):
    """Diverse 3-member ensemble: CompositeSignal + Ridge + RandomForest."""

    _MEMBERS = [
        CompositeSignalStrategy,
        RidgeRegressionStrategy,
        RandomForestStrategy,
    ]

    def forecast(self, state: BacktestState) -> float:
        dirs = self._get_member_directions(state)
        vote = sum(dirs)
        if vote > 0:
            return state.last_settlement_price + 1.0
        if vote < 0:
            return state.last_settlement_price - 1.0
        return state.last_settlement_price
