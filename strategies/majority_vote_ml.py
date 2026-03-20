"""Majority vote ensemble of ML classifiers.

Combines 5 ML direction classifiers.  Votes by majority.
"""

from __future__ import annotations

from energy_modelling.backtest.types import BacktestState
from strategies.decision_tree_direction import DecisionTreeStrategy
from strategies.ensemble_base import _EnsembleBase
from strategies.gradient_boosting_direction import GradientBoostingStrategy
from strategies.knn_direction import KNNDirectionStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.random_forest_direction import RandomForestStrategy


class MajorityVoteMLStrategy(_EnsembleBase):
    """Majority vote over 5 ML classifiers.

    Members: Logistic, RandomForest, GradientBoosting, KNN, DecisionTree.
    """

    _MEMBERS = [
        LogisticDirectionStrategy,
        RandomForestStrategy,
        GradientBoostingStrategy,
        KNNDirectionStrategy,
        DecisionTreeStrategy,
    ]

    def forecast(self, state: BacktestState) -> float:
        dirs = self._get_member_directions(state)
        vote = sum(dirs)
        if vote > 0:
            return state.last_settlement_price + 1.0
        if vote < 0:
            return state.last_settlement_price - 1.0
        return state.last_settlement_price
