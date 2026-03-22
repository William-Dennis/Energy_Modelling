"""Pruned ML ensemble strategy — diversity-focused.

Equal-weight ensemble of the 3 most structurally different ML approaches:
Ridge (linear, L2), Lasso (linear, L1), and RandomForest (non-linear, tree).

The current 11 ML regression strategies have >0.99 pairwise forecast
correlation. This ensemble prunes to 3 structurally different models to
reduce redundancy while preserving diverse modelling approaches.

Source: Phase 10g candidate #10 and Phase 10d cluster analysis showing
ML regression cluster dominance.
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState
from strategies.ensemble_base import _EnsembleBase
from strategies.lasso_top_features import LassoTopFeaturesStrategy
from strategies.random_forest_direction import RandomForestStrategy
from strategies.ridge_regression import RidgeRegressionStrategy


class PrunedMLEnsembleStrategy(_EnsembleBase):
    """Equal-weight average of Ridge, Lasso, and RandomForest forecasts.

    Trades on the mean forecast of the 3 most structurally different ML
    models, reducing redundancy while maintaining diverse modelling.
    """

    _MEMBERS = [
        RidgeRegressionStrategy,
        LassoTopFeaturesStrategy,
        RandomForestStrategy,
    ]

    def fit(self, train_data: pd.DataFrame) -> None:
        super().fit(train_data)

    def forecast(self, state: BacktestState) -> float:
        forecasts = self._get_member_forecasts(state)
        if not forecasts:
            return state.last_settlement_price
        return sum(forecasts) / len(forecasts)

    def reset(self) -> None:
        super().reset()
