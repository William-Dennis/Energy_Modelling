"""Median forecast ensemble.

Takes the median (rather than mean) of 5 regression forecasts to reduce
sensitivity to outlier predictions.
"""

from __future__ import annotations

import numpy as np

from energy_modelling.backtest.types import BacktestState
from strategies.ensemble_base import _EnsembleBase
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.ridge_regression import RidgeRegressionStrategy
from strategies.elastic_net import ElasticNetStrategy
from strategies.bayesian_ridge import BayesianRidgeStrategy
from strategies.pls_regression import PLSRegressionStrategy


class MedianForecastEnsembleStrategy(_EnsembleBase):
    """Median of Lasso, Ridge, ElasticNet, BayesianRidge, PLS forecasts."""

    _MEMBERS = [
        LassoRegressionStrategy,
        RidgeRegressionStrategy,
        ElasticNetStrategy,
        BayesianRidgeStrategy,
        PLSRegressionStrategy,
    ]

    def forecast(self, state: BacktestState) -> float:
        forecasts = self._get_member_forecasts(state)
        return float(np.median(forecasts))
