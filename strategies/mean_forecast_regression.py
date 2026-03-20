"""Mean forecast ensemble of regression strategies.

Averages the price forecasts from 4 regression strategies.  The mean
forecast is used directly; skip_buffer = median of member skip_buffers.
"""

from __future__ import annotations

import numpy as np

from energy_modelling.backtest.types import BacktestState
from strategies.ensemble_base import _EnsembleBase
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.ridge_regression import RidgeRegressionStrategy
from strategies.elastic_net import ElasticNetStrategy
from strategies.bayesian_ridge import BayesianRidgeStrategy


class MeanForecastRegressionStrategy(_EnsembleBase):
    """Mean of Lasso, Ridge, ElasticNet, BayesianRidge price forecasts."""

    _MEMBERS = [
        LassoRegressionStrategy,
        RidgeRegressionStrategy,
        ElasticNetStrategy,
        BayesianRidgeStrategy,
    ]

    def forecast(self, state: BacktestState) -> float:
        forecasts = self._get_member_forecasts(state)
        return float(np.mean(forecasts))
