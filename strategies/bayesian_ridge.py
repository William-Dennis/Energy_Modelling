"""Bayesian Ridge regression strategy.

Bayesian Ridge uses evidence-based regularisation — hyperparameters alpha and
lambda are inferred from the data rather than cross-validated.  This makes it
self-regularising and fast to fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase


class BayesianRidgeStrategy(_MLStrategyBase):
    """Bayesian Ridge regression; regularisation inferred from data."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_train(train_data)
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5
        self._pipeline = Pipeline([("s", StandardScaler()), ("m", BayesianRidge())])
        self._pipeline.fit(X, y)

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price
        x = self._get_x_row(state, self._feature_cols)
        return state.last_settlement_price + float(self._pipeline.predict(x)[0])

    def reset(self) -> None:
        pass
