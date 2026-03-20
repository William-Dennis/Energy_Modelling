"""Ridge regression strategy (L2 regularisation).

Same pattern as LassoRegressionStrategy but using Ridge (L2) which retains
all features at non-zero weights. Ridge is more stable when features are
correlated, which is common in energy data.

CV: 5-fold TimeSeriesSplit over log-spaced alpha grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
_N_SPLITS = 5


class RidgeRegressionStrategy(_MLStrategyBase):
    """L2-regularised linear regression, alpha chosen by time-series CV."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_train(train_data)
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5
        best_alpha, best_mse = _ALPHA_GRID[0], float("inf")
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
        for alpha in _ALPHA_GRID:
            pipe = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=alpha))])
            mses = [
                float(np.mean((pipe.fit(X[ti], y[ti]).predict(X[vi]) - y[vi]) ** 2))
                for ti, vi in tscv.split(X)
            ]
            if (m := float(np.mean(mses))) < best_mse:
                best_mse, best_alpha = m, alpha
        self._pipeline = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=best_alpha))])
        self._pipeline.fit(X, y)

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price
        x = self._get_x_row(state, self._feature_cols)
        return state.last_settlement_price + float(self._pipeline.predict(x)[0])

    def reset(self) -> None:
        pass
