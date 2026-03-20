"""K-Nearest Neighbours direction classifier strategy.

Predicts price-move direction using KNN.  The number of neighbours ``k`` is
chosen from {3, 5, 7, 11, 15} by 5-fold TimeSeriesSplit CV (accuracy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_K_GRID = [3, 5, 7, 11, 15]
_N_SPLITS = 5


class KNNDirectionStrategy(_MLStrategyBase):
    """K-Nearest Neighbours classifier for price-move direction.

    k chosen from {3,5,7,11,15} by time-series CV accuracy.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_direction(train_data)
        best_k, best_acc = _K_GRID[0], -1.0
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
        for k in _K_GRID:
            pipe = Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=k))])
            accs = [
                float(np.mean(pipe.fit(X[ti], y[ti]).predict(X[vi]) == y[vi]))
                for ti, vi in tscv.split(X)
            ]
            if (a := float(np.mean(accs))) > best_acc:
                best_acc, best_k = a, k
        self._pipeline = Pipeline(
            [("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=best_k))]
        )
        self._pipeline.fit(X, y)
        self.skip_buffer = 0.0

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price
        x = self._get_x_row(state, self._feature_cols)
        direction = float(self._pipeline.predict(x)[0])
        return state.last_settlement_price + direction

    def reset(self) -> None:
        pass
