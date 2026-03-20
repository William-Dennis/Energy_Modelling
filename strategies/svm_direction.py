"""Support Vector Machine direction classifier strategy.

Uses a linear SVM (``LinearSVC``) to predict price-move direction.  The
regularisation parameter ``C`` is chosen by 5-fold TimeSeriesSplit CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
_N_SPLITS = 5


class SVMDirectionStrategy(_MLStrategyBase):
    """Linear SVM classifier for price-move direction.

    C chosen from {0.001, 0.01, 0.1, 1, 10} by time-series CV accuracy.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_direction(train_data)
        best_C, best_acc = _C_GRID[0], -1.0
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
        for C in _C_GRID:
            pipe = Pipeline([("s", StandardScaler()), ("m", LinearSVC(C=C, max_iter=2000))])
            accs = []
            for ti, vi in tscv.split(X):
                pipe.fit(X[ti], y[ti])
                accs.append(float(np.mean(pipe.predict(X[vi]) == y[vi])))
            if (a := float(np.mean(accs))) > best_acc:
                best_acc, best_C = a, C
        self._pipeline = Pipeline(
            [("s", StandardScaler()), ("m", LinearSVC(C=best_C, max_iter=2000))]
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
