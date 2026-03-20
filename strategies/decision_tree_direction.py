"""Decision Tree direction classifier strategy.

Shallow decision tree (max_depth chosen by CV from {2,3,4,5,6}) to predict
price-move direction.  Simple, interpretable baseline for tree-based methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_DEPTH_GRID = [2, 3, 4, 5, 6]
_N_SPLITS = 5


class DecisionTreeStrategy(_MLStrategyBase):
    """Decision tree classifier; max_depth chosen by time-series CV."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_direction(train_data)
        best_depth, best_acc = _DEPTH_GRID[0], -1.0
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
        for depth in _DEPTH_GRID:
            pipe = Pipeline(
                [
                    ("s", StandardScaler()),
                    ("m", DecisionTreeClassifier(max_depth=depth, random_state=42)),
                ]
            )
            accs = [
                float(np.mean(pipe.fit(X[ti], y[ti]).predict(X[vi]) == y[vi]))
                for ti, vi in tscv.split(X)
            ]
            if (a := float(np.mean(accs))) > best_acc:
                best_acc, best_depth = a, depth
        self._pipeline = Pipeline(
            [
                ("s", StandardScaler()),
                ("m", DecisionTreeClassifier(max_depth=best_depth, random_state=42)),
            ]
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
