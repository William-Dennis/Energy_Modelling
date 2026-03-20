"""Logistic regression direction strategy.

Instead of predicting the price level, this strategy directly predicts the
*direction* of the next price move (+1 / -1) using logistic regression with
L2 regularisation.  The inverse regularisation strength ``C`` is chosen by
5-fold TimeSeriesSplit CV.

Signal: predicted class → trade directly (no skip_buffer dead-zone needed
because the classifier either predicts up or down).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
_N_SPLITS = 5


class LogisticDirectionStrategy(_MLStrategyBase):
    """Logistic regression classifier predicting price-move direction.

    C (inverse regularisation) chosen by time-series CV accuracy.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_direction(train_data)
        best_C, best_acc = _C_GRID[-1], -1.0
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
        for C in _C_GRID:
            pipe = Pipeline(
                [
                    ("s", StandardScaler()),
                    ("m", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
                ]
            )
            accs = []
            for ti, vi in tscv.split(X):
                pipe.fit(X[ti], y[ti])
                accs.append(float(np.mean(pipe.predict(X[vi]) == y[vi])))
            if (a := float(np.mean(accs))) > best_acc:
                best_acc, best_C = a, C
        self._pipeline = Pipeline(
            [
                ("s", StandardScaler()),
                ("m", LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs")),
            ]
        )
        self._pipeline.fit(X, y)
        self.skip_buffer = 0.0  # classifier decides directly

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price
        x = self._get_x_row(state, self._feature_cols)
        direction = float(self._pipeline.predict(x)[0])
        # Return a price that encodes direction via sign of predicted change
        return state.last_settlement_price + direction

    def reset(self) -> None:
        pass
