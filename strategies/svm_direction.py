"""Support Vector Machine direction classifier strategy.

Uses a linear SVM (``LinearSVC``) to predict price-move direction.
The regularisation parameter ``C`` is fixed at 1.0.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_C = 1.0


class SVMDirectionStrategy(_MLStrategyBase):
    """Linear SVM classifier for price-move direction.

    C fixed at 1.0 — no cross-validation search.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_direction(train_data)
        self._pipeline = Pipeline([("s", StandardScaler()), ("m", LinearSVC(C=_C, max_iter=2000))])
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
