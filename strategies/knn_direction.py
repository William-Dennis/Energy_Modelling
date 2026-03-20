"""K-Nearest Neighbours direction classifier strategy.

Predicts price-move direction using KNN with a fixed number of neighbours
(k=15).  Features are standardised before distance computation.
"""

from __future__ import annotations

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_N_NEIGHBORS = 15


class KNNDirectionStrategy(_MLStrategyBase):
    """K-Nearest Neighbours classifier for price-move direction.

    k fixed at 15 — no cross-validation search.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_direction(train_data)
        self._pipeline = Pipeline(
            [
                ("s", StandardScaler()),
                ("m", KNeighborsClassifier(n_neighbors=_N_NEIGHBORS, algorithm="ball_tree")),
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
