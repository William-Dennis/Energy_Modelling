"""Decision Tree direction classifier strategy.

Shallow decision tree with fixed max_depth=4 to predict price-move direction.
Simple, interpretable baseline for tree-based methods.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_MAX_DEPTH = 4


class DecisionTreeStrategy(_MLStrategyBase):
    """Decision tree classifier; max_depth fixed at 4."""

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
                ("m", DecisionTreeClassifier(max_depth=_MAX_DEPTH, random_state=42)),
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
