"""Random Forest direction classifier strategy.

Uses a Random Forest to predict the direction (+1 / -1) of the next day's
price move.  Fixed hyperparameters: 200 trees, max_depth=6, balanced class
weights.  No CV to keep inference fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_N_ESTIMATORS = 200
_MAX_DEPTH = 6


class RandomForestStrategy(_MLStrategyBase):
    """Random Forest classifier: 200 trees, max_depth=6, balanced classes."""

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
                (
                    "m",
                    RandomForestClassifier(
                        n_estimators=_N_ESTIMATORS,
                        max_depth=_MAX_DEPTH,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
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
