"""Gradient Boosting on derived features only (GBM Net Demand).

Uses ``HistGradientBoostingClassifier`` restricted to the 18 Phase-A derived
features.  HistGBM is significantly faster than legacy GradientBoostingClassifier.

Fixed hyperparameters: 100 iterations, lr=0.1, max_depth=3.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import DERIVED_FEATURE_COLS, _MLStrategyBase

_MAX_ITER = 100
_LEARNING_RATE = 0.1
_MAX_DEPTH = 3


class GBMNetDemandStrategy(_MLStrategyBase):
    """HistGradientBoosting classifier restricted to Phase-A derived features.

    Fixed params: 100 iterations, lr=0.1, max_depth=3.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data, candidate_cols=DERIVED_FEATURE_COLS)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_direction(train_data)
        self._pipeline = Pipeline(
            [
                ("s", StandardScaler()),
                (
                    "m",
                    HistGradientBoostingClassifier(
                        max_iter=_MAX_ITER,
                        learning_rate=_LEARNING_RATE,
                        max_depth=_MAX_DEPTH,
                        random_state=42,
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
