"""Neural Network (MLP) direction classifier strategy.

Uses sklearn's ``MLPClassifier`` with a small hidden layer (64, 32) to
predict price-move direction.  Fixed architecture; trained with Adam,
early stopping on validation fraction.
"""

from __future__ import annotations

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_HIDDEN_LAYER_SIZES = (64, 32)
_MAX_ITER = 500


class NeuralNetStrategy(_MLStrategyBase):
    """MLP classifier (64→32 hidden units) for price-move direction."""

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
                    MLPClassifier(
                        hidden_layer_sizes=_HIDDEN_LAYER_SIZES,
                        activation="relu",
                        solver="adam",
                        max_iter=_MAX_ITER,
                        early_stopping=True,
                        validation_fraction=0.1,
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
