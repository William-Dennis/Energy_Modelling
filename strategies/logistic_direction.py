"""Logistic regression direction strategy.

Instead of predicting the price level, this strategy directly predicts the
*direction* of the next price move (+1 / -1) using logistic regression with
L2 regularisation.  C is fixed at 1.0.

Signal: predicted class → trade directly (no skip_buffer dead-zone needed
because the classifier either predicts up or down).
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_C = 1.0


class LogisticDirectionStrategy(_MLStrategyBase):
    """Logistic regression classifier predicting price-move direction.

    C (inverse regularisation) fixed at 1.0 — no cross-validation search.
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
                ("m", LogisticRegression(C=_C, max_iter=1000, solver="lbfgs")),
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
