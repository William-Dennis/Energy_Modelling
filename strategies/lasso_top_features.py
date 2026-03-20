"""Lasso strategy restricted to the Top-10 correlated features.

Uses only the ten features most correlated with the direction signal (from
EDA), keeping the model parsimonious.  Alpha is fixed at 0.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import TOP10_FEATURE_COLS, _MLStrategyBase

_ALPHA = 0.1


class LassoTopFeaturesStrategy(_MLStrategyBase):
    """Lasso regression on the Top-10 EDA-selected features.

    Alpha is fixed at 0.1 — no cross-validation search.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data, candidate_cols=TOP10_FEATURE_COLS)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_train(train_data)
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5
        self._pipeline = Pipeline(
            [("s", StandardScaler()), ("m", Lasso(alpha=_ALPHA, max_iter=5000))]
        )
        self._pipeline.fit(X, y)

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price
        x = self._get_x_row(state, self._feature_cols)
        return state.last_settlement_price + float(self._pipeline.predict(x)[0])

    def reset(self) -> None:
        pass
