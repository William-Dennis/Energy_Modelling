"""Elastic Net regression strategy (L1 + L2 regularisation).

Combines the sparsity of Lasso with the stability of Ridge.  The mixing
parameter ``l1_ratio`` is fixed at 0.5 (equal L1/L2), and alpha is fixed
at 0.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_ALPHA = 0.1
_L1_RATIO = 0.5


class ElasticNetStrategy(_MLStrategyBase):
    """Elastic Net (L1+L2) regression on all available features.

    Alpha is fixed at 0.1, l1_ratio fixed at 0.5 — no cross-validation search.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_train(train_data)
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5
        self._pipeline = Pipeline(
            [
                ("s", StandardScaler()),
                ("m", ElasticNet(alpha=_ALPHA, l1_ratio=_L1_RATIO, max_iter=5000)),
            ]
        )
        self._pipeline.fit(X, y)

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price
        x = self._get_x_row(state, self._feature_cols)
        return state.last_settlement_price + float(self._pipeline.predict(x)[0])

    def reset(self) -> None:
        pass
