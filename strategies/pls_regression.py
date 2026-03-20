"""Partial Least Squares regression strategy.

PLS regression finds latent components that maximise covariance between X and
y, which is useful when features are highly collinear (as in energy data).
The number of components ``n_components`` is fixed at 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_N_COMPONENTS = 4


class PLSRegressionStrategy(_MLStrategyBase):
    """Partial Least Squares regression; n_components fixed at 4."""

    def __init__(self) -> None:
        self._pls: PLSRegression | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        X_raw = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_train(train_data)
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X_raw)

        n = min(_N_COMPONENTS, X.shape[1], X.shape[0] - 1)
        self._pls = PLSRegression(n_components=n)
        self._pls.fit(X, y)

    def forecast(self, state: BacktestState) -> float:
        if self._pls is None or self._scaler is None:
            return state.last_settlement_price
        x_raw = self._get_x_row(state, self._feature_cols)
        x = self._scaler.transform(x_raw)
        change = float(self._pls.predict(x).ravel()[0])
        return state.last_settlement_price + change

    def reset(self) -> None:
        pass
