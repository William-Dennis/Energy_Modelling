"""Partial Least Squares regression strategy.

PLS regression finds latent components that maximise covariance between X and
y, which is useful when features are highly collinear (as in energy data).
The number of components ``n_components`` is chosen by 5-fold TimeSeriesSplit
CV from {1, 2, 3, 5, 8}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_N_COMP_GRID = [1, 2, 3, 5, 8]
_N_SPLITS = 5


class PLSRegressionStrategy(_MLStrategyBase):
    """Partial Least Squares regression; n_components chosen by time-series CV."""

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

        best_n, best_mse = 1, float("inf")
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
        for n in _N_COMP_GRID:
            n = min(n, X.shape[1], X.shape[0] - 1)
            pls = PLSRegression(n_components=n)
            mses = []
            for ti, vi in tscv.split(X):
                pls.fit(X[ti], y[ti])
                pred = pls.predict(X[vi]).ravel()
                mses.append(float(np.mean((pred - y[vi]) ** 2)))
            if (m := float(np.mean(mses))) < best_mse:
                best_mse, best_n = m, n

        best_n = min(best_n, X.shape[1], X.shape[0] - 1)
        self._pls = PLSRegression(n_components=best_n)
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
