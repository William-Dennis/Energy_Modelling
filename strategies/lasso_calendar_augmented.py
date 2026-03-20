"""Lasso regression augmented with calendar features.

Adds ``dow_int`` and ``is_weekend`` (already computed by Phase A) to the full
feature set before running Lasso CV.  Calendar features may capture weekly
seasonality in energy prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _EXCLUDE_COLUMNS, _MLStrategyBase

_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
_N_SPLITS = 5
# Calendar columns guaranteed to exist after Phase A feature engineering
_CALENDAR_COLS = ["dow_int", "is_weekend"]


class LassoCalendarAugmentedStrategy(_MLStrategyBase):
    """Lasso regression with calendar features explicitly included.

    The full numeric feature set is used; calendar derived features from
    Phase A (``dow_int``, ``is_weekend``) are explicitly kept even if Lasso
    would otherwise zero them out via a higher alpha.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        # All numeric non-excluded columns (includes calendar cols from Phase A)
        self._feature_cols = self._get_feature_cols(train_data)
        X = self._get_X_train(train_data, self._feature_cols)
        y = self._get_y_train(train_data)
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5
        best_alpha, best_mse = _ALPHA_GRID[0], float("inf")
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
        for alpha in _ALPHA_GRID:
            pipe = Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=alpha, max_iter=5000))])
            mses = [
                float(np.mean((pipe.fit(X[ti], y[ti]).predict(X[vi]) - y[vi]) ** 2))
                for ti, vi in tscv.split(X)
            ]
            if (m := float(np.mean(mses))) < best_mse:
                best_mse, best_alpha = m, alpha
        self._pipeline = Pipeline(
            [("s", StandardScaler()), ("m", Lasso(alpha=best_alpha, max_iter=5000))]
        )
        self._pipeline.fit(X, y)

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price
        x = self._get_x_row(state, self._feature_cols)
        return state.last_settlement_price + float(self._pipeline.predict(x)[0])

    def reset(self) -> None:
        pass
