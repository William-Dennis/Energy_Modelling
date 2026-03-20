"""Lasso regression augmented with calendar features.

Adds ``dow_int`` and ``is_weekend`` (already computed by Phase A) to the full
feature set before running Lasso.  Calendar features may capture weekly
seasonality in energy prices.  Alpha is fixed at 0.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_ALPHA = 0.1
# Calendar columns guaranteed to exist after Phase A feature engineering
_CALENDAR_COLS = ["dow_int", "is_weekend"]


class LassoCalendarAugmentedStrategy(_MLStrategyBase):
    """Lasso regression with calendar features explicitly included.

    The full numeric feature set is used; calendar derived features from
    Phase A (``dow_int``, ``is_weekend``) are explicitly kept.
    Alpha is fixed at 0.1 — no cross-validation search.
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
