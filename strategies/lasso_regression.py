"""Lasso regression strategy for the challenge dashboard.

Trains a Lasso (L1-regularised linear regression) model on the full set of
available features to predict the next day's price change.  L1 regularisation
automatically zeroes out uninformative features, producing a sparse model that
focuses on the most predictive signals.

Training
--------
* All available numeric features in the training split are used.
* Features are standardised (zero mean, unit variance) before fitting so that
  the regularisation penalty is applied fairly across different scales.
* The regularisation strength alpha is chosen by 5-fold time-series
  cross-validation (``TimeSeriesSplit``) over a log-spaced grid.

Signal
------
    predicted_change > +skip_buffer  → long  (+1)
    predicted_change < -skip_buffer  → short (-1)
    otherwise                        → skip  (None)

The skip_buffer is set to half the median absolute price change observed in
training, filtering out low-conviction signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

# Columns that must never be used as features (labels / forward-looking)
_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {
        "delivery_date",
        "split",
        "settlement_price",
        "price_change_eur_mwh",
        "target_direction",
        "pnl_long_eur",
        "pnl_short_eur",
        "last_settlement_price",
    }
)

# Alpha candidates evaluated during cross-validation
_ALPHA_GRID: list[float] = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

_N_CV_SPLITS: int = 5


class LassoRegressionStrategy(BacktestStrategy):
    """L1-regularised linear regression on all available features.

    Uses ``TimeSeriesSplit`` CV to select the Lasso alpha that minimises
    mean-squared error on the training set, then fits a final model on the
    full training window.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        # Identify usable numeric feature columns
        self._feature_cols = [
            col
            for col in train_data.columns
            if col not in _EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(train_data[col])
        ]

        X = train_data[self._feature_cols].fillna(0.0).values.astype(float)
        y = train_data["price_change_eur_mwh"].values.astype(float)

        # Calibrate skip_buffer to half the median absolute price change
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5

        # Time-series CV to choose alpha
        best_alpha = _ALPHA_GRID[0]
        best_mse = float("inf")
        tscv = TimeSeriesSplit(n_splits=_N_CV_SPLITS)

        for alpha in _ALPHA_GRID:
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("lasso", Lasso(alpha=alpha, max_iter=5000))]
            )
            fold_mses: list[float] = []
            for train_idx, val_idx in tscv.split(X):
                pipe.fit(X[train_idx], y[train_idx])
                preds = pipe.predict(X[val_idx])
                fold_mses.append(float(np.mean((preds - y[val_idx]) ** 2)))
            mean_mse = float(np.mean(fold_mses))
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_alpha = alpha

        # Fit final model on full training data with the chosen alpha
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lasso", Lasso(alpha=best_alpha, max_iter=5000)),
            ]
        )
        self._pipeline.fit(X, y)

    def forecast(self, state: BacktestState) -> float:
        if self._pipeline is None:
            return state.last_settlement_price

        x = np.array(
            [float(state.features.get(col, 0.0)) for col in self._feature_cols],
            dtype=float,
        ).reshape(1, -1)

        predicted_change = float(self._pipeline.predict(x)[0])
        return state.last_settlement_price + predicted_change

    def reset(self) -> None:
        pass
