"""Regime-conditional Ridge regression strategy.

Trains two separate Ridge regression models: one for low-volatility periods
and one for high-volatility periods. During forecast, checks which regime
the current day falls in and uses the corresponding model.

Uses rolling_vol_7d as the regime indicator, split at the training-set median.

Source: Phase 10d found the ML cluster is more robust to volatility but
rule-based strategies show larger regime-dependent accuracy gaps. This
strategy adapts its model to the current volatility regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.ml_base import _MLStrategyBase

_VOL_COL = "rolling_vol_7d"


class RegimeRidgeStrategy(_MLStrategyBase):
    """Volatility-adaptive Ridge: low-vol model vs high-vol model."""

    def __init__(self) -> None:
        super().__init__()
        self._low_vol_pipeline: Pipeline | None = None
        self._high_vol_pipeline: Pipeline | None = None
        self._vol_threshold: float = 0.0
        self._feature_cols: list[str] = []
        self._fitted: bool = False

    def fit(self, train_data: pd.DataFrame) -> None:
        self._feature_cols = self._get_feature_cols(train_data)
        y = self._get_y_train(train_data)
        self.skip_buffer = float(np.median(np.abs(y))) * 0.5

        # Determine volatility regime threshold
        if _VOL_COL in train_data.columns:
            vol = train_data[_VOL_COL].values.astype(float)
            vol_clean = vol[np.isfinite(vol)]
            self._vol_threshold = float(np.median(vol_clean)) if len(vol_clean) > 0 else 0.0

            low_mask = train_data[_VOL_COL].fillna(0.0) <= self._vol_threshold
            high_mask = ~low_mask
        else:
            # No volatility column: treat all data as one regime
            low_mask = pd.Series([True] * len(train_data), index=train_data.index)
            high_mask = pd.Series([False] * len(train_data), index=train_data.index)
            self._vol_threshold = 0.0

        # Train low-vol model
        low_data = train_data[low_mask]
        if len(low_data) >= 5:
            X_low = self._get_X_train(low_data, self._feature_cols)
            y_low = self._get_y_train(low_data)
            self._low_vol_pipeline = Pipeline(
                [
                    ("s", StandardScaler()),
                    ("m", Ridge(alpha=1.0)),
                ]
            )
            self._low_vol_pipeline.fit(X_low, y_low)
        else:
            self._low_vol_pipeline = None

        # Train high-vol model
        high_data = train_data[high_mask]
        if len(high_data) >= 5:
            X_high = self._get_X_train(high_data, self._feature_cols)
            y_high = self._get_y_train(high_data)
            self._high_vol_pipeline = Pipeline(
                [
                    ("s", StandardScaler()),
                    ("m", Ridge(alpha=1.0)),
                ]
            )
            self._high_vol_pipeline.fit(X_high, y_high)
        else:
            self._high_vol_pipeline = None

        # Fallback: train on full data if either regime has too few samples
        if self._low_vol_pipeline is None or self._high_vol_pipeline is None:
            X_full = self._get_X_train(train_data, self._feature_cols)
            fallback = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
            fallback.fit(X_full, y)
            if self._low_vol_pipeline is None:
                self._low_vol_pipeline = fallback
            if self._high_vol_pipeline is None:
                self._high_vol_pipeline = fallback

        self._fitted = True

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("RegimeRidgeStrategy.forecast() called before fit()")

        x = self._get_x_row(state, self._feature_cols)

        # Determine current regime
        vol = float(state.features.get(_VOL_COL, 0.0))
        pipeline = self._low_vol_pipeline if vol <= self._vol_threshold else self._high_vol_pipeline

        if pipeline is None:
            return state.last_settlement_price

        predicted_change = float(pipeline.predict(x)[0])
        return state.last_settlement_price + predicted_change

    def reset(self) -> None:
        pass
