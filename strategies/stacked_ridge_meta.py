"""Stacked ensemble: Ridge meta-learner over ML base forecasts.

Base layer: Lasso, Ridge, ElasticNet, BayesianRidge (regression).
Meta layer: Ridge regression trained on base-layer out-of-fold predictions.

Uses a simple single-fold split (first 80% train, last 20% for meta-train)
to avoid look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from energy_modelling.backtest.types import BacktestState
from strategies.bayesian_ridge import BayesianRidgeStrategy
from strategies.elastic_net import ElasticNetStrategy
from strategies.ensemble_base import _EnsembleBase
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.ridge_regression import RidgeRegressionStrategy


class StackedRidgeMetaStrategy(_EnsembleBase):
    """Ridge meta-learner stacked over 4 regression base strategies."""

    _MEMBERS = [
        LassoRegressionStrategy,
        RidgeRegressionStrategy,
        ElasticNetStrategy,
        BayesianRidgeStrategy,
    ]

    def __init__(self) -> None:
        super().__init__()
        self._meta_ridge: Ridge | None = None
        self._meta_scaler: StandardScaler | None = None
        self._last_settlement_col: bool = False

    def fit(self, train_data: pd.DataFrame) -> None:
        n = len(train_data)
        split = max(int(n * 0.8), 5)  # at least 5 rows in meta-train
        base_train = train_data.iloc[:split].copy()
        meta_train = train_data.iloc[split:].copy()

        if len(meta_train) < 3:
            # Fallback: fit all members on full data, no meta layer
            super().fit(train_data)
            return

        # Fit base models on base_train
        self._fitted_members = []
        for cls in self._MEMBERS:
            m = cls()
            m.fit(base_train)
            self._fitted_members.append(m)

        # Build meta-features: base forecasts on meta_train rows
        meta_X_rows = []
        for _, row in meta_train.iterrows():
            state = BacktestState(
                delivery_date=row["delivery_date"],
                last_settlement_price=float(row["last_settlement_price"]),
                features=row.drop(
                    labels=[
                        c
                        for c in [
                            "delivery_date",
                            "split",
                            "settlement_price",
                            "price_change_eur_mwh",
                            "target_direction",
                            "pnl_long_eur",
                            "pnl_short_eur",
                        ]
                        if c in row.index
                    ],
                    errors="ignore",
                ),
                history=pd.DataFrame(),
            )
            meta_X_rows.append(self._get_member_forecasts(state))

        meta_X = np.array(meta_X_rows)
        meta_y = meta_train["settlement_price"].values.astype(float)

        self._meta_scaler = StandardScaler()
        meta_X_scaled = self._meta_scaler.fit_transform(meta_X)
        self._meta_ridge = Ridge(alpha=1.0)
        self._meta_ridge.fit(meta_X_scaled, meta_y)

        # Re-fit base models on full data for inference
        self._fitted_members = []
        for cls in self._MEMBERS:
            m = cls()
            m.fit(train_data)
            self._fitted_members.append(m)

        buffers = [m.skip_buffer for m in self._fitted_members]
        self.skip_buffer = float(np.median(buffers)) if buffers else 0.0

    def forecast(self, state: BacktestState) -> float:
        if self._meta_ridge is None or self._meta_scaler is None:
            forecasts = self._get_member_forecasts(state)
            return float(np.mean(forecasts))
        base_f = np.array(self._get_member_forecasts(state)).reshape(1, -1)
        base_f_scaled = self._meta_scaler.transform(base_f)
        return float(self._meta_ridge.predict(base_f_scaled)[0])
