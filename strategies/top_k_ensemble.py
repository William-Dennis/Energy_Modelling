"""Top-K ensemble: use only the top-K members by training accuracy.

Evaluates each member's directional accuracy on the last 20% of training
data and uses only the top-3 members for inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.ensemble_base import _EnsembleBase
from strategies.gas_trend import GasTrendStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy
from strategies.lasso_regression import LassoRegressionStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.random_forest_direction import RandomForestStrategy
from strategies.ridge_regression import RidgeRegressionStrategy

_TOP_K = 3


class TopKEnsembleStrategy(_EnsembleBase):
    """Keep only the top-3 members by validation directional accuracy."""

    _MEMBERS = [
        LogisticDirectionStrategy,
        RandomForestStrategy,
        GradientBoostingStrategy,
        LassoRegressionStrategy,
        RidgeRegressionStrategy,
        GasTrendStrategy,
    ]

    def __init__(self) -> None:
        super().__init__()
        self._top_k_members: list[BacktestStrategy] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        n = len(train_data)
        split = max(int(n * 0.8), 5)
        base_data = train_data.iloc[:split].copy()
        val_data = train_data.iloc[split:].copy()

        # Fit all members on base data
        self._fitted_members = []
        for cls in self._MEMBERS:
            m = cls()
            m.fit(base_data)
            self._fitted_members.append(m)

        # Evaluate accuracy on val_data
        accs: list[tuple[float, BacktestStrategy]] = []
        for m in self._fitted_members:
            correct = 0
            total = 0
            for _, row in val_data.iterrows():
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
                f = float(m.forecast(state))
                diff = f - state.last_settlement_price
                pred_dir = 0.0 if abs(diff) <= m.skip_buffer else 1.0 if diff > 0 else -1.0
                true_dir = float(row["target_direction"])
                if pred_dir != 0.0 and pred_dir == true_dir:
                    correct += 1
                total += 1
            acc = correct / max(total, 1)
            accs.append((acc, m))

        accs.sort(key=lambda x: x[0], reverse=True)
        top_k_idxs = [self._fitted_members.index(m) for _, m in accs[:_TOP_K]]

        # Re-fit top-K on full training data
        self._top_k_members = []
        for i in top_k_idxs:
            m = self._MEMBERS[i]()
            m.fit(train_data)
            self._top_k_members.append(m)

        self._fitted_members = self._top_k_members
        buffers = [m.skip_buffer for m in self._fitted_members]
        self.skip_buffer = float(np.median(buffers)) if buffers else 0.0

    def forecast(self, state: BacktestState) -> float:
        dirs = self._get_member_directions(state)
        vote = sum(dirs)
        if vote > 0:
            return state.last_settlement_price + 1.0
        if vote < 0:
            return state.last_settlement_price - 1.0
        return state.last_settlement_price
