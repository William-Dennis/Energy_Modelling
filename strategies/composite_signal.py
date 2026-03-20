"""Composite multi-feature signal strategy for the challenge dashboard.

Combines the top EDA-identified features into a weighted z-score signal
(Phase 3, H7). Each feature is standardised using training mean/std, then
multiplied by its EDA-derived correlation weight. The sign of the composite
score determines the trading direction.

Features and weights (from EDA correlation with price direction):
    load_forecast_mw_mean:                  +0.234
    forecast_wind_offshore_mw_mean:         -0.218
    forecast_wind_onshore_mw_mean:          -0.189
    gen_fossil_gas_mw_mean:                 -0.196
    gen_fossil_hard_coal_mw_mean:           -0.139
    gen_fossil_brown_coal_lignite_mw_mean:  -0.185

Signal:
    composite = sum(weight_i * z_score_i)
    composite > 0  → long  (+1)
    composite < 0  → short (-1)
    composite == 0 → skip  (None)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

# Feature names and their EDA-derived correlation weights
_FEATURE_WEIGHTS: list[tuple[str, float]] = [
    ("load_forecast_mw_mean", +0.234),
    ("forecast_wind_offshore_mw_mean", -0.218),
    ("forecast_wind_onshore_mw_mean", -0.189),
    ("gen_fossil_gas_mw_mean", -0.196),
    ("gen_fossil_hard_coal_mw_mean", -0.139),
    ("gen_fossil_brown_coal_lignite_mw_mean", -0.185),
]


class CompositeSignalStrategy(BacktestStrategy):
    """Weighted z-score composite of top directional features.

    During fit(), computes per-feature mean and std from training data.
    During act(), standardises today's features and computes the weighted sum.
    """

    def __init__(self) -> None:
        self._feature_names: list[str] = [name for name, _ in _FEATURE_WEIGHTS]
        self._weights: list[float] = [w for _, w in _FEATURE_WEIGHTS]
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._price_std: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        subset = train_data[self._feature_names]
        self._means = subset.mean().values.astype(float)
        self._stds = subset.std().values.astype(float)
        # Guard against zero std (constant feature)
        self._stds[self._stds == 0] = 1.0
        if "price_change_eur_mwh" in train_data.columns:
            self._price_std = float(train_data["price_change_eur_mwh"].std())
            if self._price_std <= 0:
                self._price_std = 1.0

    def _composite_score(self, state: BacktestState) -> float | None:
        if self._means is None or self._stds is None:
            msg = "CompositeSignalStrategy called before fit()"
            raise RuntimeError(msg)
        values = np.array(
            [float(state.features[name]) for name in self._feature_names],
            dtype=float,
        )
        z_scores = (values - self._means) / self._stds
        return float(np.dot(self._weights, z_scores))

    def act(self, state: BacktestState) -> int | None:
        composite = self._composite_score(state)
        if composite is None:
            return None
        if composite > 0:
            return 1
        if composite < 0:
            return -1
        return None

    def forecast(self, state: BacktestState) -> float:
        composite = self._composite_score(state)
        if composite is None:
            return state.last_settlement_price
        return state.last_settlement_price + composite * self._price_std

    def reset(self) -> None:
        pass
