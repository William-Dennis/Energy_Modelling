"""Shared base class for ML-based strategies.

Provides common infrastructure:
- Column exclusion (labels, forward-looking columns)
- Feature vector construction at forecast time (handles missing cols)
- NaN filling

Each subclass provides ``fit()`` to build a model and ``forecast()`` to
apply it. The base class exposes ``_get_X_train()``, ``_get_x_row()``,
and ``_get_y_train()`` helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

# Columns never used as features (labels / forward-looking)
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

# Derived-feature column names (Phase A)
DERIVED_FEATURE_COLS: list[str] = [
    "net_demand_mw",
    "renewable_penetration_pct",
    "de_fr_spread",
    "de_nl_spread",
    "de_avg_neighbour_spread",
    "price_zscore_20d",
    "price_range",
    "gas_trend_3d",
    "carbon_trend_3d",
    "fuel_cost_index",
    "wind_forecast_error",
    "load_surprise",
    "rolling_vol_7d",
    "rolling_vol_14d",
    "total_fossil_mw",
    "net_flow_mw",
    "dow_int",
    "is_weekend",
]

# Top-10 features by correlation with direction (from EDA)
TOP10_FEATURE_COLS: list[str] = [
    "load_forecast_mw_mean",
    "forecast_wind_offshore_mw_mean",
    "gen_wind_onshore_mw_mean",
    "weather_wind_speed_10m_kmh_mean",
    "gen_fossil_gas_mw_mean",
    "flow_nl_net_import_mw_mean",
    "gen_fossil_brown_coal_lignite_mw_mean",
    "forecast_wind_onshore_mw_mean",
    "gen_wind_offshore_mw_mean",
    "gen_fossil_hard_coal_mw_mean",
]


class _MLStrategyBase(BacktestStrategy):
    """Base class for all ML-based strategies.

    Subclasses must implement ``fit()`` and ``forecast()``. They should
    call ``_get_feature_cols()``, ``_get_X_train()``, ``_get_y_train()``,
    and ``_get_x_row()`` to keep feature handling consistent.
    """

    def _get_feature_cols(
        self,
        train_data: pd.DataFrame,
        candidate_cols: list[str] | None = None,
    ) -> list[str]:
        """Return usable numeric feature columns from *train_data*.

        If *candidate_cols* is provided, only those columns (that are
        present and numeric) are considered.  Otherwise all non-excluded
        numeric columns are returned.
        """
        if candidate_cols is not None:
            return [
                c
                for c in candidate_cols
                if c in train_data.columns and pd.api.types.is_numeric_dtype(train_data[c])
            ]
        return [
            c
            for c in train_data.columns
            if c not in _EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(train_data[c])
        ]

    def _get_X_train(self, train_data: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        """Return (N, F) float array for training, NaN-filled with 0."""
        return train_data[feature_cols].fillna(0.0).values.astype(float)

    def _get_y_train(self, train_data: pd.DataFrame) -> np.ndarray:
        """Return price-change target vector."""
        return train_data["price_change_eur_mwh"].values.astype(float)

    def _get_y_direction(self, train_data: pd.DataFrame) -> np.ndarray:
        """Return direction target vector (+1 / -1)."""
        return train_data["target_direction"].values.astype(float)

    def _get_x_row(self, state: BacktestState, feature_cols: list[str]) -> np.ndarray:
        """Return (1, F) float array for prediction, handling missing cols."""
        return np.array(
            [float(state.features.get(col, 0.0)) for col in feature_cols],
            dtype=float,
        ).reshape(1, -1)
