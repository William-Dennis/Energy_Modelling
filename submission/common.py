"""Shared utilities for simple challenge submission strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from energy_modelling.challenge.runner import _STATE_EXCLUDE_COLUMNS as _EXCLUDED_COLUMNS
from energy_modelling.challenge.types import ChallengeState


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric feature columns allowed at decision time."""

    cols: list[str] = []
    for column in frame.columns:
        if column in _EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            cols.append(column)
    return cols


def state_value(state: ChallengeState, column: str, default: float = 0.0) -> float:
    """Safely read one feature value from the current state."""

    value = state.features.get(column, default)
    if pd.isna(value):
        return float(default)
    return float(value)


def history_value(state: ChallengeState, column: str, default: float = 0.0) -> float:
    """Safely read the most recent historical value for a column."""

    if state.history.empty or column not in state.history.columns:
        return float(default)
    value = state.history.iloc[-1][column]
    if pd.isna(value):
        return float(default)
    return float(value)


@dataclass
class LinearDirectionModel:
    """Tiny ridge-style linear model for direction prediction."""

    columns: list[str]
    means: pd.Series
    scales: pd.Series
    weights: np.ndarray

    @classmethod
    def fit(
        cls,
        train_data: pd.DataFrame,
        target_column: str = "target_direction",
        ridge_penalty: float = 1.0,
    ) -> "LinearDirectionModel":
        columns = feature_columns(train_data)
        features = train_data[columns].astype(float)
        means = features.median()
        features = features.fillna(means)
        scales = features.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        x = ((features - means) / scales).to_numpy(dtype=float)
        intercept = np.ones((len(x), 1), dtype=float)
        x_design = np.hstack([intercept, x])
        y = train_data[target_column].astype(float).to_numpy(dtype=float)
        ridge = ridge_penalty * np.eye(x_design.shape[1], dtype=float)
        ridge[0, 0] = 0.0
        weights = np.linalg.solve(x_design.T @ x_design + ridge, x_design.T @ y)
        return cls(columns=columns, means=means, scales=scales, weights=weights)

    def predict_score(self, state: ChallengeState) -> float:
        row = pd.Series(
            {
                column: state_value(state, column, float(self.means[column]))
                for column in self.columns
            }
        )
        x = ((row - self.means) / self.scales).to_numpy(dtype=float)
        x_design = np.concatenate(([1.0], x))
        return float(x_design @ self.weights)
