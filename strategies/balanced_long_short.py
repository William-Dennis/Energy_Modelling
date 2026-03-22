"""Balanced long-short strategy (milestone strategy #100).

A meta-strategy that aims for balanced long/short exposure over time.
Tracks cumulative position bias (long vs short decisions) and adjusts
the forecast direction to maintain approximately equal long and short
exposure.

This is the 100th strategy in the Energy Modelling Platform, designed
to provide a structural balance signal that is independent of any
market feature.

Logic:
    1. Track cumulative net position (long=+1, short=-1)
    2. If net bias is too long (> threshold), favour short
    3. If net bias is too short (< -threshold), favour long
    4. Otherwise, follow a simple mean-reversion on price z-score

Signal:
    cumulative bias > threshold -> short (-1)  [rebalance]
    cumulative bias < -threshold -> long  (+1) [rebalance]
    otherwise -> z-score mean-reversion
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_ZSCORE_COL = "price_zscore_20d"
_BIAS_THRESHOLD = 5  # number of net positions before forcing rebalance


class BalancedLongShortStrategy(BacktestStrategy):
    """Milestone strategy #100: balanced long-short with position tracking.

    Maintains approximately equal long/short exposure over time by
    tracking cumulative net position and rebalancing when biased.
    """

    def __init__(self) -> None:
        self._cumulative_bias: int = 0
        self._mean_abs_change: float = 1.0
        self._fitted: bool = False

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        self._cumulative_bias = 0

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.3

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("BalancedLongShortStrategy.forecast() called before fit()")

        zscore = float(state.features.get(_ZSCORE_COL, 0.0))

        # Determine direction
        if self._cumulative_bias > _BIAS_THRESHOLD:
            # Too long-biased -> force short
            direction = -1
        elif self._cumulative_bias < -_BIAS_THRESHOLD:
            # Too short-biased -> force long
            direction = 1
        else:
            # Default: z-score mean-reversion
            direction = -1 if zscore > 0 else 1

        # Track the position we're about to take
        self._cumulative_bias += direction

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        self._cumulative_bias = 0
