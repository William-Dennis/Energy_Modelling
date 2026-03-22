"""Median independent strategy.

Takes the median of 3 independent rule-based signals:
1. Price z-score mean reversion
2. Gas trend momentum
3. Load-surprise signal

By using the median (instead of mean or majority vote), this strategy
is robust to a single outlier signal.

Signal:
    median(zscore_signal, gas_signal, load_signal) -> direction
"""

from __future__ import annotations

import statistics

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy


class MedianIndependentStrategy(BacktestStrategy):
    """Median of 3 independent signals for robust direction.

    Takes the median of z-score, gas trend, and load surprise signals.
    """

    def __init__(self) -> None:
        self._median_zscore: float = 0.0
        self._median_load_surprise: float = 0.0
        self._mean_abs_change: float = 1.0
        self._fitted: bool = False

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True

        if "price_zscore_20d" in train_data.columns:
            self._median_zscore = float(train_data["price_zscore_20d"].median())
        if "load_surprise" in train_data.columns:
            self._median_load_surprise = float(train_data["load_surprise"].median())

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("MedianIndependentStrategy.forecast() called before fit()")

        # Signal 1: z-score reversion (high zscore -> short)
        zscore = float(state.features.get("price_zscore_20d", 0.0))
        sig_zscore = -1 if zscore > self._median_zscore else 1

        # Signal 2: gas trend (rising gas -> long)
        gas_trend = float(state.features.get("gas_trend_3d", 0.0))
        sig_gas = 1 if gas_trend > 0 else -1

        # Signal 3: load surprise (high surprise -> long)
        load_surprise = float(state.features.get("load_surprise", 0.0))
        sig_load = 1 if load_surprise > self._median_load_surprise else -1

        direction = statistics.median([sig_zscore, sig_gas, sig_load])
        if direction == 0:
            return state.last_settlement_price

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
