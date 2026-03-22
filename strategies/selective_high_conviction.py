"""Selective high-conviction strategy (meta-wrapper).

Wraps CompositeSignalStrategy and only trades on days where the underlying
signal is unusually strong. On low-conviction days, the forecast equals the
entry price (triggering a skip via skip_buffer).

During fit(), computes the training-set standard deviation of the composite
signal's forecast deviation from entry. During forecast(), checks whether
today's signal exceeds a z-score threshold of 1.5.

Signal:
    |forecast - entry| > 1.5 * training_std  -> trade (pass through forecast)
    otherwise                                -> skip  (forecast = entry)

Source: Phase 10e found the market degrades per-date accuracy by over-weighting
strategies that are profitable overall but poor on individual days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.composite_signal import CompositeSignalStrategy

_Z_THRESHOLD = 1.5


class SelectiveHighConvictionStrategy(BacktestStrategy):
    """Only trade when the composite signal is unusually strong (|z| > 1.5)."""

    def __init__(self) -> None:
        self._inner: CompositeSignalStrategy = CompositeSignalStrategy()
        self._forecast_std: float = 1.0
        self._fitted: bool = False

    def fit(self, train_data: pd.DataFrame) -> None:
        self._inner.fit(train_data)
        self._fitted = True

        # Estimate the standard deviation of forecast deviations on training data
        deviations: list[float] = []
        if "last_settlement_price" in train_data.columns:
            price_col = "last_settlement_price"
        elif "settlement_price" in train_data.columns:
            price_col = "settlement_price"
        else:
            self._forecast_std = 1.0
            return

        for idx in range(len(train_data)):
            row = train_data.iloc[idx]
            entry = float(row[price_col])
            try:
                state = BacktestState(
                    delivery_date=pd.Timestamp(row["delivery_date"]).date()
                    if "delivery_date" in train_data.columns
                    else pd.Timestamp("2024-01-01").date(),
                    last_settlement_price=entry,
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
                        ]
                    ),
                    history=train_data.iloc[:idx],
                )
                forecast = self._inner.forecast(state)
                deviations.append(forecast - entry)
            except (KeyError, ValueError):
                continue

        if deviations:
            self._forecast_std = float(np.std(deviations))
            if self._forecast_std <= 0:
                self._forecast_std = 1.0

        self.skip_buffer = 0.0

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("SelectiveHighConvictionStrategy.forecast() called before fit()")
        inner_forecast = self._inner.forecast(state)
        deviation = inner_forecast - state.last_settlement_price
        z_score = abs(deviation) / self._forecast_std if self._forecast_std > 0 else 0.0

        if z_score >= _Z_THRESHOLD:
            return inner_forecast  # high conviction: pass through
        return state.last_settlement_price  # low conviction: skip

    def reset(self) -> None:
        self._inner.reset()
