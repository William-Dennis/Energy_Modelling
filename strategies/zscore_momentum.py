"""Rolling Z-score momentum strategy.

Uses the 20-day price Z-score (``price_zscore_20d`` derived feature) as a
momentum indicator: when prices are significantly above their recent mean
(high Z-score), momentum is bullish; when significantly below, bearish.

Different from PriceZScoreReversionStrategy, which *reverts*.  This strategy
follows the Z-score direction — a pure momentum signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DEFAULT_ZSCORE = 0.0
_DEFAULT_THRESHOLD = 0.5  # abs Z-score required to trade


class ZScoreMomentumStrategy(BacktestStrategy):
    """Trade in the direction of the Z-score when |z| > threshold.

    The threshold is calibrated to the 40th percentile of |Z| in training.
    """

    def __init__(self) -> None:
        self._threshold: float = _DEFAULT_THRESHOLD

    def fit(self, train_data: pd.DataFrame) -> None:
        if "price_zscore_20d" not in train_data.columns:
            self.skip_buffer = 0.0
            return
        z = train_data["price_zscore_20d"].abs().fillna(0.0)
        self._threshold = float(np.percentile(z, 40))
        # skip_buffer: mean price change magnitude
        self.skip_buffer = float(np.median(np.abs(train_data["price_change_eur_mwh"]))) * 0.4

    def forecast(self, state: BacktestState) -> float:
        z = float(state.features.get("price_zscore_20d", _DEFAULT_ZSCORE))
        if abs(z) < self._threshold:
            return state.last_settlement_price  # no signal
        # Momentum: positive Z → expect further rise
        signal = 1.0 if z > 0 else -1.0
        return state.last_settlement_price + signal

    def reset(self) -> None:
        pass
