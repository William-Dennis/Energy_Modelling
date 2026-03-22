"""High-volatility skip strategy.

During extreme volatility regimes, price movements are dominated by
noise.  This strategy stays neutral (skip) when volatility is in the
top quartile and applies a simple mean-reversion signal otherwise.

Uses ``rolling_vol_14d`` (the 14-day rolling volatility) to detect
high-volatility regimes.

Signal:
    vol_14d > p75(training) -> neutral (skip)
    price > overall_median -> short (-1)
    price <= overall_median -> long  (+1)
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_VOL_COL = "rolling_vol_14d"
_PRICE_COL = "price_mean"


class HighVolSkipStrategy(BacktestStrategy):
    """Skip trading during high-volatility regimes; otherwise mean-revert.

    Avoids the noise-dominated top quartile of volatility.
    """

    def __init__(self) -> None:
        self._p75_vol: float | None = None
        self._median_price: float = 50.0
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _VOL_COL in train_data.columns:
            self._p75_vol = float(train_data[_VOL_COL].quantile(0.75))
        else:
            self._p75_vol = float("inf")

        if _PRICE_COL in train_data.columns:
            self._median_price = float(train_data[_PRICE_COL].median())

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._p75_vol is None:
            raise RuntimeError("HighVolSkipStrategy.forecast() called before fit()")

        vol = float(state.features.get(_VOL_COL, 0.0))

        if vol > self._p75_vol:
            return state.last_settlement_price  # neutral (skip)

        current = state.last_settlement_price
        direction = -1 if current > self._median_price else 1
        return current + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
