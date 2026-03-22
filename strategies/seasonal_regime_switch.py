"""Seasonal regime switch strategy.

Switches between two trading modes based on the season:
- Winter (Nov-Mar): higher demand, follow gas/carbon trends (long bias when
  gas or carbon rising, short when falling).
- Summer (Apr-Oct): renewable-dominated, follow renewable penetration
  (high penetration -> short, low -> long).

Signal:
    Winter: gas_trend_3d > 0 -> long, gas_trend_3d <= 0 -> short
    Summer: renewable_pct > median -> short, renewable_pct <= median -> long
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_GAS_TREND_COL = "gas_trend_3d"
_RENEW_COL = "renewable_penetration_pct"
_WINTER_MONTHS = {11, 12, 1, 2, 3}


class SeasonalRegimeSwitchStrategy(BacktestStrategy):
    """Switch trading logic based on winter/summer season.

    Winter: follow gas trend direction.
    Summer: trade against renewable penetration.
    """

    def __init__(self) -> None:
        self._median_renew: float | None = None
        self._mean_abs_change: float = 1.0

    def fit(self, train_data: pd.DataFrame) -> None:
        if _RENEW_COL in train_data.columns:
            self._median_renew = float(train_data[_RENEW_COL].median())
        else:
            self._median_renew = 30.0  # reasonable default %

        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.4

    def forecast(self, state: BacktestState) -> float:
        if self._median_renew is None:
            raise RuntimeError("SeasonalRegimeSwitchStrategy.forecast() called before fit()")

        month = state.delivery_date.month

        if month in _WINTER_MONTHS:
            # Winter: follow gas trend
            gas_trend = float(state.features.get(_GAS_TREND_COL, 0.0))
            direction = 1 if gas_trend > 0 else -1
        else:
            # Summer: trade against renewable penetration
            renew = float(state.features.get(_RENEW_COL, 0.0))
            direction = -1 if renew > self._median_renew else 1

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
