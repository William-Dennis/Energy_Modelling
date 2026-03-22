"""Spread momentum strategy using cross-border price differentials.

Uses a 3-day exponential moving average of the average DE-FR/DE-NL spread.
When the spread EMA is rising and positive (DE is expensive vs neighbours),
expects mean-reversion and goes short. When falling and negative, goes long.

Signal:
    spread_ema > prev_spread_ema  AND  spread_ema > 0  -> short (-1)
    spread_ema < prev_spread_ema  AND  spread_ema < 0  -> long  (+1)
    otherwise                                          -> neutral (skip)

Source: Phase 10f identified cross-border spread as the most positively
contributing signal family to market quality.
"""

from __future__ import annotations

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy

_DE_FR = "de_fr_spread"
_DE_NL = "de_nl_spread"


class SpreadMomentumStrategy(BacktestStrategy):
    """Short when cross-border spread is rising and positive; long when
    falling and negative. Uses 3-day EMA of the average DE-FR/DE-NL spread.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._mean_abs_change: float = 1.0
        self._prev_spread_ema: float = 0.0
        self._spread_ema: float = 0.0
        self._ema_alpha: float = 2.0 / (3 + 1)  # 3-day EMA span

    def fit(self, train_data: pd.DataFrame) -> None:
        self._fitted = True
        if "price_change_eur_mwh" in train_data.columns:
            mac = float(train_data["price_change_eur_mwh"].abs().mean())
            self._mean_abs_change = mac if mac > 0 else 1.0
            self.skip_buffer = float(train_data["price_change_eur_mwh"].abs().median()) * 0.3

        # Initialise EMA from last few training rows
        if _DE_FR in train_data.columns and _DE_NL in train_data.columns:
            avg_spread = (train_data[_DE_FR] + train_data[_DE_NL]) / 2.0
            ema = float(avg_spread.iloc[0]) if len(avg_spread) > 0 else 0.0
            for val in avg_spread.iloc[1:]:
                ema = self._ema_alpha * float(val) + (1 - self._ema_alpha) * ema
            self._spread_ema = ema
            self._prev_spread_ema = ema

    def forecast(self, state: BacktestState) -> float:
        if not self._fitted:
            raise RuntimeError("SpreadMomentumStrategy.forecast() called before fit()")

        de_fr = float(state.features.get(_DE_FR, 0.0))
        de_nl = float(state.features.get(_DE_NL, 0.0))
        avg_spread = (de_fr + de_nl) / 2.0

        self._prev_spread_ema = self._spread_ema
        self._spread_ema = self._ema_alpha * avg_spread + (1 - self._ema_alpha) * self._spread_ema

        rising = self._spread_ema > self._prev_spread_ema
        positive = self._spread_ema > 0

        if rising and positive:
            direction = -1  # DE expensive and getting more so -> short (reversion)
        elif (not rising) and (not positive):
            direction = 1  # DE cheap and getting cheaper -> long (reversion)
        else:
            return state.last_settlement_price  # no clear signal

        return state.last_settlement_price + direction * self._mean_abs_change

    def reset(self) -> None:
        pass
