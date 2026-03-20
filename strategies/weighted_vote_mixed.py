"""Weighted vote ensemble (rule-based + ML mix).

Combines 3 rule-based and 3 ML classifiers.  Rule-based members get weight
1, ML classifiers get weight 2 (more data-driven).  The weighted vote
determines direction.
"""

from __future__ import annotations

from energy_modelling.backtest.types import BacktestState
from strategies.ensemble_base import _EnsembleBase
from strategies.wind_forecast import WindForecastStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.de_fr_spread import DEFRSpreadStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.random_forest_direction import RandomForestStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy

_WEIGHTS = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]


class WeightedVoteMixedStrategy(_EnsembleBase):
    """Weighted vote: rule-based (w=1) + ML classifiers (w=2)."""

    _MEMBERS = [
        WindForecastStrategy,
        GasTrendStrategy,
        DEFRSpreadStrategy,
        LogisticDirectionStrategy,
        RandomForestStrategy,
        GradientBoostingStrategy,
    ]

    def forecast(self, state: BacktestState) -> float:
        dirs = self._get_member_directions(state)
        vote = sum(w * d for w, d in zip(_WEIGHTS, dirs))
        if vote > 0:
            return state.last_settlement_price + 1.0
        if vote < 0:
            return state.last_settlement_price - 1.0
        return state.last_settlement_price
