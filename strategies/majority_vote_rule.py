"""Majority vote ensemble of rule-based strategies.

Combines 5 diverse rule-based strategies.  Trades long if majority vote
> 0, short if < 0, skip if tied.
"""

from __future__ import annotations

from energy_modelling.backtest.types import BacktestState
from strategies.ensemble_base import _EnsembleBase
from strategies.wind_forecast import WindForecastStrategy
from strategies.load_forecast import LoadForecastStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.carbon_trend import CarbonTrendStrategy
from strategies.day_of_week import DayOfWeekStrategy


class MajorityVoteRuleBasedStrategy(_EnsembleBase):
    """Majority vote over 5 rule-based strategies.

    Members: WindForecast, LoadForecast, GasTrend, CarbonTrend, DayOfWeek.
    """

    _MEMBERS = [
        WindForecastStrategy,
        LoadForecastStrategy,
        GasTrendStrategy,
        CarbonTrendStrategy,
        DayOfWeekStrategy,
    ]

    def forecast(self, state: BacktestState) -> float:
        dirs = self._get_member_directions(state)
        vote = sum(dirs)
        if vote > 0:
            return state.last_settlement_price + 1.0
        if vote < 0:
            return state.last_settlement_price - 1.0
        return state.last_settlement_price
