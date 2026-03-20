"""Consensus signal ensemble: trade only when all members agree.

Uses 3 diverse strategies (WindForecast, GasTrend, ZScoreMomentum).
Only trades when all 3 agree on direction; otherwise skips.
High-precision / low-recall approach.
"""

from __future__ import annotations

from energy_modelling.backtest.types import BacktestState
from strategies.ensemble_base import _EnsembleBase
from strategies.wind_forecast import WindForecastStrategy
from strategies.gas_trend import GasTrendStrategy
from strategies.zscore_momentum import ZScoreMomentumStrategy


class ConsensusSignalStrategy(_EnsembleBase):
    """Trade only when all 3 members unanimously agree on direction."""

    _MEMBERS = [
        WindForecastStrategy,
        GasTrendStrategy,
        ZScoreMomentumStrategy,
    ]

    def forecast(self, state: BacktestState) -> float:
        dirs = self._get_member_directions(state)
        if all(d == 1.0 for d in dirs):
            return state.last_settlement_price + 1.0
        if all(d == -1.0 for d in dirs):
            return state.last_settlement_price - 1.0
        return state.last_settlement_price  # no consensus
