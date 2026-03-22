"""Tests for Phase 12B Batch 6 -- milestone strategy #100.

Covers:
1. BalancedLongShortStrategy
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.balanced_long_short import BalancedLongShortStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_multi(
    features: dict[str, float],
    last_price: float = 50.0,
) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 7, 15),
        last_settlement_price=last_price,
        features=pd.Series(features),
        history=pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# BalancedLongShortStrategy
# ---------------------------------------------------------------------------


class TestBalancedLongShortStrategy:
    def test_is_strategy(self) -> None:
        assert issubclass(BalancedLongShortStrategy, BacktestStrategy)

    def test_raises_before_fit(self) -> None:
        s = BalancedLongShortStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.forecast(_state_multi({}))

    def test_negative_zscore_goes_long(self) -> None:
        s = BalancedLongShortStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        features = {"price_zscore_20d": -1.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast > 50.0  # long

    def test_positive_zscore_goes_short(self) -> None:
        s = BalancedLongShortStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)
        features = {"price_zscore_20d": 1.0}
        forecast = s.forecast(_state_multi(features))
        assert forecast < 50.0  # short

    def test_rebalance_forces_short_when_too_long(self) -> None:
        s = BalancedLongShortStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)

        # Force 6 consecutive longs to exceed threshold (5)
        for _ in range(6):
            s.forecast(_state_multi({"price_zscore_20d": -2.0}))

        # Now bias > 5, should force short even with negative zscore
        forecast = s.forecast(_state_multi({"price_zscore_20d": -2.0}))
        assert forecast < 50.0  # forced short

    def test_rebalance_forces_long_when_too_short(self) -> None:
        s = BalancedLongShortStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)

        # Force 6 consecutive shorts to exceed threshold (-5)
        for _ in range(6):
            s.forecast(_state_multi({"price_zscore_20d": 2.0}))

        # Now bias < -5, should force long even with positive zscore
        forecast = s.forecast(_state_multi({"price_zscore_20d": 2.0}))
        assert forecast > 50.0  # forced long

    def test_fit_resets_bias(self) -> None:
        s = BalancedLongShortStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)

        # Accumulate bias
        for _ in range(3):
            s.forecast(_state_multi({"price_zscore_20d": -1.0}))
        assert s._cumulative_bias == 3

        # Re-fit should reset
        s.fit(df)
        assert s._cumulative_bias == 0

    def test_reset_clears_bias(self) -> None:
        s = BalancedLongShortStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)

        for _ in range(3):
            s.forecast(_state_multi({"price_zscore_20d": -1.0}))
        assert s._cumulative_bias == 3

        s.reset()
        assert s._cumulative_bias == 0

    def test_cumulative_tracking(self) -> None:
        s = BalancedLongShortStrategy()
        df = pd.DataFrame({"price_change_eur_mwh": [1.0, -1.0, 2.0]})
        s.fit(df)

        # Long (+1)
        s.forecast(_state_multi({"price_zscore_20d": -1.0}))
        assert s._cumulative_bias == 1

        # Short (-1)
        s.forecast(_state_multi({"price_zscore_20d": 1.0}))
        assert s._cumulative_bias == 0

        # Long (+1)
        s.forecast(_state_multi({"price_zscore_20d": -1.0}))
        assert s._cumulative_bias == 1
