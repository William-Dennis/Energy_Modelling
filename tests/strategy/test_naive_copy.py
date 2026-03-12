"""Tests for strategy.naive_copy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.market_simulation.types import DayState, Trade
from energy_modelling.strategy.naive_copy import NaiveCopyStrategy


def _make_day_state(
    delivery_date: date = date(2024, 1, 15),
    last_settlement: float = 50.0,
) -> DayState:
    """Create a minimal DayState for testing."""
    return DayState(
        delivery_date=delivery_date,
        last_settlement_price=last_settlement,
        features=pd.DataFrame({"load_actual_mw_mean": [55000.0]}, index=[delivery_date]),
        neighbor_prices={"FR": 55.0, "NL": 52.0},
    )


class TestNaiveCopyStrategy:
    """Tests for NaiveCopyStrategy."""

    def test_returns_trade(self) -> None:
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        trade = strategy.act(state)
        assert isinstance(trade, Trade)

    def test_entry_price_matches_last_settlement(self) -> None:
        strategy = NaiveCopyStrategy()
        state = _make_day_state(last_settlement=42.5)
        trade = strategy.act(state)
        assert trade is not None
        assert trade.entry_price == pytest.approx(42.5)

    def test_position_is_long_1mw(self) -> None:
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        trade = strategy.act(state)
        assert trade is not None
        assert trade.position_mw == pytest.approx(1.0)

    def test_delivery_date_matches(self) -> None:
        strategy = NaiveCopyStrategy()
        target = date(2024, 6, 15)
        state = _make_day_state(delivery_date=target)
        trade = strategy.act(state)
        assert trade is not None
        assert trade.delivery_date == target

    def test_hours_is_24(self) -> None:
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        trade = strategy.act(state)
        assert trade is not None
        assert trade.hours == 24

    def test_negative_price(self) -> None:
        """Strategy should still trade when last settlement is negative."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state(last_settlement=-15.0)
        trade = strategy.act(state)
        assert trade is not None
        assert trade.entry_price == pytest.approx(-15.0)

    def test_reset_is_noop(self) -> None:
        """Reset should not raise."""
        strategy = NaiveCopyStrategy()
        strategy.reset()
