"""Tests for strategy.naive_copy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.market_simulation.types import DayState, Signal
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

    def test_returns_signal(self) -> None:
        """act() must return a Signal instance, never None."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        signal = strategy.act(state)
        assert isinstance(signal, Signal)

    def test_never_returns_none(self) -> None:
        """NaiveCopy always trades -- it never skips a day."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        assert strategy.act(state) is not None

    def test_direction_is_long(self) -> None:
        """Strategy always signals long (+1)."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == 1

    def test_delivery_date_matches(self) -> None:
        """Signal delivery date must equal the state's delivery date."""
        strategy = NaiveCopyStrategy()
        target = date(2024, 6, 15)
        state = _make_day_state(delivery_date=target)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.delivery_date == target

    def test_direction_independent_of_price_level(self) -> None:
        """Direction is always +1 regardless of the settlement price."""
        strategy = NaiveCopyStrategy()
        for price in [0.0, 10.0, 50.0, 100.0, 200.0]:
            state = _make_day_state(last_settlement=price)
            signal = strategy.act(state)
            assert signal is not None
            assert signal.direction == 1, f"Expected long for price={price}"

    def test_direction_long_when_price_negative(self) -> None:
        """Strategy still signals long when last settlement is negative."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state(last_settlement=-15.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == 1

    def test_direction_long_when_price_zero(self) -> None:
        """Strategy signals long when last settlement is exactly zero."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state(last_settlement=0.0)
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == 1

    def test_no_entry_price_on_signal(self) -> None:
        """Signal must not expose an entry_price attribute -- that is the
        market's responsibility, not the strategy's."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        signal = strategy.act(state)
        assert signal is not None
        assert not hasattr(signal, "entry_price")

    def test_reset_is_noop(self) -> None:
        """reset() should not raise and strategy should behave identically."""
        strategy = NaiveCopyStrategy()
        strategy.reset()
        state = _make_day_state()
        signal = strategy.act(state)
        assert signal is not None
        assert signal.direction == 1

    def test_consistent_across_multiple_calls(self) -> None:
        """Strategy is stateless -- repeated calls with the same state
        produce the same signal."""
        strategy = NaiveCopyStrategy()
        state = _make_day_state()
        signals = [strategy.act(state) for _ in range(5)]
        assert all(s is not None and s.direction == 1 for s in signals)
