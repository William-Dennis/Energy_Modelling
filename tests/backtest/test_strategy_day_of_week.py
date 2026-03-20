"""Tests for DayOfWeekStrategy."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.day_of_week import DayOfWeekStrategy


def _make_state(delivery_date: date) -> BacktestState:
    """Build a minimal BacktestState for a given delivery date."""
    return BacktestState(
        delivery_date=delivery_date,
        last_settlement_price=50.0,
        features=pd.Series({"load_forecast_mw_mean": 40_000.0}, dtype=float),
        history=pd.DataFrame(),
    )


def _make_train_data() -> pd.DataFrame:
    """Training data with clear day-of-week patterns.

    Generates 4 weeks where Mon/Tue have positive changes, Wed/Thu are
    near zero, and Fri/Sat/Sun have negative changes.
    """
    rows = []
    base = date(2023, 12, 4)  # a Monday
    signals = {
        0: 5.0,  # Monday
        1: 3.0,  # Tuesday
        2: 0.0,  # Wednesday
        3: 0.0,  # Thursday
        4: -4.0,  # Friday
        5: -6.0,  # Saturday
        6: -5.0,  # Sunday
    }
    for week in range(4):
        for dow, change in signals.items():
            d = base + timedelta(days=week * 7 + dow)
            rows.append({"delivery_date": d, "price_change_eur_mwh": change})
    return pd.DataFrame(rows)


class TestDayOfWeekInterface:
    """Strategy satisfies the BacktestStrategy ABC."""

    def test_is_backtest_strategy(self) -> None:
        assert issubclass(DayOfWeekStrategy, BacktestStrategy)

    def test_fit_accepts_dataframe(self) -> None:
        s = DayOfWeekStrategy()
        s.fit(pd.DataFrame({"col": [1, 2, 3]}))

    def test_reset_callable(self) -> None:
        s = DayOfWeekStrategy()
        s.reset()  # should not raise


class TestDayOfWeekSignal:
    """Core signal: day-of-week maps to direction."""

    def test_monday_long(self) -> None:
        # 2024-01-01 is a Monday
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 1))) == 1

    def test_tuesday_long(self) -> None:
        # 2024-01-02 is a Tuesday
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 2))) == 1

    def test_wednesday_skip(self) -> None:
        # 2024-01-03 is a Wednesday
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 3))) is None

    def test_thursday_skip(self) -> None:
        # 2024-01-04 is a Thursday
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 4))) is None

    def test_friday_short(self) -> None:
        # 2024-01-05 is a Friday
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 5))) == -1

    def test_saturday_short(self) -> None:
        # 2024-01-06 is a Saturday
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 6))) == -1

    def test_sunday_short(self) -> None:
        # 2024-01-07 is a Sunday
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 7))) == -1


class TestDayOfWeekMultipleDays:
    """Ensure strategy is stateless across calls."""

    def test_consistent_across_multiple_calls(self) -> None:
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        # Monday then Friday then Monday
        assert s.act(_make_state(date(2024, 1, 1))) == 1
        assert s.act(_make_state(date(2024, 1, 5))) == -1
        assert s.act(_make_state(date(2024, 1, 8))) == 1  # next Monday

    def test_reset_does_not_change_behavior(self) -> None:
        s = DayOfWeekStrategy()
        s.fit(_make_train_data())
        assert s.act(_make_state(date(2024, 1, 1))) == 1
        s.reset()
        assert s.act(_make_state(date(2024, 1, 1))) == 1
