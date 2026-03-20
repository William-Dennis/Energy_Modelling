"""Tests for WeeklyCycleStrategy."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.weekly_cycle import WeeklyCycleStrategy


def _make_history(changes: list[float], end_date: date = date(2024, 1, 14)) -> pd.DataFrame:
    """Build a history DataFrame. Last entry in `changes` is most recent day."""
    n = len(changes)
    dates = [end_date - timedelta(days=n - 1 - i) for i in range(n)]
    return pd.DataFrame(
        {
            "delivery_date": dates,
            "price_change_eur_mwh": changes,
            "settlement_price": [50.0 + sum(changes[: i + 1]) for i in range(n)],
            "last_settlement_price": [50.0 + sum(changes[:i]) for i in range(n)],
        }
    ).set_index(pd.Index(dates, name="delivery_date"))


def _make_state(
    changes: list[float],
    delivery_date: date = date(2024, 1, 15),
) -> BacktestState:
    return BacktestState(
        delivery_date=delivery_date,
        last_settlement_price=50.0,
        features=pd.Series({"load_forecast_mw_mean": 40_000.0}, dtype=float),
        history=_make_history(changes, end_date=delivery_date - timedelta(days=1)),
    )


class TestWeeklyCycleInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(WeeklyCycleStrategy, BacktestStrategy)

    def test_fit_accepts_dataframe(self) -> None:
        s = WeeklyCycleStrategy()
        s.fit(pd.DataFrame({"col": [1, 2, 3]}))

    def test_reset_callable(self) -> None:
        s = WeeklyCycleStrategy()
        s.reset()


class TestWeeklyCycleSignal:
    """Core signal: same-day-of-week persistence."""

    def test_positive_7d_ago_long(self) -> None:
        s = WeeklyCycleStrategy()
        # 7 days of history; the first entry (7 days ago) is +10
        changes = [10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert s.act(_make_state(changes)) == 1

    def test_negative_7d_ago_short(self) -> None:
        s = WeeklyCycleStrategy()
        changes = [-10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert s.act(_make_state(changes)) == -1

    def test_zero_7d_ago_skip(self) -> None:
        s = WeeklyCycleStrategy()
        changes = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert s.act(_make_state(changes)) is None


class TestWeeklyCycleEdgeCases:
    def test_insufficient_history_skip(self) -> None:
        """With fewer than 7 days of history, strategy should skip."""
        s = WeeklyCycleStrategy()
        assert s.act(_make_state([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])) is None

    def test_empty_history_skip(self) -> None:
        s = WeeklyCycleStrategy()
        state = BacktestState(
            delivery_date=date(2024, 1, 15),
            last_settlement_price=50.0,
            features=pd.Series({"load_forecast_mw_mean": 40_000.0}),
            history=pd.DataFrame(),
        )
        assert s.act(state) is None

    def test_exactly_seven_days(self) -> None:
        """Exactly 7 days of history — the earliest is the lag-7 row."""
        s = WeeklyCycleStrategy()
        changes = [5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        assert s.act(_make_state(changes)) == 1

    def test_more_than_seven_days(self) -> None:
        """With 10 days of history, we still look at -7 from the end."""
        s = WeeklyCycleStrategy()
        # 10 days; day at index -7 (from end) has change -20
        changes = [1.0, 1.0, 1.0, -20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert s.act(_make_state(changes)) == -1
