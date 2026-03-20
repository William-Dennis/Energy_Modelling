"""Tests for Lag2ReversionStrategy."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.lag2_reversion import Lag2ReversionStrategy


def _make_history(changes: list[float], end_date: date = date(2024, 1, 14)) -> pd.DataFrame:
    """Build a history DataFrame with price_change_eur_mwh column.

    The last entry in `changes` is the most recent day.
    """
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
    """Build a BacktestState with a history that has the given price changes."""
    return BacktestState(
        delivery_date=delivery_date,
        last_settlement_price=50.0,
        features=pd.Series({"load_forecast_mw_mean": 40_000.0}, dtype=float),
        history=_make_history(changes, end_date=delivery_date - timedelta(days=1)),
    )


def _make_train_data() -> pd.DataFrame:
    """Minimal training data — lag2 reversion uses it to compute volatility threshold."""
    changes = [5.0, -3.0, 10.0, -8.0, 2.0, -1.0, 15.0, -12.0, 0.5, -0.3]
    return pd.DataFrame({"price_change_eur_mwh": changes})


class TestLag2ReversionInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(Lag2ReversionStrategy, BacktestStrategy)

    def test_fit_computes_threshold(self) -> None:
        s = Lag2ReversionStrategy()
        s.fit(_make_train_data())
        assert s.skip_buffer > 0

    def test_reset_preserves_threshold(self) -> None:
        s = Lag2ReversionStrategy()
        s.fit(_make_train_data())
        s.reset()
        assert s.skip_buffer > 0


class TestLag2ReversionSignal:
    """Core signal: big up 2 days ago → short, big down 2 days ago → long."""

    def test_big_up_two_days_ago_short(self) -> None:
        s = Lag2ReversionStrategy()
        s.fit(_make_train_data())
        # Force a very large positive change 2 days ago
        # History: [..., +100 (2 days ago), +1 (yesterday)]
        assert s.act(_make_state([100.0, 1.0])) == -1

    def test_big_down_two_days_ago_long(self) -> None:
        s = Lag2ReversionStrategy()
        s.fit(_make_train_data())
        # Large negative change 2 days ago
        assert s.act(_make_state([-100.0, 1.0])) == 1

    def test_small_change_skip(self) -> None:
        s = Lag2ReversionStrategy()
        s.fit(_make_train_data())
        # Small change 2 days ago — no signal
        assert s.act(_make_state([0.1, 0.2])) is None


class TestLag2ReversionEdgeCases:
    def test_insufficient_history_skip(self) -> None:
        """With fewer than 2 days of history, strategy should skip."""
        s = Lag2ReversionStrategy()
        s.fit(_make_train_data())
        # Only 1 day of history (need 2 to get lag-2)
        assert s.act(_make_state([1.0])) is None

    def test_empty_history_skip(self) -> None:
        s = Lag2ReversionStrategy()
        s.fit(_make_train_data())
        state = BacktestState(
            delivery_date=date(2024, 1, 15),
            last_settlement_price=50.0,
            features=pd.Series({"load_forecast_mw_mean": 40_000.0}),
            history=pd.DataFrame(),
        )
        assert s.act(state) is None


class TestLag2ReversionThreshold:
    """Threshold is based on training data volatility."""

    def test_threshold_is_median_absolute_change(self) -> None:
        s = Lag2ReversionStrategy()
        train = pd.DataFrame({"price_change_eur_mwh": [10.0, -10.0, 10.0, -10.0]})
        s.fit(train)
        # median of |changes| = median([10, 10, 10, 10]) = 10
        assert s.skip_buffer == 10.0


class TestLag2ReversionNotFitted:
    def test_skip_buffer_initialized_to_zero(self) -> None:
        s = Lag2ReversionStrategy()
        assert s.skip_buffer == 0.0
