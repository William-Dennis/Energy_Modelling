"""Tests for WindForecastStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.wind_forecast import WindForecastStrategy


def _make_train_data(
    offshore_vals: list[float] | None = None,
    onshore_vals: list[float] | None = None,
) -> pd.DataFrame:
    """Build minimal training DataFrame with wind forecast columns."""
    offshore = offshore_vals or [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    onshore = onshore_vals or [2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    return pd.DataFrame(
        {
            "forecast_wind_offshore_mw_mean": offshore,
            "forecast_wind_onshore_mw_mean": onshore,
        }
    )


def _make_state(
    offshore: float = 3000.0,
    onshore: float = 4000.0,
) -> BacktestState:
    """Build a BacktestState with given wind forecast values."""
    return BacktestState(
        delivery_date=date(2024, 1, 15),
        last_settlement_price=50.0,
        features=pd.Series(
            {
                "forecast_wind_offshore_mw_mean": offshore,
                "forecast_wind_onshore_mw_mean": onshore,
                "load_forecast_mw_mean": 40_000.0,
            }
        ),
        history=pd.DataFrame(),
    )


class TestWindForecastInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(WindForecastStrategy, BacktestStrategy)

    def test_fit_computes_threshold(self) -> None:
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        assert s._threshold is not None

    def test_reset_preserves_threshold(self) -> None:
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        s.reset()
        assert s._threshold is not None


class TestWindForecastThreshold:
    """fit() computes median of combined wind forecast."""

    def test_threshold_is_median(self) -> None:
        # offshore: [1000, 2000, 3000, 4000, 5000], median = 3000
        # onshore:  [2000, 3000, 4000, 5000, 6000], median = 4000
        # combined: [3000, 5000, 7000, 9000, 11000], median = 7000
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        assert s._threshold == 7000.0


class TestWindForecastSignal:
    """Core signal: high wind → short, low wind → long."""

    def test_high_wind_short(self) -> None:
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        # combined = 5000 + 6000 = 11000 > 7000 → short
        assert s.act(_make_state(offshore=5000.0, onshore=6000.0)) == -1

    def test_low_wind_long(self) -> None:
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        # combined = 1000 + 2000 = 3000 < 7000 → long
        assert s.act(_make_state(offshore=1000.0, onshore=2000.0)) == 1

    def test_exactly_at_threshold_short(self) -> None:
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        # combined = 3000 + 4000 = 7000 == threshold → short (>= threshold)
        assert s.act(_make_state(offshore=3000.0, onshore=4000.0)) == -1

    def test_just_below_threshold_long(self) -> None:
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        # combined = 2999 + 4000 = 6999 < 7000 → long
        assert s.act(_make_state(offshore=2999.0, onshore=4000.0)) == 1


class TestWindForecastNotFitted:
    """Strategy raises if act() called before fit()."""

    def test_raises_before_fit(self) -> None:
        s = WindForecastStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())

    def test_works_after_reset(self) -> None:
        s = WindForecastStrategy()
        s.fit(_make_train_data())
        s.reset()
        # Should still work since reset preserves fitted params
        assert s.act(_make_state(offshore=1000.0, onshore=2000.0)) == 1
