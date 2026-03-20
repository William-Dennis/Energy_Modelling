"""Tests for SolarForecastStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.solar_forecast import SolarForecastStrategy

_SOLAR_COL = "forecast_solar_mw_mean"


def _make_train(solar_vals: list[float] | None = None) -> pd.DataFrame:
    vals = solar_vals or [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    return pd.DataFrame({_SOLAR_COL: vals})


def _make_state(solar: float = 3000.0, last_price: float = 50.0) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 6, 15),
        last_settlement_price=last_price,
        features=pd.Series({_SOLAR_COL: solar}),
        history=pd.DataFrame(),
    )


class TestSolarForecastInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(SolarForecastStrategy, BacktestStrategy)

    def test_fit_sets_threshold(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        assert s._threshold is not None

    def test_threshold_is_median(self) -> None:
        # [1000, 2000, 3000, 4000, 5000] -> median = 3000
        s = SolarForecastStrategy()
        s.fit(_make_train())
        assert s._threshold == 3000.0

    def test_reset_preserves_threshold(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        s.reset()
        assert s._threshold == 3000.0

    def test_raises_before_fit(self) -> None:
        s = SolarForecastStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())


class TestSolarForecastSignal:
    def test_high_solar_short(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        # 5000 >= 3000 -> short
        assert s.act(_make_state(solar=5000.0)) == -1

    def test_low_solar_long(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        # 1000 < 3000 -> long
        assert s.act(_make_state(solar=1000.0)) == 1

    def test_exactly_at_threshold_short(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        # 3000 == 3000 -> short (>= threshold)
        assert s.act(_make_state(solar=3000.0)) == -1

    def test_just_below_threshold_long(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        # 2999 < 3000 -> long
        assert s.act(_make_state(solar=2999.0)) == 1

    def test_zero_solar_long(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        # 0 < 3000 -> long (winter night)
        assert s.act(_make_state(solar=0.0)) == 1

    def test_very_high_solar_short(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        assert s.act(_make_state(solar=20_000.0)) == -1

    def test_act_returns_int(self) -> None:
        s = SolarForecastStrategy()
        s.fit(_make_train())
        result = s.act(_make_state(solar=1000.0))
        assert isinstance(result, int)

    def test_uniform_training_threshold_is_that_value(self) -> None:
        s = SolarForecastStrategy()
        s.fit(pd.DataFrame({_SOLAR_COL: [5000.0] * 10}))
        assert s._threshold == 5000.0

    def test_single_row_training(self) -> None:
        s = SolarForecastStrategy()
        s.fit(pd.DataFrame({_SOLAR_COL: [2500.0]}))
        assert s._threshold == 2500.0
        assert s.act(_make_state(solar=3000.0)) == -1
        assert s.act(_make_state(solar=2000.0)) == 1

    def test_constant_training_high_solar_short(self) -> None:
        s = SolarForecastStrategy()
        s.fit(pd.DataFrame({_SOLAR_COL: [4000.0, 4000.0, 4000.0]}))
        assert s.act(_make_state(solar=5000.0)) == -1
