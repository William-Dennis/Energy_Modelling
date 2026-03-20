"""Tests for TemperatureExtremeStrategy."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.temperature_extreme import TemperatureExtremeStrategy

_TEMP_COL = "weather_temperature_2m_degc_mean"


def _make_train(temp_vals: list[float] | None = None) -> pd.DataFrame:
    # 20 values so percentiles are well-defined
    vals = temp_vals or list(range(-5, 15))  # -5 to 14 inclusive
    return pd.DataFrame({_TEMP_COL: vals})


def _make_state(temp: float, last_price: float = 50.0) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 1, 15),
        last_settlement_price=last_price,
        features=pd.Series({_TEMP_COL: temp}),
        history=pd.DataFrame(),
    )


class TestTemperatureExtremeInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(TemperatureExtremeStrategy, BacktestStrategy)

    def test_fit_sets_p10_and_p90(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        assert s._p10 is not None
        assert s._p90 is not None

    def test_p10_below_p90(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        assert s._p10 < s._p90

    def test_reset_preserves_thresholds(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        p10 = s._p10
        p90 = s._p90
        s.reset()
        assert s._p10 == p10
        assert s._p90 == p90

    def test_raises_before_fit(self) -> None:
        s = TemperatureExtremeStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state(temp=5.0))


class TestTemperatureExtremeSignal:
    def test_extreme_cold_long(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        # Far below P10 -> cold extreme -> long
        assert s.act(_make_state(temp=-20.0)) == 1

    def test_extreme_heat_long(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        # Far above P90 -> hot extreme -> long
        assert s.act(_make_state(temp=40.0)) == 1

    def test_moderate_temp_short(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        # Middle of range -> moderate -> short
        assert s.act(_make_state(temp=5.0)) == -1

    def test_just_below_p10_long(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        p10 = s._p10
        assert s.act(_make_state(temp=p10 - 0.01)) == 1

    def test_just_above_p90_long(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        p90 = s._p90
        assert s.act(_make_state(temp=p90 + 0.01)) == 1

    def test_at_p10_short(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        # exactly at p10 = moderate
        assert s.act(_make_state(temp=s._p10)) == -1

    def test_at_p90_short(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        # exactly at p90 = moderate
        assert s.act(_make_state(temp=s._p90)) == -1

    def test_result_is_int(self) -> None:
        s = TemperatureExtremeStrategy()
        s.fit(_make_train())
        result = s.act(_make_state(temp=5.0))
        assert isinstance(result, int)

    def test_p10_p90_from_numpy_percentile(self) -> None:
        vals = list(range(-5, 15))
        s = TemperatureExtremeStrategy()
        s.fit(pd.DataFrame({_TEMP_COL: vals}))
        assert abs(s._p10 - np.percentile(vals, 10)) < 1e-9
        assert abs(s._p90 - np.percentile(vals, 90)) < 1e-9

    def test_narrow_range_still_classifies(self) -> None:
        # All same temp -> p10 == p90, moderate range collapses
        s = TemperatureExtremeStrategy()
        s.fit(pd.DataFrame({_TEMP_COL: [10.0] * 20}))
        # Any temp equals p10 and p90, so moderate -> short
        assert s.act(_make_state(temp=10.0)) == -1
