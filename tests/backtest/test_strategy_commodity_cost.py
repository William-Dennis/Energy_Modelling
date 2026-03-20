"""Tests for CommodityCostStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.commodity_cost import CommodityCostStrategy

_GAS_COL = "gas_price_usd_mean"
_CARBON_COL = "carbon_price_usd_mean"

# CCGT heat rate and emission factor (must match strategy constants)
_GAS_HEAT_RATE = 7.5
_EMISSION_FACTOR = 0.37


def _fuel_index(gas: float, carbon: float) -> float:
    return gas * _GAS_HEAT_RATE + carbon * _EMISSION_FACTOR


def _make_train(
    gas_vals: list[float] | None = None,
    carbon_vals: list[float] | None = None,
) -> pd.DataFrame:
    gas = gas_vals or [30.0, 35.0, 40.0, 45.0, 50.0]
    carbon = carbon_vals or [60.0, 65.0, 70.0, 75.0, 80.0]
    return pd.DataFrame({_GAS_COL: gas, _CARBON_COL: carbon})


def _make_state(gas: float = 40.0, carbon: float = 70.0, last_price: float = 50.0) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 3, 10),
        last_settlement_price=last_price,
        features=pd.Series({_GAS_COL: gas, _CARBON_COL: carbon}),
        history=pd.DataFrame(),
    )


class TestCommodityCostInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(CommodityCostStrategy, BacktestStrategy)

    def test_fit_sets_threshold(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        assert s._threshold is not None

    def test_reset_preserves_threshold(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        s.reset()
        assert s._threshold is not None

    def test_raises_before_fit(self) -> None:
        s = CommodityCostStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())


class TestCommodityCostThreshold:
    def test_threshold_is_median_of_fuel_index(self) -> None:
        # gas:    [30, 35, 40, 45, 50]
        # carbon: [60, 65, 70, 75, 80]
        # index:  [30*7.5+60*0.37, ..., 50*7.5+80*0.37] = [247.2, 286.55, 325.9, 365.25, 404.6]
        # median index = 325.9
        s = CommodityCostStrategy()
        s.fit(_make_train())
        expected = _fuel_index(40.0, 70.0)
        assert abs(s._threshold - expected) < 1e-6


class TestCommodityCostSignal:
    def test_high_fuel_cost_long(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        # gas=50, carbon=80 -> index=404.6 > median(325.9) -> long
        assert s.act(_make_state(gas=50.0, carbon=80.0)) == 1

    def test_low_fuel_cost_short(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        # gas=30, carbon=60 -> index=247.2 < median -> short
        assert s.act(_make_state(gas=30.0, carbon=60.0)) == -1

    def test_exactly_at_threshold_long(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        # gas=40, carbon=70 -> index == median -> long (>= threshold)
        assert s.act(_make_state(gas=40.0, carbon=70.0)) == 1

    def test_just_below_threshold_short(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        # gas=39.9, carbon=70 -> index < median -> short
        assert s.act(_make_state(gas=39.9, carbon=70.0)) == -1

    def test_result_is_int(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        result = s.act(_make_state())
        assert isinstance(result, int)

    def test_both_high_long(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        assert s.act(_make_state(gas=100.0, carbon=200.0)) == 1

    def test_both_low_short(self) -> None:
        s = CommodityCostStrategy()
        s.fit(_make_train())
        assert s.act(_make_state(gas=10.0, carbon=20.0)) == -1

    def test_uniform_data_threshold_is_that_index(self) -> None:
        s = CommodityCostStrategy()
        s.fit(pd.DataFrame({_GAS_COL: [40.0] * 5, _CARBON_COL: [70.0] * 5}))
        expected = _fuel_index(40.0, 70.0)
        assert abs(s._threshold - expected) < 1e-6
