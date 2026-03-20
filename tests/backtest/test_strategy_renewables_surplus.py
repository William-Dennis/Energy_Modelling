"""Tests for RenewablesSurplusStrategy."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.renewables_surplus import RenewablesSurplusStrategy

_OFFSHORE_COL = "forecast_wind_offshore_mw_mean"
_ONSHORE_COL = "forecast_wind_onshore_mw_mean"
_SOLAR_COL = "forecast_solar_mw_mean"


def _make_train(
    offshore_vals: list[float] | None = None,
    onshore_vals: list[float] | None = None,
    solar_vals: list[float] | None = None,
) -> pd.DataFrame:
    n = 10
    offshore = offshore_vals or [1000.0 * (i + 1) for i in range(n)]
    onshore = onshore_vals or [2000.0 * (i + 1) for i in range(n)]
    solar = solar_vals or [500.0 * (i + 1) for i in range(n)]
    return pd.DataFrame({_OFFSHORE_COL: offshore, _ONSHORE_COL: onshore, _SOLAR_COL: solar})


def _make_state(
    offshore: float,
    onshore: float,
    solar: float,
    last_price: float = 50.0,
) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 7, 15),
        last_settlement_price=last_price,
        features=pd.Series(
            {
                _OFFSHORE_COL: offshore,
                _ONSHORE_COL: onshore,
                _SOLAR_COL: solar,
            }
        ),
        history=pd.DataFrame(),
    )


class TestRenewablesSurplusInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(RenewablesSurplusStrategy, BacktestStrategy)

    def test_fit_sets_p20_and_p80(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        assert s._p20 is not None
        assert s._p80 is not None

    def test_p20_below_p80(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        assert s._p20 < s._p80

    def test_reset_preserves_thresholds(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        p20 = s._p20
        p80 = s._p80
        s.reset()
        assert s._p20 == p20
        assert s._p80 == p80

    def test_raises_before_fit(self) -> None:
        s = RenewablesSurplusStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state(offshore=1000.0, onshore=2000.0, solar=500.0))


class TestRenewablesSurplusThresholds:
    def test_p80_from_training_combined(self) -> None:
        train = _make_train()
        combined = train[_OFFSHORE_COL] + train[_ONSHORE_COL] + train[_SOLAR_COL]
        s = RenewablesSurplusStrategy()
        s.fit(train)
        assert abs(s._p80 - float(np.percentile(combined, 80))) < 1e-6

    def test_p20_from_training_combined(self) -> None:
        train = _make_train()
        combined = train[_OFFSHORE_COL] + train[_ONSHORE_COL] + train[_SOLAR_COL]
        s = RenewablesSurplusStrategy()
        s.fit(train)
        assert abs(s._p20 - float(np.percentile(combined, 20))) < 1e-6


class TestRenewablesSurplusSignal:
    def test_renewables_flood_short(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        # Very high combined > P80 -> short
        assert s.act(_make_state(offshore=10000.0, onshore=20000.0, solar=5000.0)) == -1

    def test_renewables_drought_long(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        # Very low combined < P20 -> long
        assert s.act(_make_state(offshore=100.0, onshore=200.0, solar=50.0)) == 1

    def test_moderate_renewables_skip(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        # combined in (P20, P80) -> skip
        p20 = s._p20
        p80 = s._p80
        mid = (p20 + p80) / 2
        # Split evenly across three components
        each = mid / 3
        result = s.act(_make_state(offshore=each, onshore=each, solar=each))
        assert result is None

    def test_just_above_p80_short(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        p80 = s._p80
        each = (p80 + 1.0) / 3
        assert s.act(_make_state(offshore=each, onshore=each, solar=each)) == -1

    def test_just_below_p20_long(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        p20 = s._p20
        each = (p20 - 1.0) / 3
        assert s.act(_make_state(offshore=each, onshore=each, solar=each)) == 1

    def test_at_p80_skip(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        p80 = s._p80
        each = p80 / 3
        # exactly at P80 -> not strictly above -> skip
        result = s.act(_make_state(offshore=each, onshore=each, solar=each))
        assert result is None

    def test_at_p20_skip(self) -> None:
        s = RenewablesSurplusStrategy()
        s.fit(_make_train())
        p20 = s._p20
        each = p20 / 3
        # exactly at P20 -> not strictly below -> skip
        result = s.act(_make_state(offshore=each, onshore=each, solar=each))
        assert result is None
