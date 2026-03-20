"""Tests for CrossBorderSpreadStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.cross_border_spread import CrossBorderSpreadStrategy

_FR_COL = "price_fr_eur_mwh_mean"
_NL_COL = "price_nl_eur_mwh_mean"
_PRICE_COL = "price_mean"


def _make_train(
    de_vals: list[float] | None = None,
    fr_vals: list[float] | None = None,
    nl_vals: list[float] | None = None,
) -> pd.DataFrame:
    de = de_vals or [50.0, 55.0, 60.0, 65.0, 70.0]
    fr = fr_vals or [48.0, 53.0, 58.0, 63.0, 68.0]
    nl = nl_vals or [47.0, 52.0, 57.0, 62.0, 67.0]
    return pd.DataFrame({_PRICE_COL: de, _FR_COL: fr, _NL_COL: nl})


def _make_state(
    de: float = 60.0,
    fr: float = 58.0,
    nl: float = 57.0,
    last_price: float = 60.0,
) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 4, 10),
        last_settlement_price=last_price,
        features=pd.Series({_PRICE_COL: de, _FR_COL: fr, _NL_COL: nl}),
        history=pd.DataFrame(),
    )


class TestCrossBorderSpreadInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(CrossBorderSpreadStrategy, BacktestStrategy)

    def test_fit_sets_median_spread(self) -> None:
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        assert s._median_spread is not None

    def test_reset_preserves_median(self) -> None:
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        med = s._median_spread
        s.reset()
        assert s._median_spread == med

    def test_raises_before_fit(self) -> None:
        s = CrossBorderSpreadStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())


class TestCrossBorderSpreadThreshold:
    def test_median_spread_value(self) -> None:
        # spread[i] = avg(fr[i], nl[i]) - de[i]
        # = avg(48,47)-50, avg(53,52)-55, avg(58,57)-60, avg(63,62)-65, avg(68,67)-70
        # = -2.5, -2.5, -2.5, -2.5, -2.5 -> median = -2.5
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        assert abs(s._median_spread - (-2.5)) < 1e-9


class TestCrossBorderSpreadSignal:
    def test_neighbours_expensive_long(self) -> None:
        # neighbours avg = 80, de = 60, spread = 20 >> median(-2.5) -> long
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        assert s.act(_make_state(de=60.0, fr=80.0, nl=80.0)) == 1

    def test_neighbours_cheap_short(self) -> None:
        # neighbours avg = 40, de = 60, spread = -20 << median(-2.5) -> short
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        assert s.act(_make_state(de=60.0, fr=40.0, nl=40.0)) == -1

    def test_spread_at_median_long(self) -> None:
        # spread == median -> long (>= threshold)
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        # Need spread == -2.5: avg(fr,nl) - de = -2.5, de=60, avg=57.5
        assert s.act(_make_state(de=60.0, fr=57.5, nl=57.5)) == 1

    def test_spread_just_below_median_short(self) -> None:
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        # spread = -3.0 < -2.5 -> short
        assert s.act(_make_state(de=60.0, fr=57.0, nl=57.0)) == -1

    def test_result_is_int(self) -> None:
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        result = s.act(_make_state())
        assert isinstance(result, int)

    def test_very_high_neighbours_long(self) -> None:
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        assert s.act(_make_state(de=50.0, fr=200.0, nl=200.0)) == 1

    def test_very_cheap_neighbours_short(self) -> None:
        s = CrossBorderSpreadStrategy()
        s.fit(_make_train())
        assert s.act(_make_state(de=100.0, fr=10.0, nl=10.0)) == -1
