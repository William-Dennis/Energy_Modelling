"""Tests for LoadForecastStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy
from strategies.load_forecast import LoadForecastStrategy


def _make_train_data(load_vals: list[float] | None = None) -> pd.DataFrame:
    vals = load_vals or [30_000.0, 35_000.0, 40_000.0, 45_000.0, 50_000.0]
    return pd.DataFrame({"load_forecast_mw_mean": vals})


def _make_state(load: float = 40_000.0) -> ChallengeState:
    return ChallengeState(
        delivery_date=date(2024, 1, 15),
        last_settlement_price=50.0,
        features=pd.Series({"load_forecast_mw_mean": load}, dtype=float),
        history=pd.DataFrame(),
    )


class TestLoadForecastInterface:
    def test_is_challenge_strategy(self) -> None:
        assert issubclass(LoadForecastStrategy, ChallengeStrategy)

    def test_fit_computes_threshold(self) -> None:
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        assert s._threshold is not None

    def test_reset_clears_threshold(self) -> None:
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        s.reset()
        assert s._threshold is None


class TestLoadForecastThreshold:
    def test_threshold_is_median(self) -> None:
        # [30000, 35000, 40000, 45000, 50000] → median = 40000
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        assert s._threshold == 40_000.0


class TestLoadForecastSignal:
    """High load → long, low load → short."""

    def test_high_load_long(self) -> None:
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        # 50000 > 40000 → long
        assert s.act(_make_state(load=50_000.0)) == 1

    def test_low_load_short(self) -> None:
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        # 30000 < 40000 → short
        assert s.act(_make_state(load=30_000.0)) == -1

    def test_exactly_at_threshold_long(self) -> None:
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        # 40000 == 40000 → long (>= threshold)
        assert s.act(_make_state(load=40_000.0)) == 1

    def test_just_below_threshold_short(self) -> None:
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        # 39999 < 40000 → short
        assert s.act(_make_state(load=39_999.0)) == -1


class TestLoadForecastNotFitted:
    def test_raises_before_fit(self) -> None:
        s = LoadForecastStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())

    def test_raises_after_reset(self) -> None:
        s = LoadForecastStrategy()
        s.fit(_make_train_data())
        s.reset()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())
