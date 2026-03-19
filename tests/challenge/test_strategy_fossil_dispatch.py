"""Tests for FossilDispatchStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy
from strategies.fossil_dispatch import FossilDispatchStrategy

_GAS = "gen_fossil_gas_mw_mean"
_COAL = "gen_fossil_hard_coal_mw_mean"
_LIGNITE = "gen_fossil_brown_coal_lignite_mw_mean"


def _make_train_data(
    gas: list[float] | None = None,
    coal: list[float] | None = None,
    lignite: list[float] | None = None,
) -> pd.DataFrame:
    gas = gas or [3000.0, 4000.0, 5000.0, 6000.0, 7000.0]
    coal = coal or [2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    lignite = lignite or [5000.0, 6000.0, 7000.0, 8000.0, 9000.0]
    return pd.DataFrame({_GAS: gas, _COAL: coal, _LIGNITE: lignite})


def _make_state(
    gas: float = 5000.0,
    coal: float = 4000.0,
    lignite: float = 7000.0,
) -> ChallengeState:
    return ChallengeState(
        delivery_date=date(2024, 1, 15),
        last_settlement_price=50.0,
        features=pd.Series(
            {
                _GAS: gas,
                _COAL: coal,
                _LIGNITE: lignite,
                "load_forecast_mw_mean": 40_000.0,
            }
        ),
        history=pd.DataFrame(),
    )


class TestFossilDispatchInterface:
    def test_is_challenge_strategy(self) -> None:
        assert issubclass(FossilDispatchStrategy, ChallengeStrategy)

    def test_fit_computes_threshold(self) -> None:
        s = FossilDispatchStrategy()
        s.fit(_make_train_data())
        assert s._threshold is not None

    def test_reset_preserves_threshold(self) -> None:
        s = FossilDispatchStrategy()
        s.fit(_make_train_data())
        s.reset()
        assert s._threshold is not None


class TestFossilDispatchThreshold:
    def test_threshold_is_median(self) -> None:
        # gas:     [3000, 4000, 5000, 6000, 7000], median=5000
        # coal:    [2000, 3000, 4000, 5000, 6000], median=4000
        # lignite: [5000, 6000, 7000, 8000, 9000], median=7000
        # combined: [10000, 13000, 16000, 19000, 22000], median=16000
        s = FossilDispatchStrategy()
        s.fit(_make_train_data())
        assert s._threshold == 16_000.0


class TestFossilDispatchSignal:
    """High fossil → short, low fossil → long."""

    def test_high_fossil_short(self) -> None:
        s = FossilDispatchStrategy()
        s.fit(_make_train_data())
        # 7000 + 6000 + 9000 = 22000 > 16000 → short
        assert s.act(_make_state(gas=7000.0, coal=6000.0, lignite=9000.0)) == -1

    def test_low_fossil_long(self) -> None:
        s = FossilDispatchStrategy()
        s.fit(_make_train_data())
        # 3000 + 2000 + 5000 = 10000 < 16000 → long
        assert s.act(_make_state(gas=3000.0, coal=2000.0, lignite=5000.0)) == 1

    def test_exactly_at_threshold_short(self) -> None:
        s = FossilDispatchStrategy()
        s.fit(_make_train_data())
        # 5000 + 4000 + 7000 = 16000 == threshold → short
        assert s.act(_make_state(gas=5000.0, coal=4000.0, lignite=7000.0)) == -1

    def test_just_below_threshold_long(self) -> None:
        s = FossilDispatchStrategy()
        s.fit(_make_train_data())
        # 4999 + 4000 + 7000 = 15999 < 16000 → long
        assert s.act(_make_state(gas=4999.0, coal=4000.0, lignite=7000.0)) == 1


class TestFossilDispatchNotFitted:
    def test_raises_before_fit(self) -> None:
        s = FossilDispatchStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())
