"""Tests for PerfectForesightStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.challenge.types import ChallengeState
from strategies.perfect_foresight import PerfectForesightStrategy


@pytest.fixture()
def lookup() -> dict[date, float]:
    return {
        date(2024, 1, 1): 100.0,
        date(2024, 1, 2): 90.0,
        date(2024, 1, 3): 95.0,
    }


@pytest.fixture()
def strategy(lookup: dict[date, float]) -> PerfectForesightStrategy:
    return PerfectForesightStrategy(settlement_lookup=lookup)


def _make_state(
    delivery_date: date,
    last_settlement: float,
) -> ChallengeState:
    return ChallengeState(
        delivery_date=delivery_date,
        last_settlement_price=last_settlement,
        features=pd.Series(dtype=float),
        history=pd.DataFrame(),
    )


class TestPerfectForesightDirection:
    def test_long_when_price_goes_up(self, strategy: PerfectForesightStrategy) -> None:
        # Real = 100, last = 90 → up → +1
        state = _make_state(date(2024, 1, 1), 90.0)
        assert strategy.act(state) == 1

    def test_short_when_price_goes_down(self, strategy: PerfectForesightStrategy) -> None:
        # Real = 90, last = 100 → down → -1
        state = _make_state(date(2024, 1, 2), 100.0)
        assert strategy.act(state) == -1

    def test_long_when_equal(self, strategy: PerfectForesightStrategy) -> None:
        # Real = 95, last = 95 → equal → -1 (not strictly greater)
        state = _make_state(date(2024, 1, 3), 95.0)
        assert strategy.act(state) == -1

    def test_default_long_for_unknown_date(self, strategy: PerfectForesightStrategy) -> None:
        state = _make_state(date(2024, 12, 31), 90.0)
        assert strategy.act(state) == 1


class TestPerfectForesightInterface:
    def test_reset_is_noop(self, strategy: PerfectForesightStrategy) -> None:
        strategy.reset()
        state = _make_state(date(2024, 1, 1), 90.0)
        assert strategy.act(state) == 1

    def test_fit_is_noop(self, strategy: PerfectForesightStrategy) -> None:
        strategy.fit(pd.DataFrame())
        state = _make_state(date(2024, 1, 1), 90.0)
        assert strategy.act(state) == 1

    def test_always_correct_pnl(self, lookup: dict[date, float]) -> None:
        """Every trade should have non-negative PnL."""
        strategy = PerfectForesightStrategy(settlement_lookup=lookup)
        last_prices = {
            date(2024, 1, 1): 90.0,
            date(2024, 1, 2): 100.0,
            date(2024, 1, 3): 90.0,
        }
        for d, real in lookup.items():
            state = _make_state(d, last_prices[d])
            direction = strategy.act(state)
            pnl = direction * (real - last_prices[d]) * 24
            assert pnl >= 0, f"PnL negative on {d}: {pnl}"
