"""Tests for NuclearAvailabilityStrategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.nuclear_availability import NuclearAvailabilityStrategy

_NUCLEAR_COL = "gen_nuclear_mw_mean"


def _make_train(nuclear_vals: list[float] | None = None) -> pd.DataFrame:
    # 14 rows so rolling-14 day stats are stable
    vals = nuclear_vals or [8000.0] * 14
    return pd.DataFrame({_NUCLEAR_COL: vals})


def _make_state(nuclear: float, last_price: float = 50.0) -> BacktestState:
    return BacktestState(
        delivery_date=date(2024, 3, 1),
        last_settlement_price=last_price,
        features=pd.Series({_NUCLEAR_COL: nuclear}),
        history=pd.DataFrame(),
    )


class TestNuclearAvailabilityInterface:
    def test_is_backtest_strategy(self) -> None:
        assert issubclass(NuclearAvailabilityStrategy, BacktestStrategy)

    def test_fit_sets_mean_and_std(self) -> None:
        s = NuclearAvailabilityStrategy()
        s.fit(_make_train())
        assert s._mean is not None
        assert s._std is not None

    def test_reset_preserves_fitted_params(self) -> None:
        s = NuclearAvailabilityStrategy()
        s.fit(_make_train())
        mean = s._mean
        std = s._std
        s.reset()
        assert s._mean == mean
        assert s._std == std

    def test_raises_before_fit(self) -> None:
        s = NuclearAvailabilityStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state(nuclear=5000.0))


class TestNuclearAvailabilitySignal:
    def test_large_outage_long(self) -> None:
        s = NuclearAvailabilityStrategy()
        # mean=8000, std=0 -> but we need some std for the signal to fire
        # Use varied data so std > 0
        train = pd.DataFrame(
            {
                _NUCLEAR_COL: [
                    6000.0,
                    8000.0,
                    10000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                ]
            }
        )
        s.fit(train)
        mean = s._mean
        std = s._std
        # nuclear = mean - 2*std -> clear outage -> long
        assert s.act(_make_state(nuclear=mean - 2 * std)) == 1

    def test_large_surplus_short(self) -> None:
        s = NuclearAvailabilityStrategy()
        train = pd.DataFrame(
            {
                _NUCLEAR_COL: [
                    6000.0,
                    8000.0,
                    10000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                ]
            }
        )
        s.fit(train)
        mean = s._mean
        std = s._std
        # nuclear = mean + 2*std -> clear surplus -> short
        assert s.act(_make_state(nuclear=mean + 2 * std)) == -1

    def test_normal_nuclear_skip(self) -> None:
        s = NuclearAvailabilityStrategy()
        train = pd.DataFrame(
            {
                _NUCLEAR_COL: [
                    6000.0,
                    8000.0,
                    10000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                    8000.0,
                ]
            }
        )
        s.fit(train)
        mean = s._mean
        # nuclear exactly at mean -> within ±1std -> skip
        result = s.act(_make_state(nuclear=mean))
        assert result is None

    def test_zero_std_returns_skip(self) -> None:
        s = NuclearAvailabilityStrategy()
        s.fit(_make_train())  # all 8000, std=0
        # std is 0 -> no deviation can exceed threshold -> skip
        result = s.act(_make_state(nuclear=0.0))
        assert result is None

    def test_result_type_when_active(self) -> None:
        s = NuclearAvailabilityStrategy()
        train = pd.DataFrame({_NUCLEAR_COL: [6000.0, 10000.0] * 7})
        s.fit(train)
        mean = s._mean
        std = s._std
        result = s.act(_make_state(nuclear=mean - 2 * std))
        assert result in (1, -1, None)

    def test_mean_from_training_data(self) -> None:
        vals = [
            6000.0,
            7000.0,
            8000.0,
            9000.0,
            10000.0,
            8000.0,
            8000.0,
            8000.0,
            8000.0,
            8000.0,
            8000.0,
            8000.0,
            8000.0,
            8000.0,
        ]
        s = NuclearAvailabilityStrategy()
        s.fit(pd.DataFrame({_NUCLEAR_COL: vals}))
        import numpy as np

        assert abs(s._mean - float(np.mean(vals))) < 1e-6

    def test_structurally_zero_nuclear_skips(self) -> None:
        # Simulate post-April-2023 data where nuclear is always 0
        s = NuclearAvailabilityStrategy()
        s.fit(pd.DataFrame({_NUCLEAR_COL: [0.0] * 14}))
        # std=0, any value -> skip
        assert s.act(_make_state(nuclear=0.0)) is None
