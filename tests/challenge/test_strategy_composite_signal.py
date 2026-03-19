"""Tests for CompositeSignalStrategy."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy
from strategies.composite_signal import CompositeSignalStrategy


def _make_train_data(n: int = 100) -> pd.DataFrame:
    """Build training data with realistic feature distributions."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "load_forecast_mw_mean": rng.normal(40_000, 5_000, n),
            "forecast_wind_offshore_mw_mean": rng.normal(3_000, 1_500, n),
            "forecast_wind_onshore_mw_mean": rng.normal(6_000, 2_000, n),
            "gen_fossil_gas_mw_mean": rng.normal(5_000, 2_000, n),
            "gen_fossil_hard_coal_mw_mean": rng.normal(4_000, 1_500, n),
            "gen_fossil_brown_coal_lignite_mw_mean": rng.normal(7_000, 2_000, n),
        }
    )


def _make_state(**overrides: float) -> ChallengeState:
    """Build a ChallengeState with feature values.

    Defaults to values near training means (neutral).
    """
    features = {
        "load_forecast_mw_mean": 40_000.0,
        "forecast_wind_offshore_mw_mean": 3_000.0,
        "forecast_wind_onshore_mw_mean": 6_000.0,
        "gen_fossil_gas_mw_mean": 5_000.0,
        "gen_fossil_hard_coal_mw_mean": 4_000.0,
        "gen_fossil_brown_coal_lignite_mw_mean": 7_000.0,
    }
    features.update(overrides)
    return ChallengeState(
        delivery_date=date(2024, 1, 15),
        last_settlement_price=50.0,
        features=pd.Series(features),
        history=pd.DataFrame(),
    )


class TestCompositeSignalInterface:
    def test_is_challenge_strategy(self) -> None:
        assert issubclass(CompositeSignalStrategy, ChallengeStrategy)

    def test_fit_stores_params(self) -> None:
        s = CompositeSignalStrategy()
        s.fit(_make_train_data())
        assert s._means is not None
        assert s._stds is not None

    def test_reset_clears_params(self) -> None:
        s = CompositeSignalStrategy()
        s.fit(_make_train_data())
        s.reset()
        assert s._means is None
        assert s._stds is None


class TestCompositeSignalDirection:
    """Verify the composite signal responds to feature extremes correctly."""

    def test_high_load_low_wind_low_fossil_long(self) -> None:
        """Conditions favoring price up: high load, low wind, low fossil."""
        s = CompositeSignalStrategy()
        s.fit(_make_train_data())
        result = s.act(
            _make_state(
                load_forecast_mw_mean=55_000.0,  # very high → positive weight
                forecast_wind_offshore_mw_mean=500.0,  # very low → negative weight, low value
                forecast_wind_onshore_mw_mean=1_000.0,  # very low
                gen_fossil_gas_mw_mean=1_000.0,  # very low
                gen_fossil_hard_coal_mw_mean=1_000.0,
                gen_fossil_brown_coal_lignite_mw_mean=2_000.0,
            )
        )
        assert result == 1

    def test_low_load_high_wind_high_fossil_short(self) -> None:
        """Conditions favoring price down: low load, high wind, high fossil."""
        s = CompositeSignalStrategy()
        s.fit(_make_train_data())
        result = s.act(
            _make_state(
                load_forecast_mw_mean=25_000.0,
                forecast_wind_offshore_mw_mean=8_000.0,
                forecast_wind_onshore_mw_mean=12_000.0,
                gen_fossil_gas_mw_mean=10_000.0,
                gen_fossil_hard_coal_mw_mean=8_000.0,
                gen_fossil_brown_coal_lignite_mw_mean=12_000.0,
            )
        )
        assert result == -1


class TestCompositeSignalWeights:
    """Weights should match the signs from EDA correlation analysis."""

    def test_load_weight_positive(self) -> None:
        s = CompositeSignalStrategy()
        s.fit(_make_train_data())
        load_idx = s._feature_names.index("load_forecast_mw_mean")
        assert s._weights[load_idx] > 0

    def test_wind_offshore_weight_negative(self) -> None:
        s = CompositeSignalStrategy()
        s.fit(_make_train_data())
        idx = s._feature_names.index("forecast_wind_offshore_mw_mean")
        assert s._weights[idx] < 0

    def test_fossil_gas_weight_negative(self) -> None:
        s = CompositeSignalStrategy()
        s.fit(_make_train_data())
        idx = s._feature_names.index("gen_fossil_gas_mw_mean")
        assert s._weights[idx] < 0


class TestCompositeSignalNotFitted:
    def test_raises_before_fit(self) -> None:
        s = CompositeSignalStrategy()
        with pytest.raises(RuntimeError, match="fit"):
            s.act(_make_state())
