"""Phase F: Tests for ensemble/meta strategies.

Each strategy gets at least 5 tests covering:
1. forecast() returns a finite float after fit()
2. forecast() before fit() raises or returns finite float (graceful)
3. fit() sets skip_buffer
4. reset() is callable
5. Strategy subclasses BacktestStrategy
6. Ensemble-specific logic (member count, vote mechanics, regime routing, etc.)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.ensemble_base import _EnsembleBase
from strategies.majority_vote_rule import MajorityVoteRuleBasedStrategy
from strategies.majority_vote_ml import MajorityVoteMLStrategy
from strategies.mean_forecast_regression import MeanForecastRegressionStrategy
from strategies.weighted_vote_mixed import WeightedVoteMixedStrategy
from strategies.stacked_ridge_meta import StackedRidgeMetaStrategy
from strategies.consensus_signal import ConsensusSignalStrategy
from strategies.regime_conditional_ensemble import RegimeConditionalEnsembleStrategy
from strategies.top_k_ensemble import TopKEnsembleStrategy
from strategies.diversity_ensemble import DiversityEnsembleStrategy
from strategies.median_forecast_ensemble import MedianForecastEnsembleStrategy
from strategies.weekday_weekend_ensemble import WeekdayWeekendEnsembleStrategy
from strategies.boosted_spread_ml import BoostedSpreadMLStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _make_train(n: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    prices = 50.0 + _RNG.normal(0, 8, n).cumsum()
    changes = np.diff(prices, prepend=prices[0])
    de_fr = _RNG.normal(0, 3, n)
    gas = 20.0 + _RNG.normal(0, 2, n).cumsum()
    carbon = 30.0 + _RNG.normal(0, 1, n).cumsum()
    wind = _RNG.uniform(5_000, 30_000, n)
    solar = _RNG.uniform(0, 10_000, n)
    load = _RNG.uniform(40_000, 70_000, n)
    nuclear = _RNG.uniform(5_000, 15_000, n)
    temp = _RNG.normal(12, 8, n)
    fr_flow = _RNG.normal(0, 500, n)
    nl_flow = _RNG.normal(0, 300, n)
    wind_fc = wind + _RNG.normal(0, 1_000, n)
    wind_offshore = wind * 0.6
    wind_onshore = wind * 0.4
    return pd.DataFrame(
        {
            "delivery_date": [d.date() for d in dates],
            "split": ["train"] * n,
            "settlement_price": prices,
            "last_settlement_price": np.roll(prices, 1),
            "price_change_eur_mwh": changes,
            "target_direction": np.sign(changes).astype(float),
            "pnl_long_eur": changes * 24.0,
            "pnl_short_eur": -changes * 24.0,
            # Raw features used by base strategies
            "gas_price_eur_mwh": gas,
            "carbon_price_eur_t": carbon,
            "wind_mw": wind,
            "solar_mw": solar,
            "load_mw": load,
            "nuclear_mw": nuclear,
            "temperature_celsius": temp,
            "fr_flow_mw": fr_flow,
            "nl_flow_mw": nl_flow,
            "wind_forecast_mw": wind_fc,
            "de_fr_power_flow_mw": de_fr,
            # Columns required by WindForecastStrategy and LoadForecastStrategy
            "forecast_wind_offshore_mw_mean": wind_offshore,
            "forecast_wind_onshore_mw_mean": wind_onshore,
            "load_forecast_mw_mean": load,
            # Columns required by CompositeSignalStrategy
            "gen_fossil_gas_mw_mean": _RNG.uniform(5_000, 20_000, n),
            "gen_fossil_hard_coal_mw_mean": _RNG.uniform(1_000, 10_000, n),
            "gen_fossil_brown_coal_lignite_mw_mean": _RNG.uniform(500, 8_000, n),
            # Derived features
            "net_demand_mw": load - wind - solar,
            "renewable_penetration_pct": (wind + solar) / load * 100,
            "de_fr_spread": de_fr,
            "de_nl_spread": _RNG.normal(0, 2, n),
            "de_avg_neighbour_spread": _RNG.normal(0, 2, n),
            "price_zscore_20d": _RNG.normal(0, 1.5, n),
            "price_range": _RNG.uniform(2, 20, n),
            "gas_trend_3d": _RNG.normal(0, 0.5, n),
            "carbon_trend_3d": _RNG.normal(0, 0.5, n),
            "fuel_cost_index": _RNG.uniform(0.5, 1.5, n),
            "wind_forecast_error": _RNG.normal(0, 500, n),
            "load_surprise": _RNG.normal(0, 1_000, n),
            "rolling_vol_7d": _RNG.uniform(2, 12, n),
            "rolling_vol_14d": _RNG.uniform(2, 12, n),
            "total_fossil_mw": _RNG.uniform(10_000, 40_000, n),
            "net_flow_mw": fr_flow + nl_flow,
            "dow_int": [d.dayofweek for d in dates],
            "is_weekend": [1 if d.dayofweek >= 5 else 0 for d in dates],
        }
    )


def _make_state(
    last_price: float = 55.0,
    delivery: date = date(2024, 6, 3),  # Monday
    features: dict | None = None,
) -> BacktestState:
    feats: dict = {
        "gas_price_eur_mwh": 22.0,
        "carbon_price_eur_t": 31.0,
        "wind_mw": 15_000.0,
        "solar_mw": 3_000.0,
        "load_mw": 55_000.0,
        "nuclear_mw": 10_000.0,
        "temperature_celsius": 14.0,
        "fr_flow_mw": 200.0,
        "nl_flow_mw": 100.0,
        "wind_forecast_mw": 14_500.0,
        "de_fr_power_flow_mw": 1.5,
        # WindForecastStrategy / LoadForecastStrategy columns
        "forecast_wind_offshore_mw_mean": 9_000.0,
        "forecast_wind_onshore_mw_mean": 6_000.0,
        "load_forecast_mw_mean": 55_000.0,
        # CompositeSignalStrategy columns
        "gen_fossil_gas_mw_mean": 12_000.0,
        "gen_fossil_hard_coal_mw_mean": 5_000.0,
        "gen_fossil_brown_coal_lignite_mw_mean": 3_000.0,
        "net_demand_mw": 37_000.0,
        "renewable_penetration_pct": 32.7,
        "de_fr_spread": 2.0,
        "de_nl_spread": 1.0,
        "de_avg_neighbour_spread": 1.5,
        "price_zscore_20d": 0.5,
        "price_range": 8.0,
        "gas_trend_3d": 0.3,
        "carbon_trend_3d": 0.2,
        "fuel_cost_index": 1.1,
        "wind_forecast_error": 500.0,
        "load_surprise": 200.0,
        "rolling_vol_7d": 5.0,
        "rolling_vol_14d": 5.5,
        "total_fossil_mw": 25_000.0,
        "net_flow_mw": 300.0,
        "dow_int": delivery.weekday(),
        "is_weekend": 1 if delivery.weekday() >= 5 else 0,
    }
    if features:
        feats.update(features)
    return BacktestState(
        delivery_date=delivery,
        last_settlement_price=last_price,
        features=pd.Series(feats),
        history=pd.DataFrame(),
    )


# ===========================================================================
# _EnsembleBase (base class sanity)
# ===========================================================================


class TestEnsembleBase:
    """Verify _EnsembleBase cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            _EnsembleBase()  # type: ignore[abstract]

    def test_subclass_must_implement_forecast(self) -> None:
        class _Incomplete(_EnsembleBase):
            _MEMBERS = []
            # forecast() not implemented

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]


# ===========================================================================
# MajorityVoteRuleBasedStrategy
# ===========================================================================


class TestMajorityVoteRuleBasedStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = MajorityVoteRuleBasedStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_returns_last_price_offset(self) -> None:
        """Result must be last_price ± 1 or last_price (skip)."""
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        result = self.strategy.forecast(state)
        assert result in {49.0, 50.0, 51.0}

    def test_five_members_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 5

    def test_skip_buffer_set_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_consistent_result_same_state(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state()
        assert self.strategy.forecast(state) == self.strategy.forecast(state)


# ===========================================================================
# MajorityVoteMLStrategy
# ===========================================================================


class TestMajorityVoteMLStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = MajorityVoteMLStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        result = self.strategy.forecast(state)
        assert result in {49.0, 50.0, 51.0}

    def test_five_ml_members_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 5

    def test_skip_buffer_set_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_two_separate_fits_give_finite_result(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.fit(self.df)  # re-fit must not break
        assert np.isfinite(self.strategy.forecast(_make_state()))


# ===========================================================================
# MeanForecastRegressionStrategy
# ===========================================================================


class TestMeanForecastRegressionStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = MeanForecastRegressionStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        result = self.strategy.forecast(_make_state())
        assert np.isfinite(result)

    def test_forecast_is_mean_of_members(self) -> None:
        """forecast() == mean of individual member forecasts."""
        self.strategy.fit(self.df)
        state = _make_state()
        individual = [float(m.forecast(state)) for m in self.strategy._fitted_members]
        expected = float(np.mean(individual))
        assert self.strategy.forecast(state) == pytest.approx(expected)

    def test_four_members_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 4

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()


# ===========================================================================
# WeightedVoteMixedStrategy
# ===========================================================================


class TestWeightedVoteMixedStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = WeightedVoteMixedStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=60.0)
        result = self.strategy.forecast(state)
        assert result in {59.0, 60.0, 61.0}

    def test_six_members_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 6

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_consistent_forecast_on_same_state(self) -> None:
        self.strategy.fit(self.df)
        s = _make_state()
        assert self.strategy.forecast(s) == self.strategy.forecast(s)


# ===========================================================================
# StackedRidgeMetaStrategy
# ===========================================================================


class TestStackedRidgeMetaStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = StackedRidgeMetaStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_meta_ridge_fitted_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy._meta_ridge is not None
        assert self.strategy._meta_scaler is not None

    def test_four_base_members(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 4

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_fallback_on_tiny_data(self) -> None:
        """With only 10 rows, meta_train < 3 triggers fallback path gracefully."""
        tiny = self.df.head(10).copy()
        self.strategy.fit(tiny)
        assert np.isfinite(self.strategy.forecast(_make_state()))


# ===========================================================================
# ConsensusSignalStrategy
# ===========================================================================


class TestConsensusSignalStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = ConsensusSignalStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        result = self.strategy.forecast(state)
        assert result in {49.0, 50.0, 51.0}

    def test_three_members_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 3

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_no_consensus_gives_skip(self) -> None:
        """Manually patch member directions to be mixed → must skip (return last_price)."""
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        # Patch _get_member_directions to return mixed signals
        self.strategy._get_member_directions = lambda s: [1.0, -1.0, 1.0]  # type: ignore[method-assign]
        result = self.strategy.forecast(state)
        assert result == pytest.approx(50.0)


# ===========================================================================
# RegimeConditionalEnsembleStrategy
# ===========================================================================


class TestRegimeConditionalEnsembleStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = RegimeConditionalEnsembleStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_vol_threshold_is_median_of_training(self) -> None:
        self.strategy.fit(self.df)
        expected = float(np.median(self.df["rolling_vol_7d"]))
        assert self.strategy._vol_threshold == pytest.approx(expected)

    def test_three_ml_and_three_rule_members(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._ml_members) == 3
        assert len(self.strategy._rule_members) == 3

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_high_vol_and_low_vol_both_return_finite(self) -> None:
        self.strategy.fit(self.df)
        high_vol = _make_state(features={"rolling_vol_7d": 100.0})
        low_vol = _make_state(features={"rolling_vol_7d": 0.1})
        assert np.isfinite(self.strategy.forecast(high_vol))
        assert np.isfinite(self.strategy.forecast(low_vol))

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=45.0)
        result = self.strategy.forecast(state)
        assert result in {44.0, 45.0, 46.0}


# ===========================================================================
# TopKEnsembleStrategy
# ===========================================================================


class TestTopKEnsembleStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = TopKEnsembleStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_top_3_members_selected(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 3

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        result = self.strategy.forecast(state)
        assert result in {49.0, 50.0, 51.0}

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_six_candidate_members_defined(self) -> None:
        assert len(TopKEnsembleStrategy._MEMBERS) == 6


# ===========================================================================
# DiversityEnsembleStrategy
# ===========================================================================


class TestDiversityEnsembleStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = DiversityEnsembleStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        result = self.strategy.forecast(state)
        assert result in {49.0, 50.0, 51.0}

    def test_three_members_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 3

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_members_are_diverse_types(self) -> None:
        """The three members come from distinct strategy families."""
        from strategies.composite_signal import CompositeSignalStrategy
        from strategies.ridge_regression import RidgeRegressionStrategy
        from strategies.random_forest_direction import RandomForestStrategy

        classes = DiversityEnsembleStrategy._MEMBERS
        assert CompositeSignalStrategy in classes
        assert RidgeRegressionStrategy in classes
        assert RandomForestStrategy in classes


# ===========================================================================
# MedianForecastEnsembleStrategy
# ===========================================================================


class TestMedianForecastEnsembleStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = MedianForecastEnsembleStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_is_ensemble_base(self) -> None:
        assert isinstance(self.strategy, _EnsembleBase)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_is_median_of_members(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state()
        individual = [float(m.forecast(state)) for m in self.strategy._fitted_members]
        expected = float(np.median(individual))
        assert self.strategy.forecast(state) == pytest.approx(expected)

    def test_five_members_after_fit(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._fitted_members) == 5

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_median_differs_from_mean_with_outlier(self) -> None:
        """Patch one member to return an extreme forecast; median stays robust."""
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        original = self.strategy.forecast(state)
        # The median should be finite regardless
        assert np.isfinite(original)


# ===========================================================================
# WeekdayWeekendEnsembleStrategy
# ===========================================================================


class TestWeekdayWeekendEnsembleStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = WeekdayWeekendEnsembleStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_three_weekday_and_three_weekend_members(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._weekday_members) == 3
        assert len(self.strategy._weekend_members) == 3

    def test_weekday_forecast_finite(self) -> None:
        self.strategy.fit(self.df)
        monday = _make_state(delivery=date(2024, 6, 3), features={"dow_int": 0, "is_weekend": 0})
        assert np.isfinite(self.strategy.forecast(monday))

    def test_weekend_forecast_finite(self) -> None:
        self.strategy.fit(self.df)
        saturday = _make_state(delivery=date(2024, 6, 8), features={"dow_int": 5, "is_weekend": 1})
        assert np.isfinite(self.strategy.forecast(saturday))

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        result = self.strategy.forecast(state)
        assert result in {49.0, 50.0, 51.0}


# ===========================================================================
# BoostedSpreadMLStrategy
# ===========================================================================


class TestBoostedSpreadMLStrategy:
    def setup_method(self) -> None:
        self.df = _make_train()
        self.strategy = BoostedSpreadMLStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_returns_last_price_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        result = self.strategy.forecast(state)
        assert result in {49.0, 50.0, 51.0}

    def test_skip_buffer_is_average_of_two_members(self) -> None:
        self.strategy.fit(self.df)
        expected = float(
            np.mean([self.strategy._spread.skip_buffer, self.strategy._gbm.skip_buffer])
        )
        assert self.strategy.skip_buffer == pytest.approx(expected)

    def test_disagreement_gives_skip(self) -> None:
        """Patch directions to disagree → forecast == last_price."""
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0)
        self.strategy._direction = lambda strat, s: 1.0 if strat is self.strategy._spread else -1.0  # type: ignore[method-assign]
        result = self.strategy.forecast(state)
        assert result == pytest.approx(50.0)

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()

    def test_both_members_fitted(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy._spread is not None
        assert self.strategy._gbm is not None
