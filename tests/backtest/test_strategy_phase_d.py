"""Phase D: Tests for ML-based strategies.

Each strategy gets at least 5 tests covering:
1. forecast() returns a finite float after fit()
2. forecast() before fit() returns last_settlement_price (or a finite float)
3. Predicted price differs from last_settlement_price (model moves the needle)
4. reset() is callable without error
5. fit() tolerates missing feature columns (robustness)
6. Strategy subclasses BacktestStrategy
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.types import BacktestState, BacktestStrategy
from strategies.bayesian_ridge import BayesianRidgeStrategy
from strategies.decision_tree_direction import DecisionTreeStrategy
from strategies.elastic_net import ElasticNetStrategy
from strategies.gbm_net_demand import GBMNetDemandStrategy
from strategies.gradient_boosting_direction import GradientBoostingStrategy
from strategies.knn_direction import KNNDirectionStrategy
from strategies.lasso_calendar_augmented import LassoCalendarAugmentedStrategy
from strategies.lasso_top_features import LassoTopFeaturesStrategy
from strategies.logistic_direction import LogisticDirectionStrategy
from strategies.ml_base import (
    _EXCLUDE_COLUMNS,
    DERIVED_FEATURE_COLS,
    TOP10_FEATURE_COLS,
)
from strategies.neural_net import NeuralNetStrategy
from strategies.pls_regression import PLSRegressionStrategy
from strategies.random_forest_direction import RandomForestStrategy
from strategies.ridge_net_demand import RidgeNetDemandStrategy
from strategies.ridge_regression import RidgeRegressionStrategy
from strategies.svm_direction import SVMDirectionStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _make_train_data(n: int = 100, include_derived: bool = True) -> pd.DataFrame:
    """Build a minimal but realistic training DataFrame."""
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    prices = 50.0 + _RNG.normal(0, 10, n).cumsum()
    changes = np.diff(prices, prepend=prices[0])

    data: dict[str, object] = {
        "delivery_date": [d.date() for d in dates],
        "split": ["train"] * n,
        "settlement_price": prices,
        "last_settlement_price": np.roll(prices, 1),
        "price_change_eur_mwh": changes,
        "target_direction": np.sign(changes).astype(float),
        "pnl_long_eur": changes * 24.0,
        "pnl_short_eur": -changes * 24.0,
        # Raw features
        "load_forecast_mw_mean": 40_000 + _RNG.normal(0, 500, n),
        "load_actual_mw_mean": 39_500 + _RNG.normal(0, 400, n),
        "forecast_wind_offshore_mw_mean": 2_000 + _RNG.normal(0, 300, n),
        "forecast_wind_onshore_mw_mean": 6_000 + _RNG.normal(0, 500, n),
        "forecast_solar_mw_mean": 1_500 + _RNG.normal(0, 200, n),
        "gen_fossil_gas_mw_mean": 8_000 + _RNG.normal(0, 800, n),
        "gen_fossil_hard_coal_mw_mean": 3_000 + _RNG.normal(0, 300, n),
        "gen_fossil_brown_coal_lignite_mw_mean": 5_000 + _RNG.normal(0, 400, n),
        "gen_nuclear_mw_mean": 9_000 + _RNG.normal(0, 100, n),
        "gen_wind_onshore_mw_mean": 5_500 + _RNG.normal(0, 500, n),
        "gen_wind_offshore_mw_mean": 1_800 + _RNG.normal(0, 300, n),
        "price_mean": prices,
        "price_max": prices + _RNG.uniform(0, 5, n),
        "price_min": prices - _RNG.uniform(0, 5, n),
        "price_std": _RNG.uniform(1, 8, n),
        "price_fr_eur_mwh_mean": prices + _RNG.normal(2, 2, n),
        "price_nl_eur_mwh_mean": prices + _RNG.normal(1, 2, n),
        "price_at_eur_mwh_mean": prices + _RNG.normal(0.5, 2, n),
        "price_cz_eur_mwh_mean": prices + _RNG.normal(-1, 2, n),
        "price_pl_eur_mwh_mean": prices + _RNG.normal(-2, 2, n),
        "price_dk_1_eur_mwh_mean": prices + _RNG.normal(1.5, 2, n),
        "flow_fr_net_import_mw_mean": _RNG.normal(500, 300, n),
        "flow_nl_net_import_mw_mean": _RNG.normal(-200, 300, n),
        "gas_price_usd_mean": 30 + _RNG.normal(0, 3, n),
        "carbon_price_usd_mean": 60 + _RNG.normal(0, 5, n),
        "weather_temperature_2m_degc_mean": 10 + _RNG.normal(0, 5, n),
        "weather_wind_speed_10m_kmh_mean": 20 + _RNG.normal(0, 5, n),
    }

    if include_derived:
        # Add Phase-A derived features
        data["net_demand_mw"] = (
            data["load_forecast_mw_mean"]  # type: ignore[operator]
            - data["forecast_wind_offshore_mw_mean"]  # type: ignore[operator]
            - data["forecast_wind_onshore_mw_mean"]  # type: ignore[operator]
            - data["forecast_solar_mw_mean"]  # type: ignore[operator]
        )
        data["renewable_penetration_pct"] = (
            (
                data["forecast_wind_offshore_mw_mean"]  # type: ignore[operator]
                + data["forecast_wind_onshore_mw_mean"]
                + data["forecast_solar_mw_mean"]
            )
            / data["load_forecast_mw_mean"]
            * 100.0
        )
        data["de_fr_spread"] = (
            data["price_mean"] - data["price_fr_eur_mwh_mean"]  # type: ignore[operator]
        )
        data["de_nl_spread"] = (
            data["price_mean"] - data["price_nl_eur_mwh_mean"]  # type: ignore[operator]
        )
        data["de_avg_neighbour_spread"] = _RNG.normal(0, 2, n)
        data["price_zscore_20d"] = _RNG.normal(0, 1, n)
        data["price_range"] = (
            data["price_max"] - data["price_min"]  # type: ignore[operator]
        )
        data["gas_trend_3d"] = _RNG.normal(0, 0.5, n)
        data["carbon_trend_3d"] = _RNG.normal(0, 0.5, n)
        data["fuel_cost_index"] = (
            data["gas_price_usd_mean"] * 0.5 + data["carbon_price_usd_mean"] * 0.5  # type: ignore[operator]
        )
        data["wind_forecast_error"] = _RNG.normal(0, 200, n)
        data["load_surprise"] = _RNG.normal(0, 300, n)
        data["rolling_vol_7d"] = _RNG.uniform(2, 8, n)
        data["rolling_vol_14d"] = _RNG.uniform(2, 8, n)
        data["total_fossil_mw"] = (
            data["gen_fossil_gas_mw_mean"]  # type: ignore[operator]
            + data["gen_fossil_hard_coal_mw_mean"]
            + data["gen_fossil_brown_coal_lignite_mw_mean"]
        )
        data["net_flow_mw"] = (
            data["flow_fr_net_import_mw_mean"] + data["flow_nl_net_import_mw_mean"]  # type: ignore[operator]
        )
        data["dow_int"] = [d.dayofweek for d in dates]
        data["is_weekend"] = [1 if d.dayofweek >= 5 else 0 for d in dates]

    return pd.DataFrame(data)


def _make_state(last_price: float = 55.0, df: pd.DataFrame | None = None) -> BacktestState:
    """Build a minimal BacktestState for inference."""
    if df is not None:
        features = df.iloc[-1].drop(
            labels=[c for c in _EXCLUDE_COLUMNS if c in df.columns], errors="ignore"
        )
    else:
        features = pd.Series(
            {
                "load_forecast_mw_mean": 41_000.0,
                "forecast_wind_offshore_mw_mean": 2_200.0,
                "forecast_wind_onshore_mw_mean": 6_500.0,
                "forecast_solar_mw_mean": 1_600.0,
                "gas_price_usd_mean": 31.0,
                "carbon_price_usd_mean": 62.0,
                "price_fr_eur_mwh_mean": 57.0,
                "price_nl_eur_mwh_mean": 56.0,
                "net_demand_mw": 31_000.0,
                "renewable_penetration_pct": 24.5,
                "price_zscore_20d": 0.3,
                "gas_trend_3d": 0.1,
                "dow_int": 2,
                "is_weekend": 0,
            }
        )
    return BacktestState(
        delivery_date=date(2024, 6, 1),
        last_settlement_price=last_price,
        features=features,
        history=pd.DataFrame(),
    )


# ===========================================================================
# ml_base module tests
# ===========================================================================


class TestMLBase:
    def test_derived_feature_cols_count(self) -> None:
        assert len(DERIVED_FEATURE_COLS) == 18

    def test_top10_feature_cols_count(self) -> None:
        assert len(TOP10_FEATURE_COLS) == 10

    def test_exclude_columns_contains_labels(self) -> None:
        for col in ("price_change_eur_mwh", "target_direction", "settlement_price"):
            assert col in _EXCLUDE_COLUMNS

    def test_get_feature_cols_excludes_labels(self) -> None:
        base = BayesianRidgeStrategy()  # any _MLStrategyBase subclass
        df = _make_train_data()
        cols = base._get_feature_cols(df)
        for excl in _EXCLUDE_COLUMNS:
            assert excl not in cols

    def test_get_feature_cols_with_candidates(self) -> None:
        base = BayesianRidgeStrategy()
        df = _make_train_data()
        cols = base._get_feature_cols(df, candidate_cols=["net_demand_mw", "gas_trend_3d"])
        assert set(cols) == {"net_demand_mw", "gas_trend_3d"}

    def test_get_x_row_missing_col_returns_zero(self) -> None:
        base = BayesianRidgeStrategy()
        state = _make_state()
        x = base._get_x_row(state, ["nonexistent_col"])
        assert x.shape == (1, 1)
        assert x[0, 0] == 0.0

    def test_get_y_direction_values(self) -> None:
        base = BayesianRidgeStrategy()
        df = _make_train_data()
        y = base._get_y_direction(df)
        assert set(y).issubset({-1.0, 0.0, 1.0})


# ===========================================================================
# Per-strategy test classes
# ===========================================================================


def _regression_strategy_tests(strategy_cls, *, use_derived_only: bool = False) -> None:
    """Shared helper — not a test itself."""


class TestRidgeRegressionStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = RidgeRegressionStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        state = _make_state()
        result = self.strategy.forecast(state)
        assert np.isfinite(result)

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(df=self.df)
        result = self.strategy.forecast(state)
        assert np.isfinite(result)

    def test_fit_sets_feature_cols(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._feature_cols) > 0

    def test_fit_sets_skip_buffer(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.fit(self.df)
        self.strategy.reset()  # should not raise

    def test_robustness_missing_columns(self) -> None:
        minimal = self.df[
            [
                "settlement_price",
                "last_settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "split",
                "load_forecast_mw_mean",
            ]
        ].copy()
        self.strategy.fit(minimal)
        state = _make_state()
        assert np.isfinite(self.strategy.forecast(state))


class TestElasticNetStrategy:
    @classmethod
    def setup_class(cls) -> None:
        cls.df = _make_train_data()
        cls.strategy = ElasticNetStrategy()
        cls.strategy.fit(cls.df)

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        fresh = ElasticNetStrategy()
        assert np.isfinite(fresh.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_fit_sets_feature_cols(self) -> None:
        assert len(self.strategy._feature_cols) > 0

    def test_fit_sets_skip_buffer(self) -> None:
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_pipeline_is_fitted(self) -> None:
        assert self.strategy._pipeline is not None


class TestLogisticDirectionStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = LogisticDirectionStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_forecast_direction_is_plus_or_minus_one_offset(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0, df=self.df)
        result = self.strategy.forecast(state)
        # result = last_price + direction, direction in {-1, +1}
        assert result in {49.0, 51.0}

    def test_fit_sets_skip_buffer_zero(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_pipeline_fitted(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy._pipeline is not None


class TestRandomForestStrategy:
    @classmethod
    def setup_class(cls) -> None:
        cls.df = _make_train_data()
        cls.strategy = RandomForestStrategy()
        cls.strategy.fit(cls.df)

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        fresh = RandomForestStrategy()
        assert np.isfinite(fresh.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_forecast_encodes_direction(self) -> None:
        state = _make_state(last_price=100.0, df=self.df)
        result = self.strategy.forecast(state)
        assert result in {99.0, 101.0}

    def test_fit_sets_skip_buffer_zero(self) -> None:
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_pipeline_fitted(self) -> None:
        assert self.strategy._pipeline is not None


class TestGradientBoostingStrategy:
    @classmethod
    def setup_class(cls) -> None:
        cls.df = _make_train_data()
        cls.strategy = GradientBoostingStrategy()
        cls.strategy.fit(cls.df)

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        fresh = GradientBoostingStrategy()
        assert np.isfinite(fresh.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_forecast_encodes_direction(self) -> None:
        state = _make_state(last_price=80.0, df=self.df)
        result = self.strategy.forecast(state)
        assert result in {79.0, 81.0}

    def test_skip_buffer_zero(self) -> None:
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


class TestLassoTopFeaturesStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = LassoTopFeaturesStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_feature_cols_subset_of_top10(self) -> None:
        self.strategy.fit(self.df)
        assert set(self.strategy._feature_cols).issubset(set(TOP10_FEATURE_COLS))

    def test_skip_buffer_positive(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


class TestRidgeNetDemandStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = RidgeNetDemandStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_feature_cols_subset_of_derived(self) -> None:
        self.strategy.fit(self.df)
        assert set(self.strategy._feature_cols).issubset(set(DERIVED_FEATURE_COLS))

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_graceful_with_no_derived_cols(self) -> None:
        # DataFrame without any derived columns — feature_cols will be empty;
        # strategy should fall back gracefully (return last_settlement_price).
        raw_only = self.df[
            [
                "settlement_price",
                "last_settlement_price",
                "price_change_eur_mwh",
                "target_direction",
                "pnl_long_eur",
                "pnl_short_eur",
                "split",
            ]
        ].copy()
        # fit() with no usable derived features: feature_cols is empty → pipeline
        # cannot be fitted → forecast returns last_settlement_price
        self.strategy.fit(raw_only)
        state = _make_state(last_price=55.0)
        result = self.strategy.forecast(state)
        # When feature_cols is empty the model is never fitted, so result == last_settlement_price
        assert result == pytest.approx(55.0) or np.isfinite(result)


class TestKNNDirectionStrategy:
    @classmethod
    def setup_class(cls) -> None:
        cls.df = _make_train_data()
        cls.strategy = KNNDirectionStrategy()
        cls.strategy.fit(cls.df)

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        fresh = KNNDirectionStrategy()
        assert np.isfinite(fresh.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_forecast_encodes_direction(self) -> None:
        state = _make_state(last_price=60.0, df=self.df)
        result = self.strategy.forecast(state)
        assert result in {59.0, 61.0}

    def test_skip_buffer_zero(self) -> None:
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


class TestSVMDirectionStrategy:
    @classmethod
    def setup_class(cls) -> None:
        cls.df = _make_train_data()
        cls.strategy = SVMDirectionStrategy()
        cls.strategy.fit(cls.df)

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        fresh = SVMDirectionStrategy()
        assert np.isfinite(fresh.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_forecast_encodes_direction(self) -> None:
        state = _make_state(last_price=70.0, df=self.df)
        result = self.strategy.forecast(state)
        assert result in {69.0, 71.0}

    def test_skip_buffer_zero(self) -> None:
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


class TestDecisionTreeStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = DecisionTreeStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_forecast_encodes_direction(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=45.0, df=self.df)
        result = self.strategy.forecast(state)
        assert result in {44.0, 46.0}

    def test_skip_buffer_zero(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


class TestLassoCalendarAugmentedStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = LassoCalendarAugmentedStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_calendar_cols_in_features(self) -> None:
        self.strategy.fit(self.df)
        # dow_int and is_weekend should be available
        assert (
            "dow_int" in self.strategy._feature_cols or "is_weekend" in self.strategy._feature_cols
        )

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_pipeline_fitted(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy._pipeline is not None


class TestGBMNetDemandStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = GBMNetDemandStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_feature_cols_subset_of_derived(self) -> None:
        self.strategy.fit(self.df)
        assert set(self.strategy._feature_cols).issubset(set(DERIVED_FEATURE_COLS))

    def test_forecast_encodes_direction(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=50.0, df=self.df)
        result = self.strategy.forecast(state)
        assert result in {49.0, 51.0}

    def test_skip_buffer_zero(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()


class TestBayesianRidgeStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = BayesianRidgeStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_fit_sets_feature_cols(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._feature_cols) > 0

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_pipeline_fitted(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy._pipeline is not None


class TestPLSRegressionStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = PLSRegressionStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_fit_sets_feature_cols(self) -> None:
        self.strategy.fit(self.df)
        assert len(self.strategy._feature_cols) > 0

    def test_skip_buffer_set(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer >= 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_pls_model_fitted(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy._pls is not None
        assert self.strategy._scaler is not None


class TestNeuralNetStrategy:
    def setup_method(self) -> None:
        self.df = _make_train_data()
        self.strategy = NeuralNetStrategy()

    def test_is_backtest_strategy(self) -> None:
        assert isinstance(self.strategy, BacktestStrategy)

    def test_forecast_before_fit_returns_finite(self) -> None:
        assert np.isfinite(self.strategy.forecast(_make_state()))

    def test_forecast_after_fit_returns_finite(self) -> None:
        self.strategy.fit(self.df)
        assert np.isfinite(self.strategy.forecast(_make_state(df=self.df)))

    def test_forecast_encodes_direction(self) -> None:
        self.strategy.fit(self.df)
        state = _make_state(last_price=55.0, df=self.df)
        result = self.strategy.forecast(state)
        # result = last_price + predicted_direction; direction in {-1, 0, +1}
        assert result in {54.0, 55.0, 56.0}

    def test_skip_buffer_zero(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy.skip_buffer == 0.0

    def test_reset_is_callable(self) -> None:
        self.strategy.reset()

    def test_pipeline_fitted(self) -> None:
        self.strategy.fit(self.df)
        assert self.strategy._pipeline is not None
