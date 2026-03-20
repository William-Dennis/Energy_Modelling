"""Tests for Phase G feedback loop infrastructure.

Covers:
- StrategyReport dataclass construction and fields
- strategy_correlation_matrix() correctness
- feature_contribution_analysis() correctness
- walk_forward_validate() correctness
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.feedback import (
    StrategyReport,
    feature_contribution_analysis,
    strategy_correlation_matrix,
)
from energy_modelling.backtest.walk_forward import walk_forward_validate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predictions(values: list[int | None], start: str = "2024-01-01") -> pd.Series:
    """Create a predictions Series indexed by date."""
    dates = pd.bdate_range(start, periods=len(values), freq="B")
    idx = [d.date() for d in dates]
    return pd.Series(values, index=idx, name="prediction", dtype="Int64")


def _make_daily_data(n: int = 500, start_date: str = "2019-01-02") -> pd.DataFrame:
    """Create synthetic daily data suitable for backtest runner."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range(start_date, periods=n, freq="B")
    base_price = 50.0
    prices = base_price + rng.randn(n).cumsum() * 0.5

    df = pd.DataFrame(
        {
            "delivery_date": [d.date() for d in dates],
            "settlement_price": prices,
            "last_settlement_price": np.roll(prices, 1),
            "price_change_eur_mwh": np.diff(prices, prepend=prices[0]),
            "target_direction": np.where(np.diff(prices, prepend=prices[0]) > 0, 1, -1),
            "pnl_long_eur": np.diff(prices, prepend=prices[0]) * 24,
            "pnl_short_eur": -np.diff(prices, prepend=prices[0]) * 24,
            "split": "train",
            # Minimal features
            "load_forecast_mw_mean": rng.uniform(40000, 70000, n),
            "forecast_wind_onshore_mw_mean": rng.uniform(2000, 15000, n),
            "forecast_wind_offshore_mw_mean": rng.uniform(500, 5000, n),
            "forecast_solar_mw_mean": rng.uniform(0, 10000, n),
            "gas_price_usd_mean": rng.uniform(20, 80, n),
            "carbon_price_usd_mean": rng.uniform(20, 60, n),
            "weather_temperature_2m_degc_mean": rng.uniform(-5, 35, n),
            "price_mean": prices,
            "price_max": prices + rng.uniform(0, 5, n),
            "price_min": prices - rng.uniform(0, 5, n),
            "price_std": rng.uniform(1, 10, n),
            "gen_fossil_gas_mw_mean": rng.uniform(3000, 12000, n),
            "gen_fossil_hard_coal_mw_mean": rng.uniform(2000, 8000, n),
            "gen_fossil_brown_coal_lignite_mw_mean": rng.uniform(3000, 10000, n),
            "gen_nuclear_mw_mean": rng.uniform(0, 4000, n),
            "gen_wind_onshore_mw_mean": rng.uniform(2000, 15000, n),
            "gen_wind_offshore_mw_mean": rng.uniform(500, 5000, n),
            "flow_fr_net_import_mw_mean": rng.uniform(-3000, 3000, n),
            "flow_nl_net_import_mw_mean": rng.uniform(-2000, 2000, n),
            "price_fr_eur_mwh_mean": prices + rng.uniform(-5, 5, n),
            "price_nl_eur_mwh_mean": prices + rng.uniform(-5, 5, n),
            "price_at_eur_mwh_mean": prices + rng.uniform(-3, 3, n),
            "price_cz_eur_mwh_mean": prices + rng.uniform(-4, 4, n),
            "price_pl_eur_mwh_mean": prices + rng.uniform(-6, 6, n),
            "price_dk1_eur_mwh_mean": prices + rng.uniform(-5, 5, n),
            "load_actual_mw_mean": rng.uniform(40000, 70000, n),
        }
    )
    df.loc[0, "last_settlement_price"] = df.loc[0, "settlement_price"]
    return df


# =========================================================================
# StrategyReport
# =========================================================================


class TestStrategyReport:
    """Tests for the StrategyReport dataclass."""

    def test_creation_with_required_fields(self):
        report = StrategyReport(
            name="TestStrategy",
            total_pnl=100.0,
            sharpe=0.5,
            win_rate=0.55,
            daily_predictions=_make_predictions([1, -1, 1]),
        )
        assert report.name == "TestStrategy"
        assert report.total_pnl == 100.0
        assert report.sharpe == 0.5
        assert report.win_rate == 0.55

    def test_optional_fields_default_to_none(self):
        report = StrategyReport(
            name="S",
            total_pnl=0.0,
            sharpe=0.0,
            win_rate=0.0,
            daily_predictions=_make_predictions([]),
        )
        assert report.regime_performance is None
        assert report.yearly_pnl is None
        assert report.feature_usage is None

    def test_optional_fields_accepted(self):
        report = StrategyReport(
            name="S",
            total_pnl=50.0,
            sharpe=0.3,
            win_rate=0.6,
            daily_predictions=_make_predictions([1]),
            regime_performance={"low_vol": 10.0, "high_vol": -5.0},
            yearly_pnl={2023: 30.0, 2024: 20.0},
            feature_usage=["load_forecast_mw_mean", "gas_price_usd_mean"],
        )
        assert report.regime_performance["low_vol"] == 10.0
        assert report.yearly_pnl[2024] == 20.0
        assert len(report.feature_usage) == 2

    def test_asdict_roundtrip(self):
        preds = _make_predictions([1, -1])
        report = StrategyReport(
            name="S",
            total_pnl=1.0,
            sharpe=0.1,
            win_rate=0.5,
            daily_predictions=preds,
        )
        d = asdict(report)
        assert d["name"] == "S"
        assert d["total_pnl"] == 1.0


# =========================================================================
# Strategy Correlation Matrix
# =========================================================================


class TestStrategyCorrelationMatrix:
    """Tests for strategy_correlation_matrix()."""

    def test_identical_predictions_give_correlation_one(self):
        preds = _make_predictions([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        predictions_map = {"A": preds, "B": preds.copy()}
        corr = strategy_correlation_matrix(predictions_map)
        assert corr.shape == (2, 2)
        assert corr.loc["A", "B"] == pytest.approx(1.0)

    def test_opposite_predictions_give_correlation_neg_one(self):
        preds_a = _make_predictions([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        preds_b = _make_predictions([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        corr = strategy_correlation_matrix({"A": preds_a, "B": preds_b})
        assert corr.loc["A", "B"] == pytest.approx(-1.0)

    def test_uncorrelated_predictions(self):
        rng = np.random.RandomState(123)
        n = 200
        vals_a = rng.choice([1, -1], n).tolist()
        vals_b = rng.choice([1, -1], n).tolist()
        preds_a = _make_predictions(vals_a)
        preds_b = _make_predictions(vals_b)
        corr = strategy_correlation_matrix({"A": preds_a, "B": preds_b})
        # With 200 random samples, correlation should be near zero
        assert abs(corr.loc["A", "B"]) < 0.3

    def test_none_predictions_excluded(self):
        preds_a = _make_predictions([1, None, -1, 1, None, -1, 1, -1, 1, -1])
        preds_b = _make_predictions([1, 1, -1, 1, -1, -1, 1, -1, 1, -1])
        corr = strategy_correlation_matrix({"A": preds_a, "B": preds_b})
        assert corr.shape == (2, 2)
        # Should still compute, just with fewer overlapping points
        assert np.isfinite(corr.loc["A", "B"])

    def test_diagonal_is_one(self):
        preds = _make_predictions([1, -1, 1, -1, 1])
        corr = strategy_correlation_matrix({"A": preds, "B": preds, "C": preds})
        for name in ["A", "B", "C"]:
            assert corr.loc[name, name] == pytest.approx(1.0)

    def test_output_is_symmetric(self):
        rng = np.random.RandomState(77)
        preds_a = _make_predictions(rng.choice([1, -1], 50).tolist())
        preds_b = _make_predictions(rng.choice([1, -1], 50).tolist())
        corr = strategy_correlation_matrix({"A": preds_a, "B": preds_b})
        assert corr.loc["A", "B"] == pytest.approx(corr.loc["B", "A"])


# =========================================================================
# Feature Contribution Analysis
# =========================================================================


class TestFeatureContributionAnalysis:
    """Tests for feature_contribution_analysis()."""

    def test_returns_dataframe_with_expected_columns(self):
        feature_usage = {
            "StratA": ["load_forecast_mw_mean", "gas_price_usd_mean"],
            "StratB": ["load_forecast_mw_mean"],
        }
        daily_pnl_map = {
            "StratA": pd.Series(
                [10.0, -5.0, 8.0], index=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
            ),
            "StratB": pd.Series(
                [3.0, -2.0, 7.0], index=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
            ),
        }
        result = feature_contribution_analysis(feature_usage, daily_pnl_map)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "strategy_count" in result.columns
        assert "mean_pnl" in result.columns

    def test_strategy_count_is_correct(self):
        feature_usage = {
            "A": ["f1", "f2"],
            "B": ["f1"],
            "C": ["f1", "f2", "f3"],
        }
        daily_pnl_map = {
            "A": pd.Series([1.0]),
            "B": pd.Series([2.0]),
            "C": pd.Series([3.0]),
        }
        result = feature_contribution_analysis(feature_usage, daily_pnl_map)
        row_f1 = result[result["feature"] == "f1"].iloc[0]
        assert row_f1["strategy_count"] == 3
        row_f3 = result[result["feature"] == "f3"].iloc[0]
        assert row_f3["strategy_count"] == 1

    def test_mean_pnl_is_average_of_users(self):
        feature_usage = {"A": ["f1"], "B": ["f1"]}
        daily_pnl_map = {
            "A": pd.Series([10.0, 20.0]),
            "B": pd.Series([30.0, 40.0]),
        }
        result = feature_contribution_analysis(feature_usage, daily_pnl_map)
        row = result[result["feature"] == "f1"].iloc[0]
        # A total = 30, B total = 70, mean total PnL = (30 + 70) / 2 = 50
        assert row["mean_pnl"] == pytest.approx(50.0)

    def test_empty_usage_returns_empty(self):
        result = feature_contribution_analysis({}, {})
        assert len(result) == 0


# =========================================================================
# Walk-Forward Validation
# =========================================================================


class TestWalkForwardValidate:
    """Tests for walk_forward_validate()."""

    def test_returns_dataframe(self):
        from strategies.always_long import AlwaysLongStrategy

        daily = _make_daily_data(n=1500, start_date="2019-01-02")
        result = walk_forward_validate(
            strategy_factory=AlwaysLongStrategy,
            daily_data=daily,
            eval_years=[2020, 2021],
        )
        assert isinstance(result, pd.DataFrame)
        assert "eval_year" in result.columns
        assert "total_pnl" in result.columns
        assert "sharpe_ratio" in result.columns

    def test_one_row_per_eval_year(self):
        from strategies.always_long import AlwaysLongStrategy

        daily = _make_daily_data(n=1500, start_date="2019-01-02")
        result = walk_forward_validate(
            strategy_factory=AlwaysLongStrategy,
            daily_data=daily,
            eval_years=[2020, 2021],
        )
        assert len(result) == 2
        assert set(result["eval_year"]) == {2020, 2021}

    def test_training_window_grows(self):
        """Verify that each successive eval year uses more training data."""
        from strategies.always_long import AlwaysLongStrategy

        daily = _make_daily_data(n=1500, start_date="2019-01-02")
        result = walk_forward_validate(
            strategy_factory=AlwaysLongStrategy,
            daily_data=daily,
            eval_years=[2020, 2021, 2022],
        )
        # train_end should be strictly increasing
        assert result.iloc[0]["train_end"] < result.iloc[1]["train_end"]
        assert result.iloc[1]["train_end"] < result.iloc[2]["train_end"]

    def test_pnl_values_are_finite(self):
        from strategies.always_long import AlwaysLongStrategy

        daily = _make_daily_data(n=1500, start_date="2019-01-02")
        result = walk_forward_validate(
            strategy_factory=AlwaysLongStrategy,
            daily_data=daily,
            eval_years=[2020],
        )
        assert np.isfinite(result.iloc[0]["total_pnl"])
        assert np.isfinite(result.iloc[0]["sharpe_ratio"])

    def test_eval_year_with_no_data_skipped(self):
        """If an eval year has no data, it should be skipped gracefully."""
        from strategies.always_long import AlwaysLongStrategy

        daily = _make_daily_data(n=250, start_date="2019-01-02")
        # Only 2019 data exists — 2025 should be skipped
        result = walk_forward_validate(
            strategy_factory=AlwaysLongStrategy,
            daily_data=daily,
            eval_years=[2025],
        )
        assert len(result) == 0
