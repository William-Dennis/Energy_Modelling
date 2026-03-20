"""Tests for the derived feature engineering module.

All tests use small synthetic DataFrames to verify correctness of
each derived feature function without requiring the full dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_modelling.backtest.feature_engineering import (
    add_aggregated_generation,
    add_calendar_encodings,
    add_commodity_trends,
    add_derived_features,
    add_net_demand,
    add_price_range,
    add_price_spreads,
    add_price_zscore,
    add_renewable_penetration,
    add_rolling_volatility,
    add_surprise_signals,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_df(n: int = 30) -> pd.DataFrame:
    """Minimal DataFrame with all raw columns required by derived features."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "delivery_date": dates,
            # same-day forecast features
            "load_forecast_mw_mean": 50_000 + rng.normal(0, 3_000, n),
            "forecast_wind_onshore_mw_mean": 10_000 + rng.normal(0, 4_000, n),
            "forecast_wind_offshore_mw_mean": 3_000 + rng.normal(0, 1_500, n),
            "forecast_solar_mw_mean": 5_000 + rng.normal(0, 2_000, n),
            # lagged realised price stats
            "price_mean": 80 + rng.normal(0, 20, n),
            "price_max": 120 + rng.normal(0, 25, n),
            "price_min": 40 + rng.normal(0, 15, n),
            "price_change_eur_mwh": rng.normal(0, 15, n),
            # neighbour prices
            "price_fr_eur_mwh_mean": 85 + rng.normal(0, 20, n),
            "price_nl_eur_mwh_mean": 82 + rng.normal(0, 20, n),
            "price_at_eur_mwh_mean": 83 + rng.normal(0, 20, n),
            "price_cz_eur_mwh_mean": 84 + rng.normal(0, 20, n),
            "price_pl_eur_mwh_mean": 86 + rng.normal(0, 20, n),
            "price_dk_1_eur_mwh_mean": 81 + rng.normal(0, 20, n),
            # commodity prices
            "gas_price_usd_mean": 50 + rng.normal(0, 10, n),
            "carbon_price_usd_mean": 22 + rng.normal(0, 3, n),
            # generation (lagged)
            "gen_wind_onshore_mw_mean": 11_000 + rng.normal(0, 4_000, n),
            "gen_wind_offshore_mw_mean": 3_400 + rng.normal(0, 1_500, n),
            "load_actual_mw_mean": 55_000 + rng.normal(0, 4_000, n),
            "gen_fossil_gas_mw_mean": 6_000 + rng.normal(0, 2_000, n),
            "gen_fossil_hard_coal_mw_mean": 3_500 + rng.normal(0, 1_500, n),
            "gen_fossil_brown_coal_lignite_mw_mean": 10_500 + rng.normal(0, 1_500, n),
            # flows (lagged)
            "flow_fr_net_import_mw_mean": -2_000 + rng.normal(0, 1_000, n),
            "flow_nl_net_import_mw_mean": -600 + rng.normal(0, 800, n),
        }
    )


# ---------------------------------------------------------------------------
# Group 1: Supply / demand balance
# ---------------------------------------------------------------------------


class TestNetDemand:
    def test_column_added(self) -> None:
        df = add_net_demand(_base_df())
        assert "net_demand_mw" in df.columns

    def test_formula_correct(self) -> None:
        df = _base_df(5)
        out = add_net_demand(df)
        expected = (
            df["load_forecast_mw_mean"]
            - df["forecast_wind_onshore_mw_mean"]
            - df["forecast_wind_offshore_mw_mean"]
            - df["forecast_solar_mw_mean"]
        )
        pd.testing.assert_series_equal(
            out["net_demand_mw"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_no_original_columns_modified(self) -> None:
        df = _base_df(5)
        orig_load = df["load_forecast_mw_mean"].copy()
        add_net_demand(df)
        pd.testing.assert_series_equal(df["load_forecast_mw_mean"], orig_load)


class TestRenewablePenetration:
    def test_column_added(self) -> None:
        df = add_renewable_penetration(_base_df())
        assert "renewable_penetration_pct" in df.columns

    def test_between_zero_and_one_for_typical_values(self) -> None:
        df = add_renewable_penetration(_base_df())
        pct = df["renewable_penetration_pct"]
        assert (pct >= 0.0).all()

    def test_zero_load_handled_gracefully(self) -> None:
        df = _base_df(3)
        df["load_forecast_mw_mean"] = 0.0
        out = add_renewable_penetration(df)
        assert (out["renewable_penetration_pct"] == 0.0).all()


# ---------------------------------------------------------------------------
# Group 2: Price spreads
# ---------------------------------------------------------------------------


class TestPriceSpreads:
    def test_columns_added(self) -> None:
        df = add_price_spreads(_base_df())
        for col in ("de_fr_spread", "de_nl_spread", "de_avg_neighbour_spread"):
            assert col in df.columns

    def test_de_fr_spread_formula(self) -> None:
        df = _base_df(5)
        out = add_price_spreads(df)
        expected = df["price_mean"] - df["price_fr_eur_mwh_mean"]
        pd.testing.assert_series_equal(
            out["de_fr_spread"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_avg_neighbour_spread_is_mean_of_six(self) -> None:
        df = _base_df(5)
        out = add_price_spreads(df)
        neighbours = [
            "price_fr_eur_mwh_mean",
            "price_nl_eur_mwh_mean",
            "price_at_eur_mwh_mean",
            "price_cz_eur_mwh_mean",
            "price_pl_eur_mwh_mean",
            "price_dk_1_eur_mwh_mean",
        ]
        expected = df["price_mean"] - df[neighbours].mean(axis=1)
        pd.testing.assert_series_equal(
            out["de_avg_neighbour_spread"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Group 3: Price mean-reversion
# ---------------------------------------------------------------------------


class TestPriceZScore:
    def test_column_added(self) -> None:
        df = add_price_zscore(_base_df())
        assert "price_zscore_20d" in df.columns

    def test_zero_at_constant_price(self) -> None:
        df = _base_df(25)
        df["price_mean"] = 80.0  # constant → z-score should be 0
        out = add_price_zscore(df)
        # After enough periods, std→0; fillna(0) handles it
        assert not out["price_zscore_20d"].isna().any()

    def test_positive_for_price_above_ma(self) -> None:
        df = _base_df(30)
        # Set last price much higher than all previous
        df.loc[df.index[-1], "price_mean"] = 200.0
        out = add_price_zscore(df)
        assert out["price_zscore_20d"].iloc[-1] > 0


class TestPriceRange:
    def test_column_added(self) -> None:
        assert "price_range" in add_price_range(_base_df()).columns

    def test_non_negative(self) -> None:
        df = _base_df()
        df["price_max"] = df["price_min"] + 30  # ensure max > min
        out = add_price_range(df)
        assert (out["price_range"] >= 0).all()


# ---------------------------------------------------------------------------
# Group 4: Commodity trends
# ---------------------------------------------------------------------------


class TestCommodityTrends:
    def test_columns_added(self) -> None:
        df = add_commodity_trends(_base_df())
        for col in ("gas_trend_3d", "carbon_trend_3d", "fuel_cost_index"):
            assert col in df.columns

    def test_gas_trend_is_3d_diff(self) -> None:
        df = _base_df(10)
        out = add_commodity_trends(df)
        # NaN positions (first 3 rows from .diff(3)) are filled with 0.0
        expected = df["gas_price_usd_mean"].diff(3).fillna(0.0)
        pd.testing.assert_series_equal(
            out["gas_trend_3d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_fuel_cost_index_formula(self) -> None:
        df = _base_df(5)
        out = add_commodity_trends(df)
        expected = df["gas_price_usd_mean"] * 7.5 + df["carbon_price_usd_mean"] * 0.37
        pd.testing.assert_series_equal(
            out["fuel_cost_index"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Group 5: Surprise signals
# ---------------------------------------------------------------------------


class TestSurpriseSignals:
    def test_columns_added(self) -> None:
        df = add_surprise_signals(_base_df())
        for col in ("wind_forecast_error", "load_surprise"):
            assert col in df.columns

    def test_no_nans(self) -> None:
        df = add_surprise_signals(_base_df())
        assert not df["wind_forecast_error"].isna().any()
        assert not df["load_surprise"].isna().any()

    def test_wind_forecast_error_positive_when_forecast_high(self) -> None:
        df = _base_df(5)
        df["forecast_wind_onshore_mw_mean"] = 20_000.0
        df["forecast_wind_offshore_mw_mean"] = 5_000.0
        df["gen_wind_onshore_mw_mean"] = 5_000.0
        df["gen_wind_offshore_mw_mean"] = 1_000.0
        out = add_surprise_signals(df)
        assert (out["wind_forecast_error"] > 0).all()


# ---------------------------------------------------------------------------
# Group 6: Volatility regime
# ---------------------------------------------------------------------------


class TestRollingVolatility:
    def test_columns_added(self) -> None:
        df = add_rolling_volatility(_base_df())
        for col in ("rolling_vol_7d", "rolling_vol_14d"):
            assert col in df.columns

    def test_no_nans(self) -> None:
        df = add_rolling_volatility(_base_df())
        assert not df["rolling_vol_7d"].isna().any()
        assert not df["rolling_vol_14d"].isna().any()

    def test_vol_non_negative(self) -> None:
        df = add_rolling_volatility(_base_df())
        assert (df["rolling_vol_7d"] >= 0).all()
        assert (df["rolling_vol_14d"] >= 0).all()

    def test_vol_increases_with_volatility(self) -> None:
        df = _base_df(20)
        # Set early rows to near-zero change and late rows to alternating large values
        df["price_change_eur_mwh"] = 0.0
        df.loc[df.index[-7:], "price_change_eur_mwh"] = [
            100.0,
            -100.0,
            100.0,
            -100.0,
            100.0,
            -100.0,
            100.0,
        ]
        out = add_rolling_volatility(df)
        assert out["rolling_vol_7d"].iloc[-1] > out["rolling_vol_7d"].iloc[5]


# ---------------------------------------------------------------------------
# Group 7: Aggregated generation / flow
# ---------------------------------------------------------------------------


class TestAggregatedGeneration:
    def test_columns_added(self) -> None:
        df = add_aggregated_generation(_base_df())
        for col in ("total_fossil_mw", "net_flow_mw"):
            assert col in df.columns

    def test_total_fossil_formula(self) -> None:
        df = _base_df(5)
        out = add_aggregated_generation(df)
        expected = (
            df["gen_fossil_gas_mw_mean"]
            + df["gen_fossil_hard_coal_mw_mean"]
            + df["gen_fossil_brown_coal_lignite_mw_mean"]
        )
        pd.testing.assert_series_equal(
            out["total_fossil_mw"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_net_flow_formula(self) -> None:
        df = _base_df(5)
        out = add_aggregated_generation(df)
        expected = df["flow_fr_net_import_mw_mean"] + df["flow_nl_net_import_mw_mean"]
        pd.testing.assert_series_equal(
            out["net_flow_mw"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Group 8: Calendar encodings
# ---------------------------------------------------------------------------


class TestCalendarEncodings:
    def test_columns_added(self) -> None:
        df = add_calendar_encodings(_base_df())
        for col in ("dow_int", "is_weekend"):
            assert col in df.columns

    def test_monday_is_1(self) -> None:
        df = _base_df(7)
        # 2023-01-02 is a Monday
        df["delivery_date"] = pd.date_range("2023-01-02", periods=7, freq="D")
        out = add_calendar_encodings(df)
        assert out["dow_int"].iloc[0] == 1  # Monday

    def test_sunday_is_7(self) -> None:
        df = _base_df(7)
        df["delivery_date"] = pd.date_range("2023-01-02", periods=7, freq="D")
        out = add_calendar_encodings(df)
        assert out["dow_int"].iloc[6] == 7  # Sunday

    def test_weekend_flag_correct(self) -> None:
        df = _base_df(7)
        # Mon 2023-01-02 → Sun 2023-01-08
        df["delivery_date"] = pd.date_range("2023-01-02", periods=7, freq="D")
        out = add_calendar_encodings(df)
        # Mon-Fri should be False, Sat-Sun True
        assert list(out["is_weekend"]) == [False, False, False, False, False, True, True]


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------


class TestAddDerivedFeatures:
    def test_all_columns_present(self) -> None:
        expected_cols = [
            "net_demand_mw",
            "renewable_penetration_pct",
            "de_fr_spread",
            "de_nl_spread",
            "de_avg_neighbour_spread",
            "price_zscore_20d",
            "price_range",
            "gas_trend_3d",
            "carbon_trend_3d",
            "fuel_cost_index",
            "wind_forecast_error",
            "load_surprise",
            "rolling_vol_7d",
            "rolling_vol_14d",
            "total_fossil_mw",
            "net_flow_mw",
            "dow_int",
            "is_weekend",
        ]
        df = add_derived_features(_base_df())
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self) -> None:
        df = _base_df()
        original_cols = set(df.columns)
        out = add_derived_features(df)
        assert original_cols.issubset(set(out.columns))

    def test_no_nans_in_core_columns(self) -> None:
        df = add_derived_features(_base_df(50))
        # Core columns with guaranteed non-NaN (after fillna(0.0))
        for col in (
            "net_demand_mw",
            "renewable_penetration_pct",
            "de_fr_spread",
            "rolling_vol_7d",
            "total_fossil_mw",
            "dow_int",
        ):
            assert not df[col].isna().any(), f"NaN found in {col}"
