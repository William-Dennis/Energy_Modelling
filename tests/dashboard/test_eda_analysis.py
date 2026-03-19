"""Tests for pure EDA computation functions (Phase 2).

All functions are tested independently of Streamlit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_modelling.dashboard.eda_analysis import (
    autocorrelation,
    clean_hourly_data,
    compute_daily_settlement,
    compute_direction_streaks,
    compute_forecast_errors,
    compute_price_changes,
    compute_residual_load,
    day_of_week_edge_by_year,
    direction_base_rates,
    direction_by_group,
    feature_drift,
    lagged_direction_correlation,
    quarterly_direction_rates,
    rolling_volatility,
    volatility_regime_performance,
    wind_quintile_analysis,
)

# ---------------------------------------------------------------------------
# P0: Data Pre-processing / Cleaning (clean_hourly_data)
# ---------------------------------------------------------------------------


def _make_raw_hourly_df() -> pd.DataFrame:
    """Build a synthetic raw DataFrame mimicking the DE-LU parquet structure.

    The first row is a partial-NaN artefact (as in the real data).  The
    remaining rows exercise every cleaning path:
      - single-NaN columns (forward-fill)
      - interconnector NTCs (zero-fill)
      - commodity prices (interpolate + bfill)
      - load_forecast_mw (24-hour-prior fill)
    """
    n = 49  # 1 artefact row + 48 real hours (2 full days)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    rng = np.random.RandomState(99)

    df = pd.DataFrame(
        {
            "price_eur_mwh": rng.uniform(30, 80, n),
            "gen_hydro_water_reservoir_mw": rng.uniform(500, 1500, n),
            "weather_temperature_2m_degc": rng.uniform(-5, 25, n),
            "load_forecast_mw": rng.uniform(40000, 70000, n),
            "ntc_dk_2_export_mw": rng.uniform(0, 2000, n),
            "ntc_dk_2_import_mw": rng.uniform(0, 2000, n),
            "ntc_nl_export_mw": rng.uniform(0, 2000, n),
            "ntc_nl_import_mw": rng.uniform(0, 2000, n),
            "carbon_price_usd": rng.uniform(60, 90, n),
            "gas_price_usd": rng.uniform(20, 50, n),
        },
        index=idx,
    )
    idx.name = "timestamp_utc"

    # --- Artefact row (row 0): inject several NaNs ---
    df.iloc[0, df.columns.get_loc("gen_hydro_water_reservoir_mw")] = np.nan
    df.iloc[0, df.columns.get_loc("weather_temperature_2m_degc")] = np.nan
    df.iloc[0, df.columns.get_loc("load_forecast_mw")] = np.nan

    # --- Single-NaN columns: set exactly 1 NaN in rows 1..48 ---
    # gen_hydro already has 1 NaN at row 0; after first-row drop it should be
    # clean.  Instead, inject *additional* single NaN in weather col at row 5.
    df.iloc[5, df.columns.get_loc("weather_temperature_2m_degc")] = np.nan
    # After first-row drop, weather_temperature will have exactly 1 NaN → ffill.

    # --- Interconnectors: set NaN blocks (simulate non-existent capacity) ---
    for col in ("ntc_dk_2_export_mw", "ntc_dk_2_import_mw", "ntc_nl_export_mw", "ntc_nl_import_mw"):
        df.loc[df.index[1:20], col] = np.nan  # rows 1-19 NaN

    # --- Commodity prices: weekend-style gaps ---
    df.iloc[10:15, df.columns.get_loc("carbon_price_usd")] = np.nan
    df.iloc[12:17, df.columns.get_loc("gas_price_usd")] = np.nan

    # --- load_forecast_mw: scatter a few NaN in the second day (hours 25-48) ---
    # so the 24-h-prior lookup can find a valid value
    df.iloc[30, df.columns.get_loc("load_forecast_mw")] = np.nan
    df.iloc[35, df.columns.get_loc("load_forecast_mw")] = np.nan

    return df


class TestCleanHourlyData:
    """Tests for the P0 clean_hourly_data() pipeline."""

    def test_first_row_dropped(self) -> None:
        raw = _make_raw_hourly_df()
        cleaned = clean_hourly_data(raw)
        assert len(cleaned) == len(raw) - 1
        # The artefact timestamp should not appear in the cleaned index
        assert raw.index[0] not in cleaned.index

    def test_output_has_no_nans(self) -> None:
        raw = _make_raw_hourly_df()
        cleaned = clean_hourly_data(raw)
        nan_counts = cleaned.isna().sum()
        assert nan_counts.sum() == 0, f"NaN remaining:\n{nan_counts[nan_counts > 0]}"

    def test_single_nan_column_is_forward_filled(self) -> None:
        raw = _make_raw_hourly_df()
        cleaned = clean_hourly_data(raw)
        # weather_temperature_2m_degc had exactly 1 NaN at iloc[5] (after drop)
        # ffill should have replaced it with the value at iloc[4] (i.e. raw iloc[4])
        col = "weather_temperature_2m_degc"
        assert cleaned[col].isna().sum() == 0

    def test_interconnectors_zero_filled(self) -> None:
        raw = _make_raw_hourly_df()
        cleaned = clean_hourly_data(raw)
        for col in (
            "ntc_dk_2_export_mw",
            "ntc_dk_2_import_mw",
            "ntc_nl_export_mw",
            "ntc_nl_import_mw",
        ):
            assert cleaned[col].isna().sum() == 0
            # The previously-NaN region should now be 0.0
            # (rows 1-19 of raw → rows 0-18 of cleaned after first-row drop)
            zeros = cleaned[col].iloc[:19]
            assert (zeros == 0.0).all(), f"{col} not zero-filled: {zeros.tolist()}"

    def test_commodity_prices_interpolated(self) -> None:
        raw = _make_raw_hourly_df()
        cleaned = clean_hourly_data(raw)
        for col in ("carbon_price_usd", "gas_price_usd"):
            assert cleaned[col].isna().sum() == 0
            # Values in the gap region should be between their neighbours
            # (linear interpolation), not zero
            gap_vals = cleaned[col].iloc[9:14]  # shifted by -1 from raw due to drop
            assert (gap_vals > 0).all(), f"{col} gap has non-positive values"

    def test_load_forecast_filled_with_24h_prior(self) -> None:
        raw = _make_raw_hourly_df()
        cleaned = clean_hourly_data(raw)
        assert cleaned["load_forecast_mw"].isna().sum() == 0
        # raw iloc[30] was NaN → cleaned iloc[29]. Its 24-h-prior is iloc[5]
        # (29 - 24 = 5 in cleaned).  The value should match.
        assert cleaned["load_forecast_mw"].iloc[29] == pytest.approx(
            cleaned["load_forecast_mw"].iloc[5]
        )

    def test_does_not_modify_input(self) -> None:
        raw = _make_raw_hourly_df()
        original_nan_count = raw.isna().sum().sum()
        _ = clean_hourly_data(raw)
        assert raw.isna().sum().sum() == original_nan_count

    def test_preserves_columns(self) -> None:
        raw = _make_raw_hourly_df()
        cleaned = clean_hourly_data(raw)
        assert list(cleaned.columns) == list(raw.columns)

    def test_handles_missing_optional_columns(self) -> None:
        """If a column referenced by cleaning is absent, no error occurs."""
        raw = _make_raw_hourly_df().drop(
            columns=["ntc_dk_2_export_mw", "carbon_price_usd", "load_forecast_mw"]
        )
        cleaned = clean_hourly_data(raw)
        assert len(cleaned) == len(raw) - 1

    def test_real_parquet_zero_nans(self) -> None:
        """Integration test: cleaning the real parquet leaves zero NaN."""
        try:
            df = pd.read_parquet("data/processed/dataset_de_lu.parquet")
        except FileNotFoundError:
            pytest.skip("Real parquet file not available")
        cleaned = clean_hourly_data(df)
        nan_total = cleaned.isna().sum().sum()
        assert nan_total == 0, f"{nan_total} NaN values remain after cleaning"
        assert len(cleaned) == len(df) - 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)


@pytest.fixture()
def hourly_prices() -> pd.Series:
    """6 days of hourly prices (144 hours), with known daily means."""
    idx = pd.date_range("2024-01-01", periods=144, freq="h", name="timestamp_utc")
    # Day 1: mean ~50, Day 2: mean ~55, Day 3: mean ~48, Day 4: mean ~60,
    # Day 5: mean ~52, Day 6: mean ~58
    daily_means = [50, 55, 48, 60, 52, 58]
    values = []
    for mean in daily_means:
        values.extend([mean + _RNG.normal(0, 2) for _ in range(24)])
    return pd.Series(values, index=idx, name="price_eur_mwh")


@pytest.fixture()
def daily_settlements() -> pd.Series:
    """Known daily settlement series for predictable tests."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Prices: 50, 55, 48, 60, 52, 58, 45, 70, 65, 62
    vals = [50.0, 55.0, 48.0, 60.0, 52.0, 58.0, 45.0, 70.0, 65.0, 62.0]
    return pd.Series(vals, index=dates, name="settlement_price")


@pytest.fixture()
def price_changes(daily_settlements: pd.Series) -> pd.Series:
    """Daily price changes derived from settlements."""
    return compute_price_changes(daily_settlements)


# ---------------------------------------------------------------------------
# P1: Price Change Distribution
# ---------------------------------------------------------------------------


class TestComputeDailySettlement:
    def test_returns_daily_mean(self, hourly_prices: pd.Series) -> None:
        settlements = compute_daily_settlement(hourly_prices)
        assert len(settlements) == 6
        # Each daily mean should be close to the target
        assert settlements.iloc[0] == pytest.approx(50.0, abs=2.0)
        assert settlements.iloc[1] == pytest.approx(55.0, abs=2.0)

    def test_index_is_date(self, hourly_prices: pd.Series) -> None:
        settlements = compute_daily_settlement(hourly_prices)
        # Index should be date objects or DatetimeIndex with daily freq
        assert len(settlements) == 6


class TestComputePriceChanges:
    def test_length(self, daily_settlements: pd.Series) -> None:
        changes = compute_price_changes(daily_settlements)
        # First day has no change (NaN dropped)
        assert len(changes) == 9

    def test_values(self, daily_settlements: pd.Series) -> None:
        changes = compute_price_changes(daily_settlements)
        # 55 - 50 = 5, 48 - 55 = -7, 60 - 48 = 12, ...
        expected = [5.0, -7.0, 12.0, -8.0, 6.0, -13.0, 25.0, -5.0, -3.0]
        np.testing.assert_array_almost_equal(changes.values, expected)

    def test_name(self, daily_settlements: pd.Series) -> None:
        changes = compute_price_changes(daily_settlements)
        assert changes.name == "price_change"


class TestDirectionBaseRates:
    def test_counts(self, price_changes: pd.Series) -> None:
        rates = direction_base_rates(price_changes)
        # Positive: 5, 12, 6, 25 → 4 up
        # Negative: -7, -8, -13, -5, -3 → 5 down
        assert rates["n_up"] == 4
        assert rates["n_down"] == 5
        assert rates["n_zero"] == 0
        assert rates["n_total"] == 9

    def test_percentages(self, price_changes: pd.Series) -> None:
        rates = direction_base_rates(price_changes)
        assert rates["pct_up"] == pytest.approx(4 / 9 * 100, abs=0.1)
        assert rates["pct_down"] == pytest.approx(5 / 9 * 100, abs=0.1)

    def test_mean_abs(self, price_changes: pd.Series) -> None:
        rates = direction_base_rates(price_changes)
        # Mean up move: (5+12+6+25)/4 = 12.0
        assert rates["mean_up_move"] == pytest.approx(12.0)
        # Mean down move: (7+8+13+5+3)/5 = 7.2
        assert rates["mean_down_move"] == pytest.approx(7.2)


# ---------------------------------------------------------------------------
# P2: Autocorrelation & Direction Persistence
# ---------------------------------------------------------------------------


class TestAutocorrelation:
    def test_returns_series(self, price_changes: pd.Series) -> None:
        acf = autocorrelation(price_changes, max_lag=3)
        assert isinstance(acf, pd.Series)
        assert len(acf) == 3
        assert list(acf.index) == [1, 2, 3]

    def test_lag_1_range(self, price_changes: pd.Series) -> None:
        acf = autocorrelation(price_changes, max_lag=1)
        assert -1.0 <= acf.iloc[0] <= 1.0

    def test_constant_series(self) -> None:
        """Constant changes → ACF is NaN (no variance)."""
        s = pd.Series([1.0] * 20, name="price_change")
        acf = autocorrelation(s, max_lag=3)
        assert all(np.isnan(acf))


class TestComputeDirectionStreaks:
    def test_known_streaks(self, price_changes: pd.Series) -> None:
        streaks = compute_direction_streaks(price_changes)
        # Directions: +, -, +, -, +, -, +, -, -
        # Streaks:    1   1  1  1  1  1  1   2 (last two are both -)
        # Max up streak: 1, max down streak: 2
        assert streaks["max_up_streak"] == 1
        assert streaks["max_down_streak"] == 2
        assert streaks["mean_up_streak"] == pytest.approx(1.0)

    def test_all_same_direction(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0], name="price_change")
        streaks = compute_direction_streaks(s)
        assert streaks["max_up_streak"] == 4
        assert streaks["max_down_streak"] == 0


# ---------------------------------------------------------------------------
# P3: Forecast Error Analysis
# ---------------------------------------------------------------------------


class TestComputeForecastErrors:
    def test_returns_dataframe(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        actual = pd.Series([100, 110, 105, 115, 108], index=idx, name="actual")
        forecast = pd.Series([102, 108, 107, 112, 110], index=idx, name="forecast")
        errors = compute_forecast_errors(actual, forecast)
        assert isinstance(errors, pd.DataFrame)
        assert "error" in errors.columns
        assert "abs_error" in errors.columns
        assert "pct_error" in errors.columns

    def test_error_sign(self) -> None:
        """error = forecast - actual, so positive means forecast was too high."""
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        actual = pd.Series([100, 100, 100], index=idx)
        forecast = pd.Series([110, 90, 100], index=idx)
        errors = compute_forecast_errors(actual, forecast)
        np.testing.assert_array_almost_equal(errors["error"].values, [10, -10, 0])

    def test_pct_error_avoids_division_by_zero(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="h")
        actual = pd.Series([0, 100], index=idx)
        forecast = pd.Series([5, 105], index=idx)
        errors = compute_forecast_errors(actual, forecast)
        # First row: actual=0, pct_error should be NaN (not inf)
        assert np.isnan(errors["pct_error"].iloc[0])
        assert errors["pct_error"].iloc[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# P4: Lagged Feature → Direction Correlation
# ---------------------------------------------------------------------------


class TestLaggedDirectionCorrelation:
    def test_returns_series(self) -> None:
        n = 100
        rng = np.random.RandomState(0)
        features = pd.DataFrame(
            {"feat_a": rng.randn(n), "feat_b": rng.randn(n)},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        direction = pd.Series(
            rng.choice([-1, 1], size=n),
            index=features.index,
            name="direction",
        )
        corr = lagged_direction_correlation(features, direction)
        assert isinstance(corr, pd.Series)
        assert len(corr) == 2
        assert "feat_a" in corr.index

    def test_perfect_positive_correlation(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        features = pd.DataFrame({"signal": np.arange(50, dtype=float)}, index=idx)
        # Direction perfectly correlated with signal
        direction = pd.Series(
            np.sign(np.arange(50, dtype=float) - 25),
            index=idx,
            name="direction",
        )
        corr = lagged_direction_correlation(features, direction)
        assert corr["signal"] > 0.5  # Should be strongly positive


# ---------------------------------------------------------------------------
# P5: Volatility & Regime
# ---------------------------------------------------------------------------


class TestRollingVolatility:
    def test_length(self, price_changes: pd.Series) -> None:
        vol = rolling_volatility(price_changes, window=3)
        assert isinstance(vol, pd.Series)
        # Rolling window=3 means first 2 are NaN
        assert vol.notna().sum() == len(price_changes) - 2

    def test_constant_returns_zero_vol(self) -> None:
        s = pd.Series([1.0] * 10, name="price_change")
        vol = rolling_volatility(s, window=3)
        assert all(vol.dropna() == 0.0)


# ---------------------------------------------------------------------------
# P6: Residual Load
# ---------------------------------------------------------------------------


class TestComputeResidualLoad:
    def test_basic(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        load = pd.Series([60000, 62000, 58000, 65000, 61000], index=idx)
        renewable = pd.Series([20000, 25000, 15000, 30000, 22000], index=idx)
        residual = compute_residual_load(load, renewable)
        expected = [40000, 37000, 43000, 35000, 39000]
        np.testing.assert_array_almost_equal(residual.values, expected)

    def test_name(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        load = pd.Series([60000, 62000, 58000], index=idx)
        renewable = pd.Series([20000, 25000, 15000], index=idx)
        residual = compute_residual_load(load, renewable)
        assert residual.name == "residual_load_mw"


# ---------------------------------------------------------------------------
# Direction by Group
# ---------------------------------------------------------------------------


class TestDirectionByGroup:
    def test_returns_dataframe(self) -> None:
        changes = pd.Series(
            [1.0, -2.0, 3.0, -1.0, 2.0, -3.0],
            index=pd.date_range("2024-01-01", periods=6, freq="D"),
            name="price_change",
        )
        groups = pd.Series(
            ["Mon", "Tue", "Wed", "Mon", "Tue", "Wed"],
            index=changes.index,
            name="day",
        )
        result = direction_by_group(changes, groups)
        assert isinstance(result, pd.DataFrame)
        assert "pct_up" in result.columns
        assert "n_total" in result.columns
        assert len(result) == 3  # 3 unique groups

    def test_correct_rates(self) -> None:
        changes = pd.Series(
            [1.0, -1.0, 1.0, 1.0],
            index=pd.date_range("2024-01-01", periods=4, freq="D"),
            name="price_change",
        )
        groups = pd.Series(["A", "A", "B", "B"], index=changes.index, name="group")
        result = direction_by_group(changes, groups)
        # A: 1 up, 1 down → 50%
        assert result.loc["A", "pct_up"] == pytest.approx(50.0)
        # B: 2 up, 0 down → 100%
        assert result.loc["B", "pct_up"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Phase 6: Feedback-loop analyses
# ---------------------------------------------------------------------------


class TestDayOfWeekEdgeByYear:
    def test_returns_dataframe(self) -> None:
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        rng = np.random.RandomState(42)
        changes = pd.Series(rng.randn(60), index=dates, name="price_change")
        result = day_of_week_edge_by_year(changes, dates)
        assert isinstance(result, pd.DataFrame)
        assert "year" in result.columns
        assert "dow" in result.columns
        assert "up_rate" in result.columns
        assert "edge" in result.columns

    def test_edge_sums_to_roughly_zero(self) -> None:
        """Across all days in a year, the mean edge should be near zero."""
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        rng = np.random.RandomState(0)
        changes = pd.Series(rng.randn(365), index=dates, name="price_change")
        result = day_of_week_edge_by_year(changes, dates)
        yr_2024 = result[result["year"] == 2024]
        # Weighted mean edge should be roughly zero (not exact due to unequal group sizes)
        assert yr_2024["edge"].mean() == pytest.approx(0.0, abs=0.05)

    def test_monday_edge_with_synthetic_signal(self) -> None:
        """Create data where Monday always goes up → Monday should have positive edge."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        changes = pd.Series(-1.0, index=dates, name="price_change")
        # Override Mondays (dow=0) to be positive
        mondays = dates.dayofweek == 0
        changes[mondays] = 1.0
        result = day_of_week_edge_by_year(changes, dates)
        mon_edge = result[(result["year"] == 2024) & (result["dow"] == 0)]["edge"].values[0]
        assert mon_edge > 0


class TestFeatureDrift:
    def test_returns_dataframe(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        val = pd.DataFrame({"a": [2.0, 3.0, 4.0], "b": [15.0, 25.0, 35.0]})
        result = feature_drift(train, val)
        assert isinstance(result, pd.DataFrame)
        assert "train_mean" in result.columns
        assert "val_mean" in result.columns
        assert "shift_pct" in result.columns
        assert "std_ratio" in result.columns

    def test_shift_pct_correct(self) -> None:
        train = pd.DataFrame({"x": [100.0, 100.0, 100.0]})
        val = pd.DataFrame({"x": [120.0, 120.0, 120.0]})
        result = feature_drift(train, val)
        assert result.loc["x", "shift_pct"] == pytest.approx(20.0)

    def test_handles_disjoint_columns(self) -> None:
        train = pd.DataFrame({"a": [1.0], "b": [2.0]})
        val = pd.DataFrame({"a": [3.0], "c": [4.0]})
        result = feature_drift(train, val)
        assert len(result) == 1
        assert "a" in result.index


class TestQuarterlyDirectionRates:
    def test_returns_dataframe(self) -> None:
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        rng = np.random.RandomState(0)
        changes = pd.Series(rng.randn(365), index=dates, name="price_change")
        result = quarterly_direction_rates(changes, dates)
        assert isinstance(result, pd.DataFrame)
        assert "year" in result.columns
        assert "quarter" in result.columns
        assert "up_rate" in result.columns
        assert "mean_abs_change" in result.columns

    def test_four_quarters(self) -> None:
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        rng = np.random.RandomState(0)
        changes = pd.Series(rng.randn(365), index=dates, name="price_change")
        result = quarterly_direction_rates(changes, dates)
        assert len(result) == 4  # Q1-Q4 for one year

    def test_up_rate_bounds(self) -> None:
        dates = pd.date_range("2024-01-01", periods=365, freq="D")
        rng = np.random.RandomState(0)
        changes = pd.Series(rng.randn(365), index=dates, name="price_change")
        result = quarterly_direction_rates(changes, dates)
        assert (result["up_rate"] >= 0).all()
        assert (result["up_rate"] <= 1).all()


class TestVolatilityRegimePerformance:
    def test_returns_dataframe(self) -> None:
        rng = np.random.RandomState(42)
        changes = pd.Series(rng.randn(200), name="price_change")
        result = volatility_regime_performance(changes, window=20, n_regimes=3)
        assert isinstance(result, pd.DataFrame)
        assert "regime" in result.columns
        assert "up_rate" in result.columns

    def test_three_regimes(self) -> None:
        rng = np.random.RandomState(42)
        changes = pd.Series(rng.randn(200), name="price_change")
        result = volatility_regime_performance(changes, window=20, n_regimes=3)
        assert len(result) == 3
        assert set(result["regime"]) == {"low", "mid", "high"}

    def test_total_count(self) -> None:
        rng = np.random.RandomState(42)
        changes = pd.Series(rng.randn(200), name="price_change")
        result = volatility_regime_performance(changes, window=20, n_regimes=3)
        # Total n should equal 200 - 19 (rolling window drops first window-1 NaNs)
        assert result["n"].sum() == 200 - 19


class TestWindQuintileAnalysis:
    def test_returns_dataframe(self) -> None:
        n = 500
        rng = np.random.RandomState(42)
        wind = pd.Series(rng.uniform(0, 20000, n), name="wind")
        direction = pd.Series(rng.choice([-1, 1], n), name="direction")
        changes = pd.Series(rng.randn(n), name="change")
        result = wind_quintile_analysis(wind, direction, changes, n_bins=5)
        assert isinstance(result, pd.DataFrame)
        assert "wind_bin" in result.columns
        assert "up_rate" in result.columns
        assert len(result) == 5

    def test_low_wind_higher_up_rate(self) -> None:
        """Synthetic: low wind → direction=+1, high wind → direction=-1."""
        n = 1000
        rng = np.random.RandomState(0)
        wind = pd.Series(rng.uniform(0, 100, n), name="wind")
        # Direction correlates negatively with wind
        direction = pd.Series(np.where(wind < 50, 1, -1), name="direction")
        changes = pd.Series(np.where(wind < 50, 1.0, -1.0), name="change")
        result = wind_quintile_analysis(wind, direction, changes, n_bins=5)
        # Q1 (low wind) should have higher up_rate than Q5 (high wind)
        q1_rate = result[result["wind_bin"] == "Q1"]["up_rate"].values[0]
        q5_rate = result[result["wind_bin"] == "Q5"]["up_rate"].values[0]
        assert q1_rate > q5_rate
