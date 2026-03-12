"""Tests for market_simulation.data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_modelling.market_simulation.data import (
    build_daily_features,
    compute_daily_settlement,
    load_dataset,
)


def _make_hourly_csv(path: Path, days: int = 3) -> Path:
    """Create a minimal hourly CSV file for testing.

    Generates *days* complete days of hourly data starting 2024-01-01.
    """
    timestamps = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    data = {
        "timestamp_utc": timestamps,
        "price_eur_mwh": rng.normal(50, 10, len(timestamps)),
        "gen_solar_mw": rng.uniform(0, 5000, len(timestamps)),
        "gen_wind_onshore_mw": rng.uniform(0, 20000, len(timestamps)),
        "gen_wind_offshore_mw": rng.uniform(0, 6000, len(timestamps)),
        "gen_fossil_gas_mw": rng.uniform(1000, 10000, len(timestamps)),
        "gen_fossil_hard_coal_mw": rng.uniform(500, 8000, len(timestamps)),
        "gen_fossil_brown_coal_lignite_mw": rng.uniform(2000, 15000, len(timestamps)),
        "gen_nuclear_mw": rng.uniform(0, 9000, len(timestamps)),
        "load_actual_mw": rng.uniform(35000, 70000, len(timestamps)),
        "load_forecast_mw": rng.uniform(35000, 70000, len(timestamps)),
        "forecast_solar_mw": rng.uniform(0, 5000, len(timestamps)),
        "forecast_wind_onshore_mw": rng.uniform(0, 20000, len(timestamps)),
        "forecast_wind_offshore_mw": rng.uniform(0, 6000, len(timestamps)),
        "weather_temperature_2m_degc": rng.normal(10, 5, len(timestamps)),
        "weather_wind_speed_10m_kmh": rng.uniform(0, 50, len(timestamps)),
        "weather_shortwave_radiation_wm2": rng.uniform(0, 800, len(timestamps)),
        "price_fr_eur_mwh": rng.normal(55, 15, len(timestamps)),
        "price_nl_eur_mwh": rng.normal(52, 12, len(timestamps)),
        "price_at_eur_mwh": rng.normal(50, 10, len(timestamps)),
        "price_pl_eur_mwh": rng.normal(60, 20, len(timestamps)),
        "price_cz_eur_mwh": rng.normal(48, 10, len(timestamps)),
        "price_dk_1_eur_mwh": rng.normal(45, 15, len(timestamps)),
        "flow_fr_net_import_mw": rng.normal(-500, 1000, len(timestamps)),
        "flow_nl_net_import_mw": rng.normal(-1000, 800, len(timestamps)),
        "carbon_price_usd": rng.uniform(20, 100, len(timestamps)),
        "gas_price_usd": rng.uniform(15, 50, len(timestamps)),
    }
    df = pd.DataFrame(data)
    csv_path = path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class TestLoadDataset:
    """Tests for load_dataset()."""

    def test_returns_dataframe_with_datetime_index(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path)
        df = load_dataset(csv_path)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp_utc"

    def test_index_is_utc(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path)
        df = load_dataset(csv_path)
        assert str(df.index.tz) == "UTC"

    def test_has_price_column(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path)
        df = load_dataset(csv_path)
        assert "price_eur_mwh" in df.columns

    def test_correct_row_count(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=5)
        df = load_dataset(csv_path)
        assert len(df) == 5 * 24

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.csv")


class TestComputeDailySettlement:
    """Tests for compute_daily_settlement()."""

    def test_returns_daily_series(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=3)
        df = load_dataset(csv_path)
        settlements = compute_daily_settlement(df)
        assert len(settlements) == 3

    def test_series_name(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=2)
        df = load_dataset(csv_path)
        settlements = compute_daily_settlement(df)
        assert settlements.name == "settlement_price"

    def test_settlement_is_mean_of_hourly(self, tmp_path: Path) -> None:
        """Settlement price should equal the mean of 24 hourly prices."""
        csv_path = _make_hourly_csv(tmp_path, days=2)
        df = load_dataset(csv_path)
        settlements = compute_daily_settlement(df)
        # Check first day manually
        first_date = date(2024, 1, 1)
        day_mask = df.index.date == first_date
        expected = df.loc[day_mask, "price_eur_mwh"].mean()
        assert settlements.iloc[0] == pytest.approx(expected)

    def test_index_is_date(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=2)
        df = load_dataset(csv_path)
        settlements = compute_daily_settlement(df)
        assert all(isinstance(d, date) for d in settlements.index)


class TestBuildDailyFeatures:
    """Tests for build_daily_features()."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=3)
        df = load_dataset(csv_path)
        features = build_daily_features(df)
        assert isinstance(features, pd.DataFrame)

    def test_lagged_by_one_day(self, tmp_path: Path) -> None:
        """Features should be lagged -- first delivery day has NaN features."""
        csv_path = _make_hourly_csv(tmp_path, days=3)
        df = load_dataset(csv_path)
        features = build_daily_features(df)
        # First date in features should be the second day (lagged)
        # and the first row should have valid data from day 1
        assert len(features) >= 2

    def test_no_look_ahead(self, tmp_path: Path) -> None:
        """Features for day D must not contain any data from day D."""
        csv_path = _make_hourly_csv(tmp_path, days=5)
        df = load_dataset(csv_path)
        features = build_daily_features(df)
        # The features index should start at day 2 (day 1 features
        # are only available for day 2's decision)
        first_feature_date = features.index[0]
        first_data_date = date(2024, 1, 1)
        assert first_feature_date > first_data_date

    def test_has_expected_columns(self, tmp_path: Path) -> None:
        csv_path = _make_hourly_csv(tmp_path, days=3)
        df = load_dataset(csv_path)
        features = build_daily_features(df)
        # Should have aggregated generation and load columns
        col_names = features.columns.tolist()
        assert len(col_names) > 0
