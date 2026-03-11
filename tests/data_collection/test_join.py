"""Tests for data_collection.join module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.join import (
    build_kaggle_metadata,
    compute_data_quality,
    impute_small_gaps,
    join_datasets,
    load_raw_parquet,
)

# ---------------------------------------------------------------------------
# Helpers — create realistic raw Parquet files
# ---------------------------------------------------------------------------


def _make_prices(hours: int = 168) -> pd.DataFrame:
    """Week of hourly prices."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame(
        {"price_eur_mwh": np.random.default_rng(42).uniform(20, 100, hours)}, index=idx
    )


def _make_generation(hours: int = 168) -> pd.DataFrame:
    """Week of generation data with 4 types."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "wind_onshore": rng.uniform(500, 5000, hours),
            "solar": rng.uniform(0, 3000, hours),
            "fossil_gas": rng.uniform(1000, 8000, hours),
            "nuclear": rng.uniform(3000, 5000, hours),
        },
        index=idx,
    )


def _make_weather(hours: int = 168) -> pd.DataFrame:
    """Week of weather data."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "temperature_2m": rng.uniform(-5, 20, hours),
            "wind_speed_10m": rng.uniform(0, 15, hours),
            "shortwave_radiation": rng.uniform(0, 800, hours),
        },
        index=idx,
    )


def _make_load(hours: int = 168) -> pd.DataFrame:
    """Week of load data."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "load_actual_mw": rng.uniform(30000, 70000, hours),
            "load_forecast_mw": rng.uniform(30000, 70000, hours),
        },
        index=idx,
    )


def _make_forecasts(hours: int = 168) -> pd.DataFrame:
    """Week of forecast data."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "solar": rng.uniform(0, 3000, hours),
            "wind_onshore": rng.uniform(500, 5000, hours),
        },
        index=idx,
    )


def _make_neighbour_prices(hours: int = 168) -> pd.DataFrame:
    """Week of neighbour price data."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "price_fr_eur_mwh": rng.uniform(20, 120, hours),
            "price_nl_eur_mwh": rng.uniform(20, 120, hours),
        },
        index=idx,
    )


def _make_flows(hours: int = 168) -> pd.DataFrame:
    """Week of flow data."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "flow_fr_export_mw": rng.uniform(0, 3000, hours),
            "flow_fr_import_mw": rng.uniform(0, 3000, hours),
            "flow_fr_net_import_mw": rng.uniform(-1500, 1500, hours),
        },
        index=idx,
    )


def _make_carbon(hours: int = 168) -> pd.DataFrame:
    """Week of carbon price data (hourly, forward-filled)."""
    idx = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"carbon_price_usd": rng.uniform(60, 90, hours)},
        index=idx,
    )


def _write_raw_files(tmp_path: Path, hours: int = 168, include_optional: bool = False) -> None:
    """Write all three mandatory raw Parquet files into tmp_path/raw/.

    If ``include_optional`` is True, also writes optional sources.
    """
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    _make_prices(hours).to_parquet(raw / "prices_da.parquet")
    _make_generation(hours).to_parquet(raw / "generation.parquet")
    _make_weather(hours).to_parquet(raw / "weather.parquet")

    if include_optional:
        _make_load(hours).to_parquet(raw / "load.parquet")
        _make_forecasts(hours).to_parquet(raw / "forecasts.parquet")
        _make_neighbour_prices(hours).to_parquet(raw / "neighbour_prices.parquet")
        _make_flows(hours).to_parquet(raw / "flows.parquet")
        _make_carbon(hours).to_parquet(raw / "carbon_price.parquet")


# ---------------------------------------------------------------------------
# Tests: load_raw_parquet
# ---------------------------------------------------------------------------


class TestLoadRawParquet:
    def test_loads_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.parquet"
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(path)
        df = load_raw_parquet(path, "test")
        assert len(df) == 3

    def test_raises_on_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Raw test file not found"):
            load_raw_parquet(tmp_path / "nonexistent.parquet", "test")


# ---------------------------------------------------------------------------
# Tests: impute_small_gaps
# ---------------------------------------------------------------------------


class TestImputeSmallGaps:
    def test_fills_small_gap(self) -> None:
        """A gap of 2 NaNs should be filled (default max=3)."""
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        data = [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        df = pd.DataFrame({"val": data}, index=idx)
        result = impute_small_gaps(df)
        assert not result["val"].isna().any()

    def test_leaves_large_gap(self) -> None:
        """A gap of 5 NaNs should NOT be fully filled with max_gap_hours=3."""
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        data = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0, 10.0]
        df = pd.DataFrame({"val": data}, index=idx)
        result = impute_small_gaps(df, max_gap_hours=3)
        # First 3 NaNs should be filled, remaining 2 should stay NaN
        assert result["val"].isna().sum() == 2

    def test_no_change_without_nans(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = impute_small_gaps(df)
        pd.testing.assert_frame_equal(result, df)


# ---------------------------------------------------------------------------
# Tests: compute_data_quality
# ---------------------------------------------------------------------------


class TestComputeDataQuality:
    def test_complete_data(self) -> None:
        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"a": range(24), "b": range(24)}, index=idx)
        quality = compute_data_quality(df)
        assert quality["total_rows"] == 24
        assert quality["missing_percent"]["a"] == 0.0
        assert quality["missing_percent"]["b"] == 0.0

    def test_partial_missing(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        vals: list[float | None] = [1.0] * 90 + [np.nan] * 10  # type: ignore[list-item]
        df = pd.DataFrame({"a": vals}, index=idx)
        quality = compute_data_quality(df)
        assert quality["missing_percent"]["a"] == 10.0

    def test_date_range(self) -> None:
        idx = pd.date_range("2024-06-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame({"a": range(48)}, index=idx)
        quality = compute_data_quality(df)
        assert "2024-06-01" in quality["date_range_start"]
        assert "2024-06-02" in quality["date_range_end"]


# ---------------------------------------------------------------------------
# Tests: build_kaggle_metadata
# ---------------------------------------------------------------------------


class TestBuildKaggleMetadata:
    def test_structure(self) -> None:
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024])
        quality = {
            "total_rows": 8760,
            "missing_percent": {},
            "columns": [],
            "date_range_start": None,
            "date_range_end": None,
        }
        meta = build_kaggle_metadata(cfg, quality)

        assert "title" in meta
        assert "description" in meta
        assert "sources" in meta
        assert "license" in meta
        assert meta["bidding_zone"] == "DE_LU"
        assert meta["timezone"] == "UTC"
        assert "neighbour_zones" in meta

    def test_years_included(self) -> None:
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2023, 2024])
        quality = {
            "total_rows": 0,
            "missing_percent": {},
            "columns": [],
            "date_range_start": None,
            "date_range_end": None,
        }
        meta = build_kaggle_metadata(cfg, quality)
        assert meta["years"] == [2023, 2024]

    def test_all_sources_listed(self) -> None:
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024])
        quality = {
            "total_rows": 0,
            "missing_percent": {},
            "columns": [],
            "date_range_start": None,
            "date_range_end": None,
        }
        meta = build_kaggle_metadata(cfg, quality)
        assert "carbon_price" in meta["sources"]
        assert "gas_price" in meta["sources"]
        assert "flows" in meta["sources"]
        assert "ntc" in meta["sources"]
        assert "forecasts" in meta["sources"]
        assert "load" in meta["sources"]
        assert "neighbour_prices" in meta["sources"]


# ---------------------------------------------------------------------------
# Tests: join_datasets (integration)
# ---------------------------------------------------------------------------


class TestJoinDatasets:
    def test_produces_parquet(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        assert result.exists()
        assert result.suffix == ".parquet"

    def test_contains_price_with_unit(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        df = pd.read_parquet(result)
        assert "price_eur_mwh" in df.columns

    def test_gen_columns_have_mw_suffix(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        df = pd.read_parquet(result)
        gen_cols = [c for c in df.columns if c.startswith("gen_")]
        assert len(gen_cols) > 0
        for c in gen_cols:
            assert c.endswith("_mw"), f"Generation column '{c}' should end with '_mw'"

    def test_weather_columns_have_unit_suffix(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        df = pd.read_parquet(result)
        weather_cols = [c for c in df.columns if c.startswith("weather_")]
        assert len(weather_cols) > 0
        # Each weather column should have a unit suffix
        for c in weather_cols:
            parts = c.split("_")
            assert parts[-1] in ("degc", "pct", "kmh", "wm2", "mm"), (
                f"Weather column '{c}' missing unit suffix"
            )

    def test_optional_sources_joined(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path, include_optional=True)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        df = pd.read_parquet(result)

        # Load columns
        assert "load_actual_mw" in df.columns
        assert "load_forecast_mw" in df.columns
        # Forecast columns (prefixed)
        forecast_cols = [c for c in df.columns if c.startswith("forecast_")]
        assert len(forecast_cols) > 0
        # Neighbour prices
        assert "price_fr_eur_mwh" in df.columns
        assert "price_nl_eur_mwh" in df.columns
        # Flow columns
        assert "flow_fr_export_mw" in df.columns
        # Carbon price
        assert "carbon_price_usd" in df.columns

    def test_works_without_optional_sources(self, tmp_path: Path) -> None:
        """Should work fine with only the 3 mandatory sources."""
        _write_raw_files(tmp_path, include_optional=False)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        df = pd.read_parquet(result)
        assert "price_eur_mwh" in df.columns
        assert len(df) == 168

    def test_index_is_utc(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        df = pd.read_parquet(result)
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        result = join_datasets(cfg)
        df = pd.read_parquet(result)
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates

    def test_kaggle_export(self, tmp_path: Path) -> None:
        _write_raw_files(tmp_path, include_optional=True)
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        join_datasets(cfg, kaggle=True)

        csv_path = tmp_path / "processed" / "dataset_de_lu.csv"
        meta_path = tmp_path / "processed" / "dataset_metadata.json"

        assert csv_path.exists()
        assert meta_path.exists()

        # Verify metadata is valid JSON with new fields
        with open(meta_path) as f:
            meta = json.load(f)
        assert "title" in meta
        assert "sources" in meta
        assert "neighbour_zones" in meta

    def test_raises_if_raw_missing(self, tmp_path: Path) -> None:
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2024], data_dir=tmp_path)
        cfg.ensure_dirs()
        with pytest.raises(FileNotFoundError):
            join_datasets(cfg)
