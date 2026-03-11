"""Tests for data_collection.entsoe_generation module."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_generation import (
    _clean_columns,
    download_generation,
    fetch_generation_for_year,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_generation_df(periods: int = 72, tz: str = "Europe/Berlin") -> pd.DataFrame:
    """Create a mock generation DataFrame mimicking entsoe-py output.

    Includes both Actual Aggregated and Actual Consumption columns to match
    the real ENTSO-E API response structure for DE-LU.
    """
    idx = pd.date_range("2024-01-01", periods=periods, freq="h", tz=tz)
    columns = pd.MultiIndex.from_tuples(
        [
            ("Wind Onshore", "Actual Aggregated"),
            ("Wind Onshore", "Actual Consumption"),
            ("Solar", "Actual Aggregated"),
            ("Solar", "Actual Consumption"),
            ("Fossil Gas", "Actual Aggregated"),
            ("Hydro Pumped Storage", "Actual Aggregated"),
            ("Hydro Pumped Storage", "Actual Consumption"),
            ("Nuclear", "Actual Aggregated"),
        ]
    )
    data = {col: [1000.0 + i * 10 + j for j in range(periods)] for i, col in enumerate(columns)}
    return pd.DataFrame(data, index=idx, columns=columns)


@pytest.fixture()
def mock_generation_df() -> pd.DataFrame:
    return _make_generation_df()


@pytest.fixture()
def mock_client(mock_generation_df: pd.DataFrame) -> MagicMock:
    client = MagicMock()
    client.query_generation.return_value = mock_generation_df
    return client


# ---------------------------------------------------------------------------
# Tests: _clean_columns
# ---------------------------------------------------------------------------


class TestCleanColumns:
    def test_flattens_multiindex(self, mock_generation_df: pd.DataFrame) -> None:
        """MultiIndex columns should be flattened to single level."""
        result = _clean_columns(mock_generation_df.copy())
        assert not isinstance(result.columns, pd.MultiIndex)

    def test_snake_case(self, mock_generation_df: pd.DataFrame) -> None:
        """Column names should be normalised to snake_case."""
        result = _clean_columns(mock_generation_df.copy())
        for col in result.columns:
            assert col == col.lower()
            assert " " not in col

    def test_expected_column_names(self, mock_generation_df: pd.DataFrame) -> None:
        """Should produce expected cleaned names for Actual Aggregated columns."""
        result = _clean_columns(mock_generation_df.copy())
        assert "wind_onshore" in result.columns
        assert "solar" in result.columns
        assert "fossil_gas" in result.columns
        assert "nuclear" in result.columns

    def test_consumption_columns_get_suffix(self, mock_generation_df: pd.DataFrame) -> None:
        """Actual Consumption sub-columns should get a _consumption suffix."""
        result = _clean_columns(mock_generation_df.copy())
        assert "wind_onshore_consumption" in result.columns
        assert "solar_consumption" in result.columns
        assert "hydro_pumped_storage_consumption" in result.columns

    def test_no_duplicate_columns(self, mock_generation_df: pd.DataFrame) -> None:
        """Cleaned columns must all be unique — no duplicates allowed."""
        result = _clean_columns(mock_generation_df.copy())
        assert len(result.columns) == len(set(result.columns))

    def test_handles_flat_columns(self) -> None:
        """Should handle already-flat string columns gracefully."""
        df = pd.DataFrame({"Wind Onshore": [1, 2], "Solar": [3, 4]})
        result = _clean_columns(df)
        assert "wind_onshore" in result.columns
        assert "solar" in result.columns


# ---------------------------------------------------------------------------
# Tests: fetch_generation_for_year
# ---------------------------------------------------------------------------


class TestFetchGenerationForYear:
    def test_returns_dataframe(self, mock_client: MagicMock) -> None:
        df = fetch_generation_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert isinstance(df, pd.DataFrame)

    def test_index_is_utc(self, mock_client: MagicMock) -> None:
        df = fetch_generation_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_index_name(self, mock_client: MagicMock) -> None:
        df = fetch_generation_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert df.index.name == "timestamp_utc"

    def test_columns_are_clean(self, mock_client: MagicMock) -> None:
        df = fetch_generation_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        for col in df.columns:
            assert col == col.lower()
            assert " " not in col

    def test_hourly_resolution(self, mock_client: MagicMock) -> None:
        df = fetch_generation_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_resamples_15min_to_hourly(self) -> None:
        """15-min data should be resampled to hourly means."""
        idx = pd.date_range("2024-01-01", periods=96, freq="15min", tz="Europe/Berlin")
        df_15min = pd.DataFrame({"Solar": range(96), "Wind": range(96)}, index=idx)
        client = MagicMock()
        client.query_generation.return_value = df_15min

        df = fetch_generation_for_year(client, 2024, "DE_LU", "Europe/Berlin")
        # 96 quarters = 24 hours
        assert len(df) == 24
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()


# ---------------------------------------------------------------------------
# Tests: download_generation
# ---------------------------------------------------------------------------


class TestDownloadGeneration:
    def test_creates_parquet_files(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_generation.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_generation(cfg)

        assert (tmp_path / "raw" / "generation_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "generation.parquet"

    def test_skips_existing_year(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_generation.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)

        download_generation(cfg)
        download_generation(cfg)

        assert mock_client.query_generation.call_count == 1

    def test_force_redownloads(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_generation.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)

        download_generation(cfg)
        download_generation(cfg, force=True)

        assert mock_client.query_generation.call_count == 2

    def test_consolidated_has_all_columns(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_generation.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_generation(cfg)

        df = pd.read_parquet(result_path)
        assert "wind_onshore" in df.columns
        assert "solar" in df.columns
        assert len(df) > 0

    def test_multiple_years(self, tmp_path: Path, mocker: MockerFixture) -> None:
        client = MagicMock()
        client.query_generation.side_effect = [
            _make_generation_df(48, "Europe/Berlin"),
            _make_generation_df(48, "Europe/Berlin"),
        ]
        mocker.patch(
            "energy_modelling.data_collection.entsoe_generation.EntsoePandasClient",
            return_value=client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2023, 2024], data_dir=tmp_path)
        result_path = download_generation(cfg)

        df = pd.read_parquet(result_path)
        assert len(df) > 0
        assert df.index.is_monotonic_increasing
