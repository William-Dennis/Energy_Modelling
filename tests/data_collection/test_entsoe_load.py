"""Tests for data_collection.entsoe_load module."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_load import (
    download_load,
    fetch_load_for_year,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_load_series() -> pd.Series:
    """Hourly actual load series for a small date range."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="Europe/Berlin")
    return pd.Series([40000.0 + i * 10 for i in range(72)], index=idx, name="load")


@pytest.fixture()
def mock_forecast_series() -> pd.Series:
    """Hourly forecast load series for a small date range."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="Europe/Berlin")
    return pd.Series([39000.0 + i * 10 for i in range(72)], index=idx, name="forecast")


@pytest.fixture()
def mock_client(mock_load_series: pd.Series, mock_forecast_series: pd.Series) -> MagicMock:
    """A mocked EntsoePandasClient."""
    client = MagicMock()
    client.query_load.return_value = mock_load_series
    client.query_load_forecast.return_value = mock_forecast_series
    return client


# ---------------------------------------------------------------------------
# Tests: fetch_load_for_year
# ---------------------------------------------------------------------------


class TestFetchLoadForYear:
    def test_returns_dataframe(self, mock_client: MagicMock) -> None:
        df = fetch_load_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, mock_client: MagicMock) -> None:
        df = fetch_load_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert "load_actual_mw" in df.columns
        assert "load_forecast_mw" in df.columns

    def test_index_is_utc(self, mock_client: MagicMock) -> None:
        df = fetch_load_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_index_name(self, mock_client: MagicMock) -> None:
        df = fetch_load_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert df.index.name == "timestamp_utc"

    def test_hourly_resolution(self, mock_client: MagicMock) -> None:
        df = fetch_load_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_handles_dataframe_return(self, mock_client: MagicMock) -> None:
        """Should handle case where query_load returns a DataFrame."""
        idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="Europe/Berlin")
        mock_client.query_load.return_value = pd.DataFrame(
            {"Actual Load": [40000.0] * 72}, index=idx
        )
        df = fetch_load_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert "load_actual_mw" in df.columns


# ---------------------------------------------------------------------------
# Tests: download_load
# ---------------------------------------------------------------------------


class TestDownloadLoad:
    def test_creates_parquet_files(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_load.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_load(cfg)

        assert (tmp_path / "raw" / "load_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "load.parquet"

    def test_skips_existing_year(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_load.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        download_load(cfg)
        download_load(cfg)
        assert mock_client.query_load.call_count == 1

    def test_force_redownloads(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_load.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        download_load(cfg)
        download_load(cfg, force=True)
        assert mock_client.query_load.call_count == 2

    def test_consolidated_readable(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_load.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_load(cfg)
        df = pd.read_parquet(result_path)
        assert "load_actual_mw" in df.columns
        assert "load_forecast_mw" in df.columns
        assert len(df) > 0
