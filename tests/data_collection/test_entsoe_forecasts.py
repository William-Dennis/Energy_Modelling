"""Tests for data_collection.entsoe_forecasts module."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_forecasts import (
    _clean_forecast_columns,
    download_forecasts,
    fetch_forecasts_for_year,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_forecast_df() -> pd.DataFrame:
    """Hourly forecast data with MultiIndex columns."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="Europe/Berlin")
    data = {
        ("Solar", "Day Ahead"): [1000.0 + i for i in range(72)],
        ("Wind Onshore", "Day Ahead"): [5000.0 + i for i in range(72)],
        ("Wind Offshore", "Day Ahead"): [3000.0 + i for i in range(72)],
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def mock_client(mock_forecast_df: pd.DataFrame) -> MagicMock:
    """A mocked EntsoePandasClient."""
    client = MagicMock()
    client.query_wind_and_solar_forecast.return_value = mock_forecast_df
    return client


# ---------------------------------------------------------------------------
# Tests: _clean_forecast_columns
# ---------------------------------------------------------------------------


class TestCleanForecastColumns:
    def test_multiindex_flattened(self) -> None:
        arrays = [["Solar", "Wind Onshore"], ["Day Ahead", "Day Ahead"]]
        columns = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame([[1, 2]], columns=columns)
        result = _clean_forecast_columns(df)
        assert not isinstance(result.columns, pd.MultiIndex)
        assert all(isinstance(c, str) for c in result.columns)

    def test_flat_columns_normalised(self) -> None:
        df = pd.DataFrame({"Wind Onshore": [1], "Solar": [2]})
        result = _clean_forecast_columns(df)
        assert "wind_onshore" in result.columns
        assert "solar" in result.columns


# ---------------------------------------------------------------------------
# Tests: fetch_forecasts_for_year
# ---------------------------------------------------------------------------


class TestFetchForecastsForYear:
    def test_returns_dataframe(self, mock_client: MagicMock) -> None:
        df = fetch_forecasts_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert isinstance(df, pd.DataFrame)

    def test_index_is_utc(self, mock_client: MagicMock) -> None:
        df = fetch_forecasts_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_index_name(self, mock_client: MagicMock) -> None:
        df = fetch_forecasts_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert df.index.name == "timestamp_utc"

    def test_hourly_resolution(self, mock_client: MagicMock) -> None:
        df = fetch_forecasts_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_has_columns(self, mock_client: MagicMock) -> None:
        df = fetch_forecasts_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert len(df.columns) >= 1


# ---------------------------------------------------------------------------
# Tests: download_forecasts
# ---------------------------------------------------------------------------


class TestDownloadForecasts:
    def test_creates_parquet_files(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_forecasts.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_forecasts(cfg)

        assert (tmp_path / "raw" / "forecasts_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "forecasts.parquet"

    def test_skips_existing_year(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_forecasts.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        download_forecasts(cfg)
        download_forecasts(cfg)
        assert mock_client.query_wind_and_solar_forecast.call_count == 1

    def test_force_redownloads(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_forecasts.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        download_forecasts(cfg)
        download_forecasts(cfg, force=True)
        assert mock_client.query_wind_and_solar_forecast.call_count == 2
