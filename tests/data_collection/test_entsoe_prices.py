"""Tests for data_collection.entsoe_prices module."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_prices import (
    _year_range,
    download_prices,
    fetch_prices_for_year,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_price_series() -> pd.Series:
    """Hourly price series for a small date range (3 days in Berlin tz)."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="Europe/Berlin")
    return pd.Series([50.0 + i for i in range(72)], index=idx, name="price")


@pytest.fixture()
def mock_client(mock_price_series: pd.Series) -> MagicMock:
    """A mocked EntsoePandasClient that returns ``mock_price_series``."""
    client = MagicMock()
    client.query_day_ahead_prices.return_value = mock_price_series
    return client


# ---------------------------------------------------------------------------
# Tests: _year_range
# ---------------------------------------------------------------------------


class TestYearRange:
    def test_boundaries(self) -> None:
        start, end = _year_range(2024, "Europe/Berlin")
        assert start == pd.Timestamp("2024-01-01", tz="Europe/Berlin")
        assert end == pd.Timestamp("2025-01-01", tz="Europe/Berlin")

    def test_utc_timezone(self) -> None:
        start, end = _year_range(2023, "UTC")
        assert start.tzinfo is not None
        assert start.year == 2023
        assert end.year == 2024


# ---------------------------------------------------------------------------
# Tests: fetch_prices_for_year
# ---------------------------------------------------------------------------


class TestFetchPricesForYear:
    def test_returns_dataframe(self, mock_client: MagicMock) -> None:
        """Should return a DataFrame, not a Series."""
        df = fetch_prices_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert isinstance(df, pd.DataFrame)

    def test_column_name(self, mock_client: MagicMock) -> None:
        """Column must be 'price_eur_mwh'."""
        df = fetch_prices_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert "price_eur_mwh" in df.columns

    def test_index_is_utc(self, mock_client: MagicMock) -> None:
        """Index should be converted to UTC."""
        df = fetch_prices_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_index_name(self, mock_client: MagicMock) -> None:
        """Index name should be 'timestamp_utc'."""
        df = fetch_prices_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        assert df.index.name == "timestamp_utc"

    def test_hourly_resolution(self, mock_client: MagicMock) -> None:
        """All consecutive rows should be 1 hour apart."""
        df = fetch_prices_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()

    def test_client_called_with_correct_args(self, mock_client: MagicMock) -> None:
        """Underlying client should be called with correct zone and date range."""
        fetch_prices_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin")
        mock_client.query_day_ahead_prices.assert_called_once()
        call_args = mock_client.query_day_ahead_prices.call_args
        assert call_args[0][0] == "DE_LU"


# ---------------------------------------------------------------------------
# Tests: download_prices (integration with filesystem, mocked API)
# ---------------------------------------------------------------------------


class TestDownloadPrices:
    def test_creates_parquet_files(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        """Should create per-year and consolidated Parquet files."""
        mocker.patch(
            "energy_modelling.data_collection.entsoe_prices.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_prices(cfg)

        assert (tmp_path / "raw" / "prices_da_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "prices_da.parquet"

    def test_skips_existing_year(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        """Should skip download when the per-year file already exists."""
        mocker.patch(
            "energy_modelling.data_collection.entsoe_prices.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)

        # First download
        download_prices(cfg)
        # Second download — should skip
        download_prices(cfg)

        # Client should only have been called once (first run)
        assert mock_client.query_day_ahead_prices.call_count == 1

    def test_force_redownloads(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        """force=True should re-download even if file exists."""
        mocker.patch(
            "energy_modelling.data_collection.entsoe_prices.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)

        download_prices(cfg)
        download_prices(cfg, force=True)

        assert mock_client.query_day_ahead_prices.call_count == 2

    def test_consolidated_file_is_readable(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        """The consolidated Parquet file should be readable as a DataFrame."""
        mocker.patch(
            "energy_modelling.data_collection.entsoe_prices.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_prices(cfg)

        df = pd.read_parquet(result_path)
        assert "price_eur_mwh" in df.columns
        assert len(df) > 0

    def test_multiple_years_consolidated(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Consolidation should merge multiple years into one sorted file."""
        # Create mock data for 2023 and 2024
        idx_2023 = pd.date_range("2023-06-01", periods=48, freq="h", tz="Europe/Berlin")
        idx_2024 = pd.date_range("2024-06-01", periods=48, freq="h", tz="Europe/Berlin")

        client = MagicMock()
        client.query_day_ahead_prices.side_effect = [
            pd.Series(40.0, index=idx_2023),
            pd.Series(60.0, index=idx_2024),
        ]
        mocker.patch(
            "energy_modelling.data_collection.entsoe_prices.EntsoePandasClient",
            return_value=client,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2023, 2024], data_dir=tmp_path)
        result_path = download_prices(cfg)

        df = pd.read_parquet(result_path)
        assert len(df) == 96  # 48 + 48
        # Verify sorted
        assert df.index.is_monotonic_increasing
