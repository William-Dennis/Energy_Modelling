"""Tests for data_collection.entsoe_neighbours module."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_neighbours import (
    download_neighbour_prices,
    fetch_neighbour_prices_for_year,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_price_series(zone: str) -> pd.Series:
    """Generate a mock price series for a zone."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="Europe/Berlin")
    return pd.Series([50.0 + i for i in range(72)], index=idx, name=zone)


@pytest.fixture()
def mock_client() -> MagicMock:
    """A mocked EntsoePandasClient that returns price series for any zone."""
    client = MagicMock()
    client.query_day_ahead_prices.side_effect = lambda zone, **kw: _mock_price_series(zone)
    return client


# ---------------------------------------------------------------------------
# Tests: fetch_neighbour_prices_for_year
# ---------------------------------------------------------------------------


class TestFetchNeighbourPricesForYear:
    def test_returns_dataframe(self, mock_client: MagicMock) -> None:
        df = fetch_neighbour_prices_for_year(mock_client, 2024, "Europe/Berlin", ["FR", "NL"])
        assert isinstance(df, pd.DataFrame)

    def test_columns_named_correctly(self, mock_client: MagicMock) -> None:
        df = fetch_neighbour_prices_for_year(mock_client, 2024, "Europe/Berlin", ["FR", "NL"])
        assert "price_fr_eur_mwh" in df.columns
        assert "price_nl_eur_mwh" in df.columns

    def test_index_is_utc(self, mock_client: MagicMock) -> None:
        df = fetch_neighbour_prices_for_year(mock_client, 2024, "Europe/Berlin", ["FR"])
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_index_name(self, mock_client: MagicMock) -> None:
        df = fetch_neighbour_prices_for_year(mock_client, 2024, "Europe/Berlin", ["FR"])
        assert df.index.name == "timestamp_utc"

    def test_handles_api_failure_gracefully(self, mock_client: MagicMock) -> None:
        """Should skip zones that fail and still return data for successful ones."""
        call_count = 0

        def _side_effect(zone: str, **kw: object) -> pd.Series:
            nonlocal call_count
            call_count += 1
            if zone == "FR":
                raise RuntimeError("API error")
            return _mock_price_series(zone)

        mock_client.query_day_ahead_prices.side_effect = _side_effect
        df = fetch_neighbour_prices_for_year(mock_client, 2024, "Europe/Berlin", ["FR", "NL"])
        assert "price_nl_eur_mwh" in df.columns
        assert "price_fr_eur_mwh" not in df.columns

    def test_empty_when_all_fail(self, mock_client: MagicMock) -> None:
        mock_client.query_day_ahead_prices.side_effect = RuntimeError("fail")
        df = fetch_neighbour_prices_for_year(mock_client, 2024, "Europe/Berlin", ["FR", "NL"])
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: download_neighbour_prices
# ---------------------------------------------------------------------------


class TestDownloadNeighbourPrices:
    def test_creates_parquet_files(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_neighbours.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            neighbour_zones=["FR", "NL"],
        )
        result_path = download_neighbour_prices(cfg)

        assert (tmp_path / "raw" / "neighbour_prices_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "neighbour_prices.parquet"

    def test_skips_existing_year(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_neighbours.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            neighbour_zones=["FR"],
        )
        download_neighbour_prices(cfg)
        download_neighbour_prices(cfg)
        # Only called once for one zone
        assert mock_client.query_day_ahead_prices.call_count == 1
