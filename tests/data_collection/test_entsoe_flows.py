"""Tests for data_collection.entsoe_flows module."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_flows import (
    download_flows,
    fetch_flows_for_year,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_flow_series() -> pd.Series:
    """Generate a mock flow series."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="Europe/Berlin")
    return pd.Series([500.0 + i for i in range(72)], index=idx, name="flow")


@pytest.fixture()
def mock_client() -> MagicMock:
    """A mocked EntsoePandasClient that returns flow series."""
    client = MagicMock()
    client.query_crossborder_flows.return_value = _mock_flow_series()
    return client


# ---------------------------------------------------------------------------
# Tests: fetch_flows_for_year
# ---------------------------------------------------------------------------


class TestFetchFlowsForYear:
    def test_returns_dataframe(self, mock_client: MagicMock) -> None:
        df = fetch_flows_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin", ["FR"])
        assert isinstance(df, pd.DataFrame)

    def test_columns_for_single_neighbour(self, mock_client: MagicMock) -> None:
        df = fetch_flows_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin", ["FR"])
        assert "flow_fr_export_mw" in df.columns
        assert "flow_fr_import_mw" in df.columns
        assert "flow_fr_net_import_mw" in df.columns

    def test_net_import_is_import_minus_export(self, mock_client: MagicMock) -> None:
        df = fetch_flows_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin", ["FR"])
        expected = df["flow_fr_import_mw"] - df["flow_fr_export_mw"]
        pd.testing.assert_series_equal(df["flow_fr_net_import_mw"], expected, check_names=False)

    def test_index_is_utc(self, mock_client: MagicMock) -> None:
        df = fetch_flows_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin", ["FR"])
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_index_name(self, mock_client: MagicMock) -> None:
        df = fetch_flows_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin", ["FR"])
        assert df.index.name == "timestamp_utc"

    def test_handles_partial_failure(self, mock_client: MagicMock) -> None:
        """Should still return data if only one direction fails."""
        call_count = 0

        def _side_effect(from_zone: str, to_zone: str, **kw: object) -> pd.Series:
            nonlocal call_count
            call_count += 1
            if from_zone == "FR":
                raise RuntimeError("API error")
            return _mock_flow_series()

        mock_client.query_crossborder_flows.side_effect = _side_effect
        df = fetch_flows_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin", ["FR"])
        assert "flow_fr_export_mw" in df.columns
        # Import direction failed, so no import or net columns
        assert "flow_fr_import_mw" not in df.columns

    def test_empty_when_all_fail(self, mock_client: MagicMock) -> None:
        mock_client.query_crossborder_flows.side_effect = RuntimeError("fail")
        df = fetch_flows_for_year(mock_client, 2024, "DE_LU", "Europe/Berlin", ["FR"])
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: download_flows
# ---------------------------------------------------------------------------


class TestDownloadFlows:
    def test_creates_parquet_files(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_flows.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            neighbour_zones=["FR"],
        )
        result_path = download_flows(cfg)

        assert (tmp_path / "raw" / "flows_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "flows.parquet"

    def test_skips_existing_year(
        self, tmp_path: Path, mock_client: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "energy_modelling.data_collection.entsoe_flows.EntsoePandasClient",
            return_value=mock_client,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            neighbour_zones=["FR"],
        )
        download_flows(cfg)
        download_flows(cfg)
        # 2 calls first time (export + import), 0 second time
        assert mock_client.query_crossborder_flows.call_count == 2
