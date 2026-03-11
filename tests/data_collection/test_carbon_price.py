"""Tests for data_collection.carbon_price module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from pytest_mock import MockerFixture

from energy_modelling.data_collection.carbon_price import (
    download_carbon_price,
    fetch_carbon_price,
)
from energy_modelling.data_collection.config import DataCollectionConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_yf_history() -> pd.DataFrame:
    """Generate mock yfinance history data."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")  # Business days
    return pd.DataFrame(
        {
            "Open": [70.0 + i for i in range(30)],
            "High": [72.0 + i for i in range(30)],
            "Low": [68.0 + i for i in range(30)],
            "Close": [71.0 + i for i in range(30)],
            "Volume": [1000] * 30,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests: fetch_carbon_price
# ---------------------------------------------------------------------------


class TestFetchCarbonPrice:
    def test_returns_dataframe(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker", return_value=mock_ticker
        ):
            df = fetch_carbon_price("2024-01-01", "2025-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_column_name(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker", return_value=mock_ticker
        ):
            df = fetch_carbon_price("2024-01-01", "2025-01-01")
        assert "carbon_price_usd" in df.columns

    def test_index_is_utc(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker", return_value=mock_ticker
        ):
            df = fetch_carbon_price("2024-01-01", "2025-01-01")
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_hourly_forward_filled(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker", return_value=mock_ticker
        ):
            df = fetch_carbon_price("2024-01-01", "2025-01-01")
        # Should have more rows than daily (forward-filled to hourly)
        assert len(df) > 30

    def test_empty_when_no_data(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker", return_value=mock_ticker
        ):
            df = fetch_carbon_price("2024-01-01", "2025-01-01")
        assert "carbon_price_usd" in df.columns
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: download_carbon_price
# ---------------------------------------------------------------------------


class TestDownloadCarbonPrice:
    def test_creates_parquet_files(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        mocker.patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker",
            return_value=mock_ticker,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_carbon_price(cfg)

        assert (tmp_path / "raw" / "carbon_price_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "carbon_price.parquet"

    def test_skips_existing_year(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        mock_yf = mocker.patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker",
            return_value=mock_ticker,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        download_carbon_price(cfg)
        download_carbon_price(cfg)
        assert mock_yf.call_count == 1

    def test_force_redownloads(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        mock_yf = mocker.patch(
            "energy_modelling.data_collection.carbon_price.yf.Ticker",
            return_value=mock_ticker,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        download_carbon_price(cfg)
        download_carbon_price(cfg, force=True)
        assert mock_yf.call_count == 2
