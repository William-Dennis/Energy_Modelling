"""Tests for data_collection.gas_price module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.gas_price import (
    download_gas_price,
    fetch_gas_price,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_yf_history() -> pd.DataFrame:
    """Generate mock yfinance history data."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    return pd.DataFrame(
        {
            "Open": [25.0 + i for i in range(30)],
            "High": [27.0 + i for i in range(30)],
            "Low": [23.0 + i for i in range(30)],
            "Close": [26.0 + i for i in range(30)],
            "Volume": [5000] * 30,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests: fetch_gas_price
# ---------------------------------------------------------------------------


class TestFetchGasPrice:
    def test_returns_dataframe(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            return_value=mock_ticker,
        ):
            df = fetch_gas_price("2024-01-01", "2025-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_column_name(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            return_value=mock_ticker,
        ):
            df = fetch_gas_price("2024-01-01", "2025-01-01")
        assert "gas_price_usd" in df.columns

    def test_index_is_utc(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            return_value=mock_ticker,
        ):
            df = fetch_gas_price("2024-01-01", "2025-01-01")
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"

    def test_hourly_forward_filled(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        with patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            return_value=mock_ticker,
        ):
            df = fetch_gas_price("2024-01-01", "2025-01-01")
        assert len(df) > 30

    def test_falls_back_on_first_ticker_failure(self) -> None:
        """If first ticker fails, should try the next."""
        call_count = 0

        def _ticker_factory(name: str) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            if name == "TTF=F":
                mock.history.return_value = pd.DataFrame()  # empty = no data
            else:
                mock.history.return_value = _mock_yf_history()
            return mock

        with patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            side_effect=_ticker_factory,
        ):
            df = fetch_gas_price("2024-01-01", "2025-01-01")
        assert "gas_price_usd" in df.columns
        assert len(df) > 0

    def test_empty_when_all_fail(self) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        with patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            return_value=mock_ticker,
        ):
            df = fetch_gas_price("2024-01-01", "2025-01-01")
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests: download_gas_price
# ---------------------------------------------------------------------------


class TestDownloadGasPrice:
    def test_creates_parquet_files(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        mocker.patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            return_value=mock_ticker,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        result_path = download_gas_price(cfg)

        assert (tmp_path / "raw" / "gas_price_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "gas_price.parquet"

    def test_skips_existing_year(self, tmp_path: Path, mocker: MockerFixture) -> None:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _mock_yf_history()
        mock_yf = mocker.patch(
            "energy_modelling.data_collection.gas_price.yf.Ticker",
            return_value=mock_ticker,
        )
        cfg = DataCollectionConfig(entsoe_api_key="test", years=[2024], data_dir=tmp_path)
        download_gas_price(cfg)
        download_gas_price(cfg)
        # First call creates the ticker; second run skips entirely
        assert mock_yf.call_count == 1
