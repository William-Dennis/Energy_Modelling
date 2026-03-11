"""Tests for data_collection.weather module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
from pytest_mock import MockerFixture

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.weather import (
    _build_params,
    _parse_response,
    download_weather,
    fetch_weather_for_year,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_VARIABLES = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]


def _make_api_response(hours: int = 72) -> dict[str, Any]:
    """Create a mock Open-Meteo JSON response."""
    times = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    return {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": [5.0 + i * 0.1 for i in range(hours)],
            "wind_speed_10m": [3.0 + i * 0.05 for i in range(hours)],
            "shortwave_radiation": [0.0 + i * 1.0 for i in range(hours)],
        }
    }


def _mock_session(response_json: dict[str, Any]) -> MagicMock:
    """Create a mock requests.Session whose .get() returns the given JSON."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_json
    mock_resp.raise_for_status.return_value = None

    session = MagicMock()
    session.get.return_value = mock_resp
    return session


# ---------------------------------------------------------------------------
# Tests: _build_params
# ---------------------------------------------------------------------------


class TestBuildParams:
    def test_contains_all_keys(self) -> None:
        params = _build_params(51.5, 10.5, "2024-01-01", "2024-12-31", SAMPLE_VARIABLES)
        assert params["latitude"] == 51.5
        assert params["longitude"] == 10.5
        assert params["start_date"] == "2024-01-01"
        assert params["end_date"] == "2024-12-31"
        assert params["timezone"] == "UTC"

    def test_hourly_is_comma_separated(self) -> None:
        params = _build_params(51.5, 10.5, "2024-01-01", "2024-12-31", SAMPLE_VARIABLES)
        assert params["hourly"] == "temperature_2m,wind_speed_10m,shortwave_radiation"


# ---------------------------------------------------------------------------
# Tests: _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_returns_dataframe(self) -> None:
        resp = _make_api_response(48)
        df = _parse_response(resp, SAMPLE_VARIABLES)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self) -> None:
        resp = _make_api_response(48)
        df = _parse_response(resp, SAMPLE_VARIABLES)
        assert list(df.columns) == SAMPLE_VARIABLES

    def test_utc_index(self) -> None:
        resp = _make_api_response(48)
        df = _parse_response(resp, SAMPLE_VARIABLES)
        assert str(pd.DatetimeIndex(df.index).tz) == "UTC"
        assert df.index.name == "timestamp_utc"

    def test_row_count(self) -> None:
        resp = _make_api_response(48)
        df = _parse_response(resp, SAMPLE_VARIABLES)
        assert len(df) == 48


# ---------------------------------------------------------------------------
# Tests: fetch_weather_for_year
# ---------------------------------------------------------------------------


class TestFetchWeatherForYear:
    def test_returns_dataframe(self) -> None:
        session = _mock_session(_make_api_response(72))
        df = fetch_weather_for_year(2024, 51.5, 10.5, SAMPLE_VARIABLES, session=session)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 72

    def test_calls_correct_url(self) -> None:
        session = _mock_session(_make_api_response())
        fetch_weather_for_year(2024, 51.5, 10.5, SAMPLE_VARIABLES, session=session)
        session.get.assert_called_once()
        call_args = session.get.call_args
        assert "archive-api.open-meteo.com" in call_args[0][0]

    def test_date_range_in_params(self) -> None:
        session = _mock_session(_make_api_response())
        fetch_weather_for_year(2024, 51.5, 10.5, SAMPLE_VARIABLES, session=session)
        params = session.get.call_args[1]["params"]
        assert params["start_date"] == "2024-01-01"
        assert params["end_date"] == "2024-12-31"


# ---------------------------------------------------------------------------
# Tests: download_weather
# ---------------------------------------------------------------------------


class TestDownloadWeather:
    def test_creates_parquet_files(self, tmp_path: Path, mocker: MockerFixture) -> None:
        session = _mock_session(_make_api_response(72))
        mocker.patch(
            "energy_modelling.data_collection.weather.requests.Session",
            return_value=session,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            weather_variables=SAMPLE_VARIABLES,
        )
        result_path = download_weather(cfg)

        assert (tmp_path / "raw" / "weather_2024.parquet").exists()
        assert result_path.exists()
        assert result_path.name == "weather.parquet"

    def test_skips_existing_year(self, tmp_path: Path, mocker: MockerFixture) -> None:
        session = _mock_session(_make_api_response(72))
        mocker.patch(
            "energy_modelling.data_collection.weather.requests.Session",
            return_value=session,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            weather_variables=SAMPLE_VARIABLES,
        )

        download_weather(cfg)
        download_weather(cfg)

        # Session.get should only be called once (first run)
        assert session.get.call_count == 1

    def test_force_redownloads(self, tmp_path: Path, mocker: MockerFixture) -> None:
        session = _mock_session(_make_api_response(72))
        mocker.patch(
            "energy_modelling.data_collection.weather.requests.Session",
            return_value=session,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            weather_variables=SAMPLE_VARIABLES,
        )

        download_weather(cfg)
        download_weather(cfg, force=True)

        assert session.get.call_count == 2

    def test_consolidated_is_readable(self, tmp_path: Path, mocker: MockerFixture) -> None:
        session = _mock_session(_make_api_response(72))
        mocker.patch(
            "energy_modelling.data_collection.weather.requests.Session",
            return_value=session,
        )
        cfg = DataCollectionConfig(
            entsoe_api_key="test",
            years=[2024],
            data_dir=tmp_path,
            weather_variables=SAMPLE_VARIABLES,
        )
        result_path = download_weather(cfg)

        df = pd.read_parquet(result_path)
        assert "temperature_2m" in df.columns
        assert len(df) == 72
