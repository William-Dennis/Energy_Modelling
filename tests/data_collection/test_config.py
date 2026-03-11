"""Tests for data_collection.config module."""

from pathlib import Path

import pytest

from energy_modelling.data_collection.config import DataCollectionConfig


class TestDataCollectionConfig:
    """Tests for DataCollectionConfig."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults for DE-LU zone."""
        cfg = DataCollectionConfig(entsoe_api_key="test-key")
        assert cfg.bidding_zone == "DE_LU"
        assert cfg.timezone == "Europe/Berlin"
        assert cfg.years == [2024]
        assert cfg.weather_latitude == 51.5
        assert cfg.weather_longitude == 10.5
        assert cfg.data_dir == Path("data")

    def test_api_key_required_empty_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key defaults to empty string (validated at usage time, not config time)."""
        monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
        cfg = DataCollectionConfig(_env_file=None)  # type: ignore[call-arg]
        assert cfg.entsoe_api_key == ""

    def test_custom_years(self) -> None:
        """Should accept a custom list of years."""
        cfg = DataCollectionConfig(entsoe_api_key="k", years=[2020, 2021, 2022])
        assert cfg.years == [2020, 2021, 2022]

    def test_raw_dir_property(self) -> None:
        """raw_dir should be data_dir / 'raw'."""
        cfg = DataCollectionConfig(entsoe_api_key="k", data_dir=Path("/tmp/test"))
        assert cfg.raw_dir == Path("/tmp/test/raw")

    def test_processed_dir_property(self) -> None:
        """processed_dir should be data_dir / 'processed'."""
        cfg = DataCollectionConfig(entsoe_api_key="k", data_dir=Path("/tmp/test"))
        assert cfg.processed_dir == Path("/tmp/test/processed")

    def test_ensure_dirs_creates_directories(self, tmp_path: Path) -> None:
        """ensure_dirs should create raw and processed subdirectories."""
        cfg = DataCollectionConfig(entsoe_api_key="k", data_dir=tmp_path / "mydata")
        cfg.ensure_dirs()
        assert (tmp_path / "mydata" / "raw").is_dir()
        assert (tmp_path / "mydata" / "processed").is_dir()

    def test_weather_variables_default(self) -> None:
        """Should include standard weather variables for energy forecasting."""
        cfg = DataCollectionConfig(entsoe_api_key="k")
        expected = {
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_speed_100m",
            "shortwave_radiation",
            "direct_normal_irradiance",
            "precipitation",
        }
        assert set(cfg.weather_variables) == expected

    def test_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should load API key from environment variable."""
        monkeypatch.setenv("ENTSOE_API_KEY", "env-test-key")
        cfg = DataCollectionConfig()
        assert cfg.entsoe_api_key == "env-test-key"
