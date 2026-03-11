"""Configuration for the data collection pipeline.

Uses pydantic-settings to load from .env files and environment variables.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class DataCollectionConfig(BaseSettings):
    """Configuration for the data collection pipeline.

    Loads ENTSOE_API_KEY from .env file or environment variables.
    All other fields have sensible defaults for the DE-LU bidding zone.
    """

    # --- API Keys ---
    entsoe_api_key: str = ""

    # --- Temporal scope ---
    years: list[int] = [2024]

    # --- Market zone ---
    bidding_zone: str = "DE_LU"
    timezone: str = "Europe/Berlin"

    # --- Weather grid point (central Germany) ---
    weather_latitude: float = 51.5
    weather_longitude: float = 10.5
    weather_variables: list[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_speed_100m",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "precipitation",
    ]

    # --- Storage ---
    data_dir: Path = Path("data")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def raw_dir(self) -> Path:
        """Directory for raw downloaded data."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory for processed/joined data."""
        return self.data_dir / "processed"

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
