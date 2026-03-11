"""Download historical ERA5 weather data from the Open-Meteo Archive API.

Uses the Open-Meteo Historical Weather API (``archive-api.open-meteo.com``)
which serves ERA5 reanalysis data.  **No API key required.**

Data is fetched as JSON, converted to a pandas DataFrame, and saved as
Parquet — one file per year plus a consolidated file.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import requests
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def _build_params(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str],
) -> dict[str, Any]:
    """Build query parameters for the Open-Meteo Archive API."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }


def _parse_response(response_json: dict[str, Any], variables: list[str]) -> pd.DataFrame:
    """Parse the Open-Meteo JSON response into a DataFrame.

    Parameters
    ----------
    response_json:
        Decoded JSON from the API response.
    variables:
        List of variable names that were requested.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC ``DatetimeIndex`` named ``"timestamp_utc"``
        and one column per weather variable.
    """
    hourly = response_json["hourly"]
    timestamps = pd.to_datetime(hourly["time"], utc=True)

    data: dict[str, list[float | None]] = {}
    for var in variables:
        data[var] = hourly[var]

    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "timestamp_utc"
    return df


def fetch_weather_for_year(
    year: int,
    latitude: float,
    longitude: float,
    variables: list[str],
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch hourly ERA5 weather data for a single year.

    Parameters
    ----------
    year:
        Calendar year to fetch.
    latitude:
        Grid point latitude.
    longitude:
        Grid point longitude.
    variables:
        List of Open-Meteo variable names.
    session:
        Optional requests session (for caching / retries).

    Returns
    -------
    pd.DataFrame
        Hourly weather data with UTC timestamps.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    params = _build_params(latitude, longitude, start_date, end_date, variables)
    logger.info("Fetching weather for year={} at ({}, {})", year, latitude, longitude)

    http = session or requests.Session()
    resp = http.get(OPEN_METEO_URL, params=params, timeout=120)
    resp.raise_for_status()

    df = _parse_response(resp.json(), variables)
    logger.info("  -> {} rows for year {}", len(df), year)
    return df


def download_weather(
    config: DataCollectionConfig,
    *,
    force: bool = False,
) -> Path:
    """Download weather data for all configured years and save to Parquet.

    Parameters
    ----------
    config:
        Pipeline configuration.
    force:
        If *True*, re-download even if the file already exists.

    Returns
    -------
    Path
        Path to the consolidated Parquet file.
    """
    config.ensure_dirs()
    consolidated_path = config.raw_dir / "weather.parquet"
    frames: list[pd.DataFrame] = []

    session = requests.Session()

    for year in sorted(config.years):
        year_path = config.raw_dir / f"weather_{year}.parquet"

        if year_path.exists() and not force:
            logger.info("Skipping year {} — {} already exists", year, year_path)
            frames.append(pd.read_parquet(year_path))
            continue

        df = fetch_weather_for_year(
            year,
            config.weather_latitude,
            config.weather_longitude,
            config.weather_variables,
            session=session,
        )
        df.to_parquet(year_path, engine="pyarrow")
        logger.info("Saved {}", year_path)
        frames.append(df)

    # Consolidate
    all_weather = pd.concat(frames).sort_index()
    all_weather = all_weather[~all_weather.index.duplicated(keep="first")]
    all_weather.to_parquet(consolidated_path, engine="pyarrow")
    logger.info("Consolidated weather -> {} ({} rows)", consolidated_path, len(all_weather))
    return consolidated_path
