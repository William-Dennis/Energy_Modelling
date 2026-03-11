"""Scan years backward from 2025 to assess data availability per source.

For each year, probes:
  1. ENTSO-E day-ahead prices (DE_LU)
  2. ENTSO-E generation mix (DE_LU)
  3. Open-Meteo ERA5 weather

Saves per-year Parquet files for successful fetches and prints a summary table.
Stops when a year has severely limited data (prices AND generation both fail).

Usage:
    uv run python scripts/scan_years.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_generation import _clean_columns
from energy_modelling.data_collection.entsoe_generation import _year_range as gen_year_range
from energy_modelling.data_collection.entsoe_prices import _year_range as price_year_range
from energy_modelling.data_collection.weather import fetch_weather_for_year

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_YEAR = 2025
END_YEAR = 2000  # will stop earlier if data is severely limited
BIDDING_ZONE = "DE_LU"
TIMEZONE = "Europe/Berlin"

# Severity threshold: if a year has < this many price rows, it's "severely limited"
MIN_PRICE_ROWS = 4000  # ~6 months of hourly data


def probe_prices(
    client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False
) -> dict:
    """Try to fetch prices for a year. Returns a result dict."""
    year_path = raw_dir / f"prices_da_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "path": year_path}

    try:
        start, end = price_year_range(year, TIMEZONE)
        series = client.query_day_ahead_prices(BIDDING_ZONE, start=start, end=end)
        df = series.to_frame(name="price_eur_mwh")
        dt_index = pd.DatetimeIndex(df.index)
        df.index = dt_index.tz_convert("UTC")
        df.index.name = "timestamp_utc"
        df = df.resample("1h").mean()
        df.to_parquet(year_path, engine="pyarrow")
        return {"status": "ok", "rows": len(df), "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "error": str(e)[:200]}


def probe_generation(
    client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False
) -> dict:
    """Try to fetch generation for a year. Returns a result dict."""
    year_path = raw_dir / f"generation_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        start, end = gen_year_range(year, TIMEZONE)
        df = client.query_generation(BIDDING_ZONE, start=start, end=end)
        df = _clean_columns(df)
        dt_index = pd.DatetimeIndex(df.index)
        df.index = dt_index.tz_convert("UTC")
        df.index.name = "timestamp_utc"
        df = df.resample("1h").mean()
        df.to_parquet(year_path, engine="pyarrow")
        return {"status": "ok", "rows": len(df), "cols": len(df.columns), "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_weather(year: int, raw_dir: Path, *, force: bool = False) -> dict:
    """Try to fetch weather for a year. Returns a result dict."""
    year_path = raw_dir / f"weather_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "path": year_path}

    try:
        df = fetch_weather_for_year(
            year,
            51.5,
            10.5,
            [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_speed_100m",
                "shortwave_radiation",
                "direct_normal_irradiance",
                "precipitation",
            ],
        )
        df.to_parquet(year_path, engine="pyarrow")
        return {"status": "ok", "rows": len(df), "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "error": str(e)[:200]}


def main() -> None:
    config = DataCollectionConfig()
    config.ensure_dirs()
    raw_dir = config.raw_dir
    client = EntsoePandasClient(api_key=config.entsoe_api_key)

    results: list[dict] = []

    for year in range(START_YEAR, END_YEAR - 1, -1):
        logger.info("=" * 60)
        logger.info("Probing year {}", year)
        logger.info("=" * 60)

        # Prices
        logger.info("  [prices] ...")
        price_result = probe_prices(client, year, raw_dir)
        logger.info("  [prices] {} -> {} rows", price_result["status"], price_result["rows"])
        # Be nice to the API
        if price_result["status"] == "ok":
            time.sleep(1)

        # Generation
        logger.info("  [generation] ...")
        gen_result = probe_generation(client, year, raw_dir)
        logger.info(
            "  [generation] {} -> {} rows, {} cols",
            gen_result["status"],
            gen_result["rows"],
            gen_result.get("cols", "?"),
        )
        if gen_result["status"] == "ok":
            time.sleep(1)

        # Weather
        logger.info("  [weather] ...")
        weather_result = probe_weather(year, raw_dir)
        logger.info("  [weather] {} -> {} rows", weather_result["status"], weather_result["rows"])
        if weather_result["status"] == "ok":
            time.sleep(0.5)

        row = {
            "year": year,
            "prices_status": price_result["status"],
            "prices_rows": price_result["rows"],
            "gen_status": gen_result["status"],
            "gen_rows": gen_result["rows"],
            "gen_cols": gen_result.get("cols", 0),
            "weather_status": weather_result["status"],
            "weather_rows": weather_result["rows"],
        }
        # Add error messages if any
        if "error" in price_result:
            row["prices_error"] = price_result["error"]
        if "error" in gen_result:
            row["gen_error"] = gen_result["error"]
        if "error" in weather_result:
            row["weather_error"] = weather_result["error"]

        results.append(row)

        # Stop condition: both prices AND generation failed
        if price_result["status"] == "error" and gen_result["status"] == "error":
            logger.warning("STOPPING: Year {} has no prices and no generation data", year)
            break

        # Stop condition: prices exist but are severely limited
        if price_result["status"] != "error" and price_result["rows"] < MIN_PRICE_ROWS:
            logger.warning(
                "STOPPING: Year {} has only {} price rows (< {} threshold)",
                year,
                price_result["rows"],
                MIN_PRICE_ROWS,
            )
            break

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("DATA AVAILABILITY SUMMARY")
    logger.info("=" * 80)
    header = (
        f"{'Year':<6} {'Prices':<10} {'P.Rows':<8} "
        f"{'Gen':<10} {'G.Rows':<8} {'G.Cols':<7} "
        f"{'Weather':<10} {'W.Rows':<8}"
    )
    logger.info(header)
    logger.info("-" * 80)
    for r in sorted(results, key=lambda x: x["year"]):
        line = (
            f"{r['year']:<6} {r['prices_status']:<10} {r['prices_rows']:<8} "
            f"{r['gen_status']:<10} {r['gen_rows']:<8} {r['gen_cols']:<7} "
            f"{r['weather_status']:<10} {r['weather_rows']:<8}"
        )
        logger.info(line)
        if "prices_error" in r:
            logger.info(f"  prices_error: {r['prices_error']}")
        if "gen_error" in r:
            logger.info(f"  gen_error: {r['gen_error']}")
        if "weather_error" in r:
            logger.info(f"  weather_error: {r['weather_error']}")

    # Determine valid years
    valid_years = sorted(
        [
            r["year"]
            for r in results
            if r["prices_status"] in ("ok", "cached")
            and r["gen_status"] in ("ok", "cached")
            and r["prices_rows"] >= MIN_PRICE_ROWS
        ]
    )
    logger.info("\nValid years for dataset: {}", valid_years)


if __name__ == "__main__":
    main()
