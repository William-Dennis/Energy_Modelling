"""Scan years backward from 2025 to assess data availability per source.

For each year, probes all 10 data sources:
  1.  ENTSO-E day-ahead prices (DE_LU)
  2.  ENTSO-E generation mix (DE_LU)
  3.  Open-Meteo ERA5 weather
  4.  ENTSO-E total load (actual + forecast)
  5.  ENTSO-E wind/solar DA forecasts
  6.  ENTSO-E neighbour zone day-ahead prices
  7.  ENTSO-E cross-border physical flows
  8.  ENTSO-E day-ahead NTC
  9.  Carbon price (CARB.L via Yahoo Finance)
  10. Gas price (TTF=F / NG=F via Yahoo Finance)

Saves per-year Parquet files for successful fetches and prints a summary table.
Stops when a year has severely limited data (prices AND generation both fail).

Usage:
    uv run python scripts/scan_years.py [--force]

Options:
    --force   Re-fetch even if per-year Parquet files already exist.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from entsoe import EntsoePandasClient
from loguru import logger

from energy_modelling.data_collection.carbon_price import fetch_carbon_price
from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_flows import fetch_flows_for_year
from energy_modelling.data_collection.entsoe_forecasts import fetch_forecasts_for_year
from energy_modelling.data_collection.entsoe_generation import _clean_columns
from energy_modelling.data_collection.entsoe_generation import _year_range as gen_year_range
from energy_modelling.data_collection.entsoe_load import fetch_load_for_year
from energy_modelling.data_collection.entsoe_neighbours import fetch_neighbour_prices_for_year
from energy_modelling.data_collection.entsoe_ntc import fetch_ntc_for_year
from energy_modelling.data_collection.entsoe_prices import _year_range as price_year_range
from energy_modelling.data_collection.gas_price import fetch_gas_price
from energy_modelling.data_collection.weather import fetch_weather_for_year

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_YEAR = 2025
END_YEAR = 2000  # will stop earlier if data is severely limited
BIDDING_ZONE = "DE_LU"
TIMEZONE = "Europe/Berlin"
NEIGHBOUR_ZONES = ["FR", "NL", "AT", "PL", "CZ", "DK_1", "DK_2", "BE", "SE_4"]

# Severity threshold: if a year has < this many price rows, it's "severely limited"
MIN_PRICE_ROWS = 4000  # ~6 months of hourly data


# ---------------------------------------------------------------------------
# Per-source probe functions
# ---------------------------------------------------------------------------


def probe_prices(
    client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False
) -> dict:
    """Try to fetch prices for a year. Returns a result dict."""
    year_path = raw_dir / f"prices_da_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": 1, "path": year_path}

    try:
        start, end = price_year_range(year, TIMEZONE)
        series = client.query_day_ahead_prices(BIDDING_ZONE, start=start, end=end)
        df = series.to_frame(name="price_eur_mwh")
        dt_index = pd.DatetimeIndex(df.index)
        df.index = dt_index.tz_convert("UTC")
        df.index.name = "timestamp_utc"
        df = df.resample("1h").mean()
        df.to_parquet(year_path, engine="pyarrow")
        return {"status": "ok", "rows": len(df), "cols": 1, "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


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
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

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
        return {"status": "ok", "rows": len(df), "cols": len(df.columns), "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_load(
    client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False
) -> dict:
    """Try to fetch load (actual + forecast) for a year. Returns a result dict."""
    year_path = raw_dir / f"load_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        df = fetch_load_for_year(client, year, BIDDING_ZONE, TIMEZONE)
        df.to_parquet(year_path, engine="pyarrow")
        return {"status": "ok", "rows": len(df), "cols": len(df.columns), "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_forecasts(
    client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False
) -> dict:
    """Try to fetch wind/solar DA forecasts for a year. Returns a result dict."""
    year_path = raw_dir / f"forecasts_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        df = fetch_forecasts_for_year(client, year, BIDDING_ZONE, TIMEZONE)
        df.to_parquet(year_path, engine="pyarrow")
        return {"status": "ok", "rows": len(df), "cols": len(df.columns), "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_neighbours(
    client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False
) -> dict:
    """Try to fetch neighbour zone DA prices for a year. Returns a result dict."""
    year_path = raw_dir / f"neighbour_prices_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        df = fetch_neighbour_prices_for_year(client, year, TIMEZONE, NEIGHBOUR_ZONES)
        df.to_parquet(year_path, engine="pyarrow")
        cols = len(df.columns) if not df.empty else 0
        return {"status": "ok", "rows": len(df), "cols": cols, "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_flows(
    client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False
) -> dict:
    """Try to fetch cross-border flows for a year. Returns a result dict."""
    year_path = raw_dir / f"flows_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        df = fetch_flows_for_year(client, year, BIDDING_ZONE, TIMEZONE, NEIGHBOUR_ZONES)
        df.to_parquet(year_path, engine="pyarrow")
        cols = len(df.columns) if not df.empty else 0
        return {"status": "ok", "rows": len(df), "cols": cols, "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_ntc(client: EntsoePandasClient, year: int, raw_dir: Path, *, force: bool = False) -> dict:
    """Try to fetch day-ahead NTC for a year. Returns a result dict."""
    year_path = raw_dir / f"ntc_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        df = fetch_ntc_for_year(client, year, BIDDING_ZONE, TIMEZONE, NEIGHBOUR_ZONES)
        df.to_parquet(year_path, engine="pyarrow")
        cols = len(df.columns) if not df.empty else 0
        return {"status": "ok", "rows": len(df), "cols": cols, "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_carbon(year: int, raw_dir: Path, *, force: bool = False) -> dict:
    """Try to fetch carbon price (CARB.L) for a year. Returns a result dict."""
    year_path = raw_dir / f"carbon_price_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        df = fetch_carbon_price(f"{year}-01-01", f"{year + 1}-01-01")
        df.to_parquet(year_path, engine="pyarrow")
        cols = len(df.columns) if not df.empty else 0
        return {"status": "ok", "rows": len(df), "cols": cols, "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


def probe_gas(year: int, raw_dir: Path, *, force: bool = False) -> dict:
    """Try to fetch gas price (TTF=F / NG=F) for a year. Returns a result dict."""
    year_path = raw_dir / f"gas_price_{year}.parquet"
    if year_path.exists() and not force:
        df = pd.read_parquet(year_path)
        return {"status": "cached", "rows": len(df), "cols": len(df.columns), "path": year_path}

    try:
        df = fetch_gas_price(f"{year}-01-01", f"{year + 1}-01-01")
        df.to_parquet(year_path, engine="pyarrow")
        cols = len(df.columns) if not df.empty else 0
        return {"status": "ok", "rows": len(df), "cols": cols, "path": year_path}
    except Exception as e:
        return {"status": "error", "rows": 0, "cols": 0, "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _probe_label(result: dict) -> str:
    """Format a compact status label: ok/cached/error + rows[xcols]."""
    status = result["status"]
    rows = result["rows"]
    cols = result.get("cols", "?")
    if status == "error":
        return "ERROR"
    return f"{status}({rows}r/{cols}c)"


def main(force: bool = False) -> None:
    config = DataCollectionConfig()
    config.ensure_dirs()
    raw_dir = config.raw_dir
    client = EntsoePandasClient(api_key=config.entsoe_api_key)

    results: list[dict] = []

    for year in range(START_YEAR, END_YEAR - 1, -1):
        logger.info("=" * 70)
        logger.info("Probing year {}", year)
        logger.info("=" * 70)

        # --- Mandatory sources ---
        logger.info("  [prices] fetching...")
        price_r = probe_prices(client, year, raw_dir, force=force)
        logger.info(
            "  [prices] {} rows={} cols={}", price_r["status"], price_r["rows"], price_r.get("cols")
        )
        if price_r["status"] == "ok":
            time.sleep(1)

        logger.info("  [generation] fetching... (slow ~2-3 min)")
        gen_r = probe_generation(client, year, raw_dir, force=force)
        logger.info(
            "  [generation] {} rows={} cols={}", gen_r["status"], gen_r["rows"], gen_r.get("cols")
        )
        if gen_r["status"] == "ok":
            time.sleep(1)

        logger.info("  [weather] fetching...")
        weather_r = probe_weather(year, raw_dir, force=force)
        logger.info(
            "  [weather] {} rows={} cols={}",
            weather_r["status"],
            weather_r["rows"],
            weather_r.get("cols"),
        )
        if weather_r["status"] == "ok":
            time.sleep(0.5)

        # Stop early if mandatory sources are both missing
        if price_r["status"] == "error" and gen_r["status"] == "error":
            logger.warning("STOPPING: Year {} — no prices and no generation data", year)
            results.append(
                {
                    "year": year,
                    "price": price_r,
                    "gen": gen_r,
                    "weather": weather_r,
                    "load": {"status": "skipped", "rows": 0, "cols": 0},
                    "forecasts": {"status": "skipped", "rows": 0, "cols": 0},
                    "neighbours": {"status": "skipped", "rows": 0, "cols": 0},
                    "flows": {"status": "skipped", "rows": 0, "cols": 0},
                    "ntc": {"status": "skipped", "rows": 0, "cols": 0},
                    "carbon": {"status": "skipped", "rows": 0, "cols": 0},
                    "gas": {"status": "skipped", "rows": 0, "cols": 0},
                }
            )
            break

        if price_r["status"] != "error" and price_r["rows"] < MIN_PRICE_ROWS:
            logger.warning(
                "STOPPING: Year {} has only {} price rows (< {} threshold)",
                year,
                price_r["rows"],
                MIN_PRICE_ROWS,
            )
            results.append(
                {
                    "year": year,
                    "price": price_r,
                    "gen": gen_r,
                    "weather": weather_r,
                    "load": {"status": "skipped", "rows": 0, "cols": 0},
                    "forecasts": {"status": "skipped", "rows": 0, "cols": 0},
                    "neighbours": {"status": "skipped", "rows": 0, "cols": 0},
                    "flows": {"status": "skipped", "rows": 0, "cols": 0},
                    "ntc": {"status": "skipped", "rows": 0, "cols": 0},
                    "carbon": {"status": "skipped", "rows": 0, "cols": 0},
                    "gas": {"status": "skipped", "rows": 0, "cols": 0},
                }
            )
            break

        # --- Optional ENTSO-E sources ---
        logger.info("  [load] fetching...")
        load_r = probe_load(client, year, raw_dir, force=force)
        logger.info(
            "  [load] {} rows={} cols={}", load_r["status"], load_r["rows"], load_r.get("cols")
        )
        if load_r["status"] == "ok":
            time.sleep(1)

        logger.info("  [forecasts] fetching...")
        forecasts_r = probe_forecasts(client, year, raw_dir, force=force)
        logger.info(
            "  [forecasts] {} rows={} cols={}",
            forecasts_r["status"],
            forecasts_r["rows"],
            forecasts_r.get("cols"),
        )
        if forecasts_r["status"] == "ok":
            time.sleep(1)

        logger.info("  [neighbours] fetching...")
        neighbours_r = probe_neighbours(client, year, raw_dir, force=force)
        logger.info(
            "  [neighbours] {} rows={} cols={}",
            neighbours_r["status"],
            neighbours_r["rows"],
            neighbours_r.get("cols"),
        )
        if neighbours_r["status"] == "ok":
            time.sleep(1)

        logger.info("  [flows] fetching...")
        flows_r = probe_flows(client, year, raw_dir, force=force)
        logger.info(
            "  [flows] {} rows={} cols={}", flows_r["status"], flows_r["rows"], flows_r.get("cols")
        )
        if flows_r["status"] == "ok":
            time.sleep(1)

        logger.info("  [ntc] fetching...")
        ntc_r = probe_ntc(client, year, raw_dir, force=force)
        logger.info("  [ntc] {} rows={} cols={}", ntc_r["status"], ntc_r["rows"], ntc_r.get("cols"))
        if ntc_r["status"] == "ok":
            time.sleep(1)

        # --- Yahoo Finance sources (no ENTSO-E client needed) ---
        logger.info("  [carbon] fetching...")
        carbon_r = probe_carbon(year, raw_dir, force=force)
        logger.info(
            "  [carbon] {} rows={} cols={}",
            carbon_r["status"],
            carbon_r["rows"],
            carbon_r.get("cols"),
        )

        logger.info("  [gas] fetching...")
        gas_r = probe_gas(year, raw_dir, force=force)
        logger.info("  [gas] {} rows={} cols={}", gas_r["status"], gas_r["rows"], gas_r.get("cols"))

        results.append(
            {
                "year": year,
                "price": price_r,
                "gen": gen_r,
                "weather": weather_r,
                "load": load_r,
                "forecasts": forecasts_r,
                "neighbours": neighbours_r,
                "flows": flows_r,
                "ntc": ntc_r,
                "carbon": carbon_r,
                "gas": gas_r,
            }
        )

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    logger.info("\n" + "=" * 100)
    logger.info("DATA AVAILABILITY SUMMARY  (rows/cols per source)")
    logger.info("=" * 100)

    sources = [
        "price",
        "gen",
        "weather",
        "load",
        "forecasts",
        "neighbours",
        "flows",
        "ntc",
        "carbon",
        "gas",
    ]
    header = f"{'Year':<6} " + " ".join(f"{s:<22}" for s in sources)
    logger.info(header)
    logger.info("-" * 100)

    for r in sorted(results, key=lambda x: x["year"]):
        line = f"{r['year']:<6} " + " ".join(f"{_probe_label(r[s]):<22}" for s in sources)
        logger.info(line)
        # Print errors on next line
        for s in sources:
            if "error" in r[s]:
                logger.info(f"  [{s}] error: {r[s]['error'][:120]}")

    # Valid years: prices and generation both OK, sufficient rows
    valid_years = sorted(
        [
            r["year"]
            for r in results
            if r["price"]["status"] in ("ok", "cached")
            and r["gen"]["status"] in ("ok", "cached")
            and r["price"]["rows"] >= MIN_PRICE_ROWS
        ]
    )
    logger.info("\nValid years for dataset (prices + generation OK): {}", valid_years)

    # Per-source summary
    logger.info("\nPer-source valid years:")
    for s in sources:
        ok_years = sorted(
            [r["year"] for r in results if r[s]["status"] in ("ok", "cached") and r[s]["rows"] > 0]
        )
        logger.info("  {:12s}: {}", s, ok_years)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch even if cached Parquet files exist"
    )
    args = parser.parse_args()
    main(force=args.force)
