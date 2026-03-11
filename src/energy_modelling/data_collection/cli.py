"""CLI entry point for the data collection pipeline.

Usage::

    uv run collect-data --years 2023 2024
    uv run collect-data --years 2024 --step prices
    uv run collect-data --step join --kaggle
    uv run python -m energy_modelling.data_collection --years 2024
"""

from pathlib import Path

import click
from loguru import logger

from energy_modelling.data_collection.config import DataCollectionConfig
from energy_modelling.data_collection.entsoe_generation import download_generation
from energy_modelling.data_collection.entsoe_prices import download_prices
from energy_modelling.data_collection.join import join_datasets
from energy_modelling.data_collection.weather import download_weather

STEPS = ("prices", "generation", "weather", "join", "all")


@click.command()
@click.option(
    "--years",
    "-y",
    multiple=True,
    type=int,
    help="Years to fetch (e.g. --years 2023 --years 2024). Defaults to config.",
)
@click.option(
    "--step",
    "-s",
    type=click.Choice(STEPS),
    default="all",
    help="Which pipeline step to run. Default: all.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Force re-download even if files exist.",
)
@click.option(
    "--kaggle",
    is_flag=True,
    default=False,
    help="Export Kaggle-ready CSV and metadata JSON alongside the join step.",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the data directory (default: data/).",
)
def main(
    years: tuple[int, ...],
    step: str,
    force: bool,
    kaggle: bool,
    data_dir: Path | None,
) -> None:
    """Download and join energy market data for DE-LU.

    Fetches day-ahead prices, generation mix, and weather data, then
    joins them into a single hourly dataset.
    """
    # Build config
    config_kwargs: dict = {}
    if years:
        config_kwargs["years"] = list(years)
    if data_dir is not None:
        config_kwargs["data_dir"] = data_dir

    config = DataCollectionConfig(**config_kwargs)

    logger.info("Data collection pipeline starting")
    logger.info("  Zone: {}", config.bidding_zone)
    logger.info("  Years: {}", config.years)
    logger.info("  Data dir: {}", config.data_dir)
    logger.info("  Step: {}", step)

    if step in ("prices", "all"):
        logger.info("--- Downloading day-ahead prices ---")
        download_prices(config, force=force)

    if step in ("generation", "all"):
        logger.info("--- Downloading generation mix ---")
        download_generation(config, force=force)

    if step in ("weather", "all"):
        logger.info("--- Downloading weather data ---")
        download_weather(config, force=force)

    if step in ("join", "all"):
        logger.info("--- Joining datasets ---")
        output = join_datasets(config, kaggle=kaggle)
        logger.info("Final dataset: {}", output)

    logger.info("Done.")
