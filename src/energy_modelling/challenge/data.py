"""Daily dataset preparation for the student strategy challenge."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from energy_modelling.market_simulation.data import (
    _FORECAST_FEATURE_COLS,
    _REALISED_FEATURE_COLS,
    build_daily_features,
    compute_daily_settlement,
    load_dataset,
)

PUBLIC_TRAIN_YEARS = frozenset({2019, 2020, 2021, 2022, 2023})
PUBLIC_VALIDATION_YEARS = frozenset({2024})
HIDDEN_TEST_YEARS = frozenset({2025})

LABEL_COLUMNS = (
    "settlement_price",
    "price_change_eur_mwh",
    "target_direction",
    "pnl_long_eur",
    "pnl_short_eur",
)

TIMING_CATEGORY_DESCRIPTIONS = {
    "label": "Target or score column. Not available to students on hidden test.",
    "lagged_realised": "Realised or observed data lagged by one day before use.",
    "same_day_forecast": "Day-ahead forecast available for the delivery day at decision time.",
    "reference": "Reference field such as split or prior settlement used by the runner.",
}


def _split_for_year(year: int) -> str:
    if year in PUBLIC_TRAIN_YEARS:
        return "train"
    if year in PUBLIC_VALIDATION_YEARS:
        return "validation"
    if year in HIDDEN_TEST_YEARS:
        return "hidden_test"
    return "unused"


def build_daily_challenge_frame(dataset_path: Path | str) -> pd.DataFrame:
    """Build a daily challenge table from the hourly market dataset.

    The returned table has one row per delivery date, starting from the first
    date that has a prior settlement. Feature timing follows the market
    simulation contract: realised data is lagged by one day, while day-ahead
    forecast columns remain aligned to the current delivery date.
    """

    hourly = load_dataset(dataset_path)
    settlements = compute_daily_settlement(hourly).astype(float)
    features = build_daily_features(hourly)

    daily = features.copy()
    daily.index.name = "delivery_date"
    daily.insert(0, "delivery_date", daily.index)

    lagged_settlement = settlements.shift(1)
    daily["last_settlement_price"] = lagged_settlement.reindex(daily.index).astype(float)
    daily["settlement_price"] = settlements.reindex(daily.index).astype(float)
    daily = daily.dropna(subset=["last_settlement_price", "settlement_price"])

    daily["price_change_eur_mwh"] = daily["settlement_price"] - daily["last_settlement_price"]
    daily["target_direction"] = np.sign(daily["price_change_eur_mwh"]).astype(int)
    daily["pnl_long_eur"] = daily["price_change_eur_mwh"] * 24.0
    daily["pnl_short_eur"] = -daily["pnl_long_eur"]
    daily["split"] = [_split_for_year(delivery_date.year) for delivery_date in daily.index]

    ordered = [
        "delivery_date",
        "split",
        "last_settlement_price",
        "settlement_price",
        "price_change_eur_mwh",
        "target_direction",
        "pnl_long_eur",
        "pnl_short_eur",
    ]
    feature_cols = [column for column in daily.columns if column not in ordered]
    return daily[ordered + feature_cols].copy()


def build_public_daily_dataset(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Return only public train and validation rows."""

    public = daily_data[daily_data["split"].isin({"train", "validation"})].copy()
    return public.reset_index(drop=True)


def build_feature_glossary(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Build a glossary classifying daily challenge columns by timing group."""

    lagged_realised = {f"{column}_mean" for column in _REALISED_FEATURE_COLS}
    same_day_forecast = {f"{column}_mean" for column in _FORECAST_FEATURE_COLS}
    lagged_realised.update({"price_mean", "price_max", "price_min", "price_std"})
    reference = {"delivery_date", "split", "last_settlement_price"}
    labels = set(LABEL_COLUMNS)

    rows = []
    for column in daily_data.columns:
        if column in labels:
            timing_group = "label"
        elif column in same_day_forecast:
            timing_group = "same_day_forecast"
        elif column in lagged_realised:
            timing_group = "lagged_realised"
        else:
            timing_group = "reference"

        rows.append(
            {
                "column": column,
                "timing_group": timing_group,
                "description": TIMING_CATEGORY_DESCRIPTIONS[timing_group],
            }
        )

    glossary = pd.DataFrame(rows)
    return glossary.sort_values(["timing_group", "column"]).reset_index(drop=True)


def strip_hidden_labels(daily_data: pd.DataFrame) -> pd.DataFrame:
    """Drop hidden target columns from a challenge frame."""

    hidden = daily_data.drop(columns=list(LABEL_COLUMNS), errors="ignore").copy()
    return hidden.reset_index(drop=True)


def write_challenge_data(
    dataset_path: Path | str,
    output_dir: Path | str,
    include_hidden_test: bool = False,
) -> dict[str, Path]:
    """Write public challenge data, and optionally private hidden-test data."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = build_daily_challenge_frame(dataset_path)
    public = build_public_daily_dataset(daily)
    public_path = output_dir / "daily_public.csv"
    glossary_path = output_dir / "daily_public_glossary.csv"
    public.to_csv(public_path, index=False)
    build_feature_glossary(daily).to_csv(glossary_path, index=False)

    written = {"public": public_path, "glossary": glossary_path}

    if include_hidden_test:
        hidden = daily[daily["split"] == "hidden_test"].copy()
        hidden_path = output_dir / "daily_hidden_test_full.csv"
        hidden.to_csv(hidden_path, index=False)
        written["hidden_test"] = hidden_path

    return written
