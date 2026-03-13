"""Tests for challenge.data."""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import pandas as pd

from energy_modelling.challenge.data import build_daily_challenge_frame, build_public_daily_dataset


def _make_selected_day_csv(path: Path) -> Path:
    days = [
        "2023-12-30",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
        "2025-01-01",
    ]
    chunks = [pd.date_range(day, periods=24, freq="h", tz="UTC") for day in days]
    timestamps = pd.DatetimeIndex(
        reduce(lambda left, right: left.union(right), chunks[1:], chunks[0])
    )

    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "price_eur_mwh": [50.0 + (i // 24) for i in range(len(timestamps))],
            "load_actual_mw": [40_000.0] * len(timestamps),
            "load_forecast_mw": [39_500.0] * len(timestamps),
            "forecast_solar_mw": [2_000.0] * len(timestamps),
            "forecast_wind_onshore_mw": [6_000.0] * len(timestamps),
            "forecast_wind_offshore_mw": [1_500.0] * len(timestamps),
            "gen_solar_mw": [1_800.0] * len(timestamps),
            "gen_wind_onshore_mw": [5_500.0] * len(timestamps),
            "gen_wind_offshore_mw": [1_200.0] * len(timestamps),
            "gen_fossil_gas_mw": [3_500.0] * len(timestamps),
            "gen_fossil_hard_coal_mw": [2_500.0] * len(timestamps),
            "gen_fossil_brown_coal_lignite_mw": [4_500.0] * len(timestamps),
            "gen_nuclear_mw": [0.0] * len(timestamps),
            "weather_temperature_2m_degc": [10.0] * len(timestamps),
            "weather_wind_speed_10m_kmh": [12.0] * len(timestamps),
            "weather_shortwave_radiation_wm2": [150.0] * len(timestamps),
            "price_fr_eur_mwh": [55.0] * len(timestamps),
            "price_nl_eur_mwh": [54.0] * len(timestamps),
            "price_at_eur_mwh": [53.0] * len(timestamps),
            "price_pl_eur_mwh": [56.0] * len(timestamps),
            "price_cz_eur_mwh": [52.0] * len(timestamps),
            "price_dk_1_eur_mwh": [51.0] * len(timestamps),
            "flow_fr_net_import_mw": [100.0] * len(timestamps),
            "flow_nl_net_import_mw": [75.0] * len(timestamps),
            "carbon_price_usd": [80.0] * len(timestamps),
            "gas_price_usd": [35.0] * len(timestamps),
        }
    )
    csv_path = path / "challenge_data.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def test_build_daily_challenge_frame_adds_labels_and_split(tmp_path: Path) -> None:
    csv_path = _make_selected_day_csv(tmp_path)
    daily = build_daily_challenge_frame(csv_path)

    assert "last_settlement_price" in daily.columns
    assert "settlement_price" in daily.columns
    assert "target_direction" in daily.columns
    assert set(daily["split"]) >= {"train", "validation", "hidden_test"}


def test_build_public_daily_dataset_excludes_hidden_test_rows(tmp_path: Path) -> None:
    csv_path = _make_selected_day_csv(tmp_path)
    daily = build_daily_challenge_frame(csv_path)
    public = build_public_daily_dataset(daily)

    assert "hidden_test" not in set(public["split"])
    assert set(public["split"]) == {"train", "validation"}
