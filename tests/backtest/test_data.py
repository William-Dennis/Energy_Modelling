"""Tests for challenge.data."""

from __future__ import annotations

from datetime import date
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_modelling.backtest.data import (
    build_daily_backtest_frame,
    build_feature_glossary,
    build_public_daily_dataset,
)


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


def test_build_daily_backtest_frame_adds_labels_and_split(tmp_path: Path) -> None:
    csv_path = _make_selected_day_csv(tmp_path)
    daily = build_daily_backtest_frame(csv_path)

    assert "last_settlement_price" in daily.columns
    assert "settlement_price" in daily.columns
    assert "target_direction" in daily.columns
    assert set(daily["split"]) >= {"train", "validation", "hidden_test"}


def test_build_public_daily_dataset_excludes_hidden_test_rows(tmp_path: Path) -> None:
    csv_path = _make_selected_day_csv(tmp_path)
    daily = build_daily_backtest_frame(csv_path)
    public = build_public_daily_dataset(daily)

    assert "hidden_test" not in set(public["split"])
    assert set(public["split"]) == {"train", "validation"}


def test_backtest_frame_keeps_target_day_forecasts_but_lags_realised_values(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2023-12-31", periods=72, freq="h", tz="UTC")
    day_numbers = np.repeat([1.0, 2.0, 3.0], 24)
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "price_eur_mwh": day_numbers * 10.0,
            "load_actual_mw": day_numbers * 100.0,
            "load_forecast_mw": day_numbers * 1000.0,
            "forecast_solar_mw": day_numbers * 2000.0,
            "forecast_wind_onshore_mw": day_numbers * 3000.0,
            "forecast_wind_offshore_mw": day_numbers * 4000.0,
            "weather_temperature_2m_degc": day_numbers * 5.0,
        }
    )
    csv_path = tmp_path / "timing_case.csv"
    frame.to_csv(csv_path, index=False)

    daily = build_daily_backtest_frame(csv_path)
    row = daily.loc[daily["delivery_date"] == date(2024, 1, 1)].iloc[0]

    assert row["last_settlement_price"] == pytest.approx(10.0)
    assert row["load_actual_mw_mean"] == pytest.approx(100.0)
    assert row["weather_temperature_2m_degc_mean"] == pytest.approx(5.0)
    assert row["load_forecast_mw_mean"] == pytest.approx(2000.0)
    assert row["forecast_solar_mw_mean"] == pytest.approx(4000.0)


def test_build_daily_backtest_frame_applies_hourly_cleaning(tmp_path: Path) -> None:
    """Cleaning pipeline removes NaN artefacts before daily aggregation.

    We inject NaN values into columns that ``clean_hourly_data`` targets:
    - first row all-NaN artefact (dropped)
    - an interconnector column with NaN (zero-filled)
    - a commodity column with NaN (interpolated)
    - load_forecast_mw with a NaN gap (24 h-prior fill)

    After ``build_daily_backtest_frame`` the resulting daily features must
    contain **no** NaN values, proving cleaning runs inside the pipeline.
    """
    # 3 full days so we get at least 1 usable daily row after lagging
    timestamps = pd.date_range("2023-12-30", periods=72, freq="h", tz="UTC")
    n = len(timestamps)

    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "price_eur_mwh": [50.0] * n,
            "load_actual_mw": [40_000.0] * n,
            "load_forecast_mw": [39_500.0] * n,
            "forecast_solar_mw": [2_000.0] * n,
            "forecast_wind_onshore_mw": [6_000.0] * n,
            "forecast_wind_offshore_mw": [1_500.0] * n,
            "gen_solar_mw": [1_800.0] * n,
            "gen_wind_onshore_mw": [5_500.0] * n,
            "gen_wind_offshore_mw": [1_200.0] * n,
            "gen_fossil_gas_mw": [3_500.0] * n,
            "gen_fossil_hard_coal_mw": [2_500.0] * n,
            "gen_fossil_brown_coal_lignite_mw": [4_500.0] * n,
            "gen_nuclear_mw": [0.0] * n,
            "weather_temperature_2m_degc": [10.0] * n,
            "weather_wind_speed_10m_kmh": [12.0] * n,
            "weather_shortwave_radiation_wm2": [150.0] * n,
            "price_fr_eur_mwh": [55.0] * n,
            "price_nl_eur_mwh": [54.0] * n,
            "price_at_eur_mwh": [53.0] * n,
            "price_pl_eur_mwh": [56.0] * n,
            "price_cz_eur_mwh": [52.0] * n,
            "price_dk_1_eur_mwh": [51.0] * n,
            "flow_fr_net_import_mw": [100.0] * n,
            "flow_nl_net_import_mw": [75.0] * n,
            "carbon_price_usd": [80.0] * n,
            "gas_price_usd": [35.0] * n,
            # Interconnector columns with NaN (should be zero-filled)
            "ntc_dk_2_export_mw": [np.nan] * n,
            "ntc_nl_export_mw": [np.nan] * n,
        }
    )

    # Inject NaN into commodity column (hour 25 = second day, hour 1)
    frame.loc[25, "carbon_price_usd"] = np.nan

    # Inject NaN into load_forecast_mw (hour 30 = second day, hour 6)
    frame.loc[30, "load_forecast_mw"] = np.nan

    csv_path = tmp_path / "dirty_data.csv"
    frame.to_csv(csv_path, index=False)

    daily = build_daily_backtest_frame(csv_path)

    # The frame should have rows and NO NaN feature values
    assert len(daily) > 0, "Expected at least one daily row"
    feature_cols = [c for c in daily.columns if c not in ("delivery_date", "split")]
    nan_counts = daily[feature_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    assert cols_with_nan.empty, f"NaN found after cleaning: {cols_with_nan.to_dict()}"


def test_feature_glossary_classifies_timing_groups(tmp_path: Path) -> None:
    csv_path = _make_selected_day_csv(tmp_path)
    daily = build_daily_backtest_frame(csv_path)
    glossary = build_feature_glossary(daily)

    assert (
        glossary.loc[glossary["column"] == "load_actual_mw_mean", "timing_group"].iloc[0]
        == "lagged_realised"
    )
    assert (
        glossary.loc[glossary["column"] == "load_forecast_mw_mean", "timing_group"].iloc[0]
        == "same_day_forecast"
    )
    assert glossary.loc[glossary["column"] == "settlement_price", "timing_group"].iloc[0] == "label"
