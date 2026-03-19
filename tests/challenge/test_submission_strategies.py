"""Tests for example submission strategies and dashboard discovery."""

from __future__ import annotations

from datetime import date

import pandas as pd

from energy_modelling.challenge.runner import run_challenge_backtest
from energy_modelling.dashboard._challenge import _discover_submission_strategies
from submission.student_strategy import StudentStrategy
from submission.tiny_ml_strategy import TinyMLStrategy


def _make_daily_frame() -> pd.DataFrame:
    rows = []
    for idx, delivery_date in enumerate(
        [
            date(2023, 12, 28),
            date(2023, 12, 29),
            date(2023, 12, 30),
            date(2023, 12, 31),
            date(2024, 1, 1),
            date(2024, 1, 2),
        ]
    ):
        last_settlement = 50.0 + idx
        target_direction = 1 if idx % 2 == 0 else -1
        settlement = last_settlement + target_direction * 1.5
        rows.append(
            {
                "delivery_date": delivery_date,
                "split": "train" if delivery_date.year == 2023 else "validation",
                "last_settlement_price": last_settlement,
                "settlement_price": settlement,
                "price_change_eur_mwh": settlement - last_settlement,
                "target_direction": target_direction,
                "pnl_long_eur": (settlement - last_settlement) * 24.0,
                "pnl_short_eur": (last_settlement - settlement) * 24.0,
                "load_forecast_mw_mean": 40_000.0 + idx * 500.0,
                "load_actual_mw_mean": 39_500.0 + idx * 400.0,
                "forecast_solar_mw_mean": 1_500.0 + idx * 50.0,
                "forecast_wind_onshore_mw_mean": 6_000.0 - idx * 100.0,
                "price_fr_eur_mwh_mean": last_settlement + 2.0,
                "gas_price_usd_mean": 30.0 + idx,
                "price_mean": last_settlement,
            }
        )
    return pd.DataFrame(rows)


def test_submission_dashboard_discovers_all_strategy_modules() -> None:
    factories, descriptions = _discover_submission_strategies()

    assert "Student" in factories
    assert "Tiny ML" in factories
    assert "Price Level Mean Reversion" in factories
    assert descriptions["Student"]


def test_student_strategy_runs_on_public_validation() -> None:
    result = run_challenge_backtest(
        strategy=StudentStrategy(),
        daily_data=_make_daily_frame(),
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )

    assert result.trade_count == 2
    assert result.days_evaluated == 2


def test_tiny_ml_strategy_fits_and_returns_valid_predictions() -> None:
    result = run_challenge_backtest(
        strategy=TinyMLStrategy(),
        daily_data=_make_daily_frame(),
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )

    assert set(result.predictions.dropna().astype(int).unique()).issubset({-1, 1})
