"""Tests for submission strategy baselines and dashboard discovery."""

from __future__ import annotations

from datetime import date

import pandas as pd

from energy_modelling.backtest.runner import run_backtest
from energy_modelling.dashboard._backtest import _discover_submission_strategies
from strategies.always_long import AlwaysLongStrategy
from strategies.always_short import AlwaysShortStrategy


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


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


def test_discover_finds_always_long() -> None:
    factories, descriptions = _discover_submission_strategies()
    assert "Always Long" in factories


def test_discover_finds_always_short() -> None:
    factories, descriptions = _discover_submission_strategies()
    assert "Always Short" in factories


def test_discover_returns_all_strategies() -> None:
    factories, _ = _discover_submission_strategies()
    assert len(factories) == 55


def test_discover_returns_descriptions() -> None:
    _, descriptions = _discover_submission_strategies()
    assert descriptions["Always Long"]
    assert descriptions["Always Short"]


# ---------------------------------------------------------------------------
# AlwaysLong tests
# ---------------------------------------------------------------------------


def test_always_long_returns_positive_one() -> None:
    daily = _make_daily_frame()
    result = run_backtest(
        strategy=AlwaysLongStrategy(),
        daily_data=daily,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    assert set(result.predictions.dropna().astype(int).unique()) == {1}


def test_always_long_trades_every_day() -> None:
    daily = _make_daily_frame()
    result = run_backtest(
        strategy=AlwaysLongStrategy(),
        daily_data=daily,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    assert result.trade_count == 2
    assert result.days_evaluated == 2


# ---------------------------------------------------------------------------
# AlwaysShort tests
# ---------------------------------------------------------------------------


def test_always_short_returns_negative_one() -> None:
    daily = _make_daily_frame()
    result = run_backtest(
        strategy=AlwaysShortStrategy(),
        daily_data=daily,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    assert set(result.predictions.dropna().astype(int).unique()) == {-1}


def test_always_short_trades_every_day() -> None:
    daily = _make_daily_frame()
    result = run_backtest(
        strategy=AlwaysShortStrategy(),
        daily_data=daily,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    assert result.trade_count == 2
    assert result.days_evaluated == 2


# ---------------------------------------------------------------------------
# Symmetry: long + short PnL cancel out
# ---------------------------------------------------------------------------


def test_long_short_pnl_symmetry() -> None:
    daily = _make_daily_frame()
    long_result = run_backtest(
        strategy=AlwaysLongStrategy(),
        daily_data=daily,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    short_result = run_backtest(
        strategy=AlwaysShortStrategy(),
        daily_data=daily,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )
    total = long_result.daily_pnl.values + short_result.daily_pnl.values
    assert all(abs(v) < 1e-10 for v in total)
