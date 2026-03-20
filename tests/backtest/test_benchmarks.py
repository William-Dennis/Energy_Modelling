"""Tests for entry-price benchmarks and their integration with run_backtest."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.benchmarks import (
    ALL_BENCHMARKS,
    biased_settlement,
    get_benchmark,
    noisy_settlement,
    perfect_foresight_price,
    yesterday_settlement,
)
from energy_modelling.backtest.runner import run_backtest
from energy_modelling.backtest.types import BacktestState, BacktestStrategy


def _make_daily_frame() -> pd.DataFrame:
    index = [
        date(2023, 12, 30),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2024, 1, 2),
    ]
    return pd.DataFrame(
        {
            "delivery_date": index,
            "split": ["train", "train", "validation", "validation"],
            "last_settlement_price": [50.0, 51.0, 52.0, 53.0],
            "settlement_price": [51.0, 52.0, 53.5, 52.0],
            "price_change_eur_mwh": [1.0, 1.0, 1.5, -1.0],
            "target_direction": [1, 1, 1, -1],
            "pnl_long_eur": [24.0, 24.0, 36.0, -24.0],
            "pnl_short_eur": [-24.0, -24.0, -36.0, 24.0],
            "load_actual_mw_mean": [40_000.0, 41_000.0, 42_000.0, 43_000.0],
            "price_mean": [50.0, 51.0, 52.0, 53.0],
        }
    )


class _AlwaysLong(BacktestStrategy):
    def act(self, state: BacktestState) -> int | None:
        return 1


# ── Factory unit tests ──────────────────────────────────────────────


def test_yesterday_settlement_shape_and_index() -> None:
    df = _make_daily_frame()
    series = yesterday_settlement(df)
    assert isinstance(series, pd.Series)
    assert len(series) == len(df)
    for idx in series.index:
        assert isinstance(idx, date)


def test_noisy_settlement_differs_from_baseline() -> None:
    df = _make_daily_frame()
    base = yesterday_settlement(df)
    noisy = noisy_settlement(df, std_eur=5.0, seed=42)
    assert len(noisy) == len(base)
    assert not base.equals(noisy)


def test_noisy_settlement_reproducible() -> None:
    df = _make_daily_frame()
    a = noisy_settlement(df, std_eur=5.0, seed=42)
    b = noisy_settlement(df, std_eur=5.0, seed=42)
    pd.testing.assert_series_equal(a, b)


def test_biased_settlement_exact_shift() -> None:
    df = _make_daily_frame()
    base = yesterday_settlement(df)
    bias = 7.5
    biased = biased_settlement(df, bias_eur=bias)
    pd.testing.assert_series_equal(
        biased, (base + bias).rename("biased_settlement"), check_names=True
    )


def test_perfect_foresight_matches_settlement() -> None:
    df = _make_daily_frame()
    pf = perfect_foresight_price(df)
    expected = df.set_index("delivery_date")["settlement_price"].astype(float)
    pd.testing.assert_series_equal(
        pf.reset_index(drop=True),
        expected.reset_index(drop=True).rename("perfect_foresight"),
        check_names=True,
    )


# ── get_benchmark tests ─────────────────────────────────────────────


def test_get_benchmark_all_ids() -> None:
    df = _make_daily_frame()
    for benchmark_id in ALL_BENCHMARKS:
        series = get_benchmark(benchmark_id, df)
        assert isinstance(series, pd.Series)
        assert len(series) > 0


def test_get_benchmark_unknown_raises() -> None:
    df = _make_daily_frame()
    with pytest.raises(ValueError, match="Unknown benchmark"):
        get_benchmark("does_not_exist", df)


# ── Integration with run_backtest ────────────────────────────────────


def test_run_backtest_with_custom_entry_prices() -> None:
    """Custom entry_prices should change PnL compared to the default."""
    df = _make_daily_frame()

    baseline_result = run_backtest(
        strategy=_AlwaysLong(),
        daily_data=df,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )

    # Use a constant bias so entry prices differ from last_settlement_price
    biased_prices = biased_settlement(df, bias_eur=5.0)
    biased_result = run_backtest(
        strategy=_AlwaysLong(),
        daily_data=df,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
        entry_prices=biased_prices,
    )

    assert not baseline_result.daily_pnl.equals(biased_result.daily_pnl)


def test_run_backtest_baseline_entry_prices_match_default() -> None:
    """Passing yesterday_settlement as entry_prices should match default PnL."""
    df = _make_daily_frame()

    default_result = run_backtest(
        strategy=_AlwaysLong(),
        daily_data=df,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )

    baseline_prices = yesterday_settlement(df)
    explicit_result = run_backtest(
        strategy=_AlwaysLong(),
        daily_data=df,
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
        entry_prices=baseline_prices,
    )

    pd.testing.assert_series_equal(default_result.daily_pnl, explicit_result.daily_pnl)
