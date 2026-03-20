"""Tests for challenge.runner."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_modelling.backtest.runner import run_backtest
from energy_modelling.backtest.types import BacktestState, BacktestStrategy


def _make_daily_frame() -> pd.DataFrame:
    index = [
        date(2023, 12, 30),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2024, 1, 2),
    ]
    frame = pd.DataFrame(
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
    return frame


class _TrackingStrategy(BacktestStrategy):
    def __init__(self) -> None:
        self.fit_rows = 0
        self.history_lengths: list[int] = []

    def fit(self, train_data: pd.DataFrame) -> None:
        self.fit_rows = len(train_data)

    def forecast(self, state: BacktestState) -> float:
        self.history_lengths.append(len(state.history))
        direction = 1 if state.features["load_actual_mw_mean"] >= 42_000.0 else -1
        return state.last_settlement_price + direction * 1.0


def test_runner_fits_on_train_rows_and_uses_prior_history() -> None:
    strategy = _TrackingStrategy()
    result = run_backtest(
        strategy=strategy,
        daily_data=_make_daily_frame(),
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )

    assert strategy.fit_rows == 2
    assert strategy.history_lengths == [2, 3]
    assert result.trade_count == 2
    assert result.days_evaluated == 2


def test_runner_computes_daily_pnl() -> None:
    strategy = _TrackingStrategy()
    result = run_backtest(
        strategy=strategy,
        daily_data=_make_daily_frame(),
        training_end=date(2023, 12, 31),
        evaluation_start=date(2024, 1, 1),
        evaluation_end=date(2024, 1, 2),
    )

    assert result.daily_pnl.loc[date(2024, 1, 1)] == pytest.approx(36.0)
    assert result.daily_pnl.loc[date(2024, 1, 2)] == pytest.approx(-24.0)
    assert result.metrics["total_pnl"] == pytest.approx(12.0)


class _BadStrategy(BacktestStrategy):
    # Overrides act() to return an invalid value (2) for validation testing.
    def act(self, state: BacktestState) -> int | None:
        return 2  # type: ignore[return-value]

    def forecast(self, state: BacktestState) -> float:
        return state.last_settlement_price


def test_runner_rejects_invalid_prediction() -> None:
    with pytest.raises(ValueError, match="invalid prediction"):
        run_backtest(
            strategy=_BadStrategy(),
            daily_data=_make_daily_frame(),
            training_end=date(2023, 12, 31),
            evaluation_start=date(2024, 1, 1),
            evaluation_end=date(2024, 1, 2),
        )
