"""Tests for scoring helpers -- base metrics, market-adjusted, and analysis."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from energy_modelling.challenge.scoring import (
    compute_challenge_metrics,
    compute_market_adjusted_metrics,
    leaderboard_score,
    market_leaderboard_score,
    monthly_pnl,
    rolling_sharpe,
)


class TestComputeMarketAdjustedMetrics:
    def test_includes_base_metrics(self) -> None:
        market_pnl = pd.Series([100.0, -50.0, 80.0])
        original_pnl = pd.Series([120.0, -60.0, 90.0])
        metrics = compute_market_adjusted_metrics(market_pnl, original_pnl, trade_count=3)
        # Should include all base keys
        base = compute_challenge_metrics(market_pnl, 3)
        for key in base:
            assert key in metrics

    def test_alpha_pnl_correct(self) -> None:
        market_pnl = pd.Series([100.0, -50.0, 80.0])  # total = 130
        original_pnl = pd.Series([120.0, -60.0, 90.0])  # total = 150
        metrics = compute_market_adjusted_metrics(market_pnl, original_pnl, trade_count=3)
        assert metrics["alpha_pnl"] == pytest.approx(130.0 - 150.0)
        assert metrics["original_total_pnl"] == pytest.approx(150.0)

    def test_alpha_pnl_positive_when_market_better(self) -> None:
        market_pnl = pd.Series([200.0])  # total = 200
        original_pnl = pd.Series([100.0])  # total = 100
        metrics = compute_market_adjusted_metrics(market_pnl, original_pnl, trade_count=1)
        assert metrics["alpha_pnl"] == pytest.approx(100.0)

    def test_total_pnl_is_market_based(self) -> None:
        market_pnl = pd.Series([50.0, 60.0])
        original_pnl = pd.Series([30.0, 40.0])
        metrics = compute_market_adjusted_metrics(market_pnl, original_pnl, trade_count=2)
        assert metrics["total_pnl"] == pytest.approx(110.0)  # from market_pnl

    def test_zero_trades(self) -> None:
        market_pnl = pd.Series([0.0, 0.0])
        original_pnl = pd.Series([0.0, 0.0])
        metrics = compute_market_adjusted_metrics(market_pnl, original_pnl, trade_count=0)
        assert metrics["total_pnl"] == pytest.approx(0.0)
        assert metrics["alpha_pnl"] == pytest.approx(0.0)


class TestMarketLeaderboardScore:
    def test_returns_tuple_of_three(self) -> None:
        metrics = {"total_pnl": 100.0, "sharpe_ratio": 1.5, "max_drawdown": 20.0}
        score = market_leaderboard_score(metrics)
        assert len(score) == 3

    def test_higher_pnl_ranks_first(self) -> None:
        a = market_leaderboard_score(
            {"total_pnl": 200.0, "sharpe_ratio": 1.0, "max_drawdown": 10.0}
        )
        b = market_leaderboard_score(
            {"total_pnl": 100.0, "sharpe_ratio": 1.0, "max_drawdown": 10.0}
        )
        assert a > b

    def test_drawdown_negated(self) -> None:
        score = market_leaderboard_score(
            {"total_pnl": 100.0, "sharpe_ratio": 1.0, "max_drawdown": 30.0}
        )
        assert score[2] == pytest.approx(-30.0)

    def test_same_structure_as_original(self) -> None:
        metrics = {"total_pnl": 100.0, "sharpe_ratio": 1.5, "max_drawdown": 20.0}
        assert leaderboard_score(metrics) == market_leaderboard_score(metrics)


# ---------------------------------------------------------------------------
# Tests for new base metrics (profit_factor, annualized_pnl_eur)
# ---------------------------------------------------------------------------


class TestComputeChallengeMetricsExtended:
    def test_profit_factor_present(self) -> None:
        metrics = compute_challenge_metrics(pd.Series([100.0, -50.0]), trade_count=2)
        assert "profit_factor" in metrics

    def test_annualized_pnl_present(self) -> None:
        metrics = compute_challenge_metrics(pd.Series([100.0, -50.0]), trade_count=2)
        assert "annualized_pnl_eur" in metrics

    def test_profit_factor_correct(self) -> None:
        pnl = pd.Series([100.0, -50.0, 200.0, -25.0])
        metrics = compute_challenge_metrics(pnl, trade_count=4)
        # gross_profit = 300, gross_loss = 75
        assert metrics["profit_factor"] == pytest.approx(300.0 / 75.0)

    def test_profit_factor_inf_when_no_losses(self) -> None:
        pnl = pd.Series([100.0, 50.0, 200.0])
        metrics = compute_challenge_metrics(pnl, trade_count=3)
        assert metrics["profit_factor"] == float("inf")

    def test_annualized_pnl_scales_to_252(self) -> None:
        # 252 days of +1 EUR/day = 252 EUR total, annualized = 252
        pnl = pd.Series([1.0] * 252)
        metrics = compute_challenge_metrics(pnl, trade_count=252)
        assert metrics["annualized_pnl_eur"] == pytest.approx(252.0)

    def test_annualized_pnl_scales_for_partial_year(self) -> None:
        # 126 days (half year) of +1 = 126 total, annualized = 252
        pnl = pd.Series([1.0] * 126)
        metrics = compute_challenge_metrics(pnl, trade_count=126)
        assert metrics["annualized_pnl_eur"] == pytest.approx(252.0)

    def test_all_expected_keys_present(self) -> None:
        metrics = compute_challenge_metrics(pd.Series([10.0, -5.0]), trade_count=2)
        expected_keys = {
            "total_pnl",
            "days_evaluated",
            "trade_count",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "avg_win",
            "avg_loss",
            "best_day",
            "worst_day",
            "profit_factor",
            "annualized_pnl_eur",
        }
        assert set(metrics.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests for monthly_pnl
# ---------------------------------------------------------------------------


class TestMonthlyPnl:
    def _make_daily_pnl(self, days: int = 90) -> pd.Series:
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(days)]
        return pd.Series(np.random.default_rng(42).standard_normal(days) * 100, index=dates)

    def test_returns_dataframe_with_correct_columns(self) -> None:
        pnl = self._make_daily_pnl()
        result = monthly_pnl(pnl)
        assert list(result.columns) == ["year", "month", "pnl"]

    def test_monthly_sums_match_total(self) -> None:
        pnl = self._make_daily_pnl()
        result = monthly_pnl(pnl)
        assert result["pnl"].sum() == pytest.approx(pnl.sum(), abs=1e-6)

    def test_correct_number_of_months(self) -> None:
        # 90 days from Jan 1 -> Jan(31) + Feb(29) + Mar(30) = 90, covers 3 months
        pnl = self._make_daily_pnl(90)
        result = monthly_pnl(pnl)
        assert len(result) == 3  # Jan, Feb, Mar

    def test_empty_series(self) -> None:
        pnl = pd.Series([], dtype=float)
        result = monthly_pnl(pnl)
        assert result.empty


# ---------------------------------------------------------------------------
# Tests for rolling_sharpe
# ---------------------------------------------------------------------------


class TestRollingSharpe:
    def _make_daily_pnl(self, days: int = 60) -> pd.Series:
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(days)]
        return pd.Series(np.random.default_rng(42).standard_normal(days) * 100, index=dates)

    def test_returns_series_same_length(self) -> None:
        pnl = self._make_daily_pnl()
        result = rolling_sharpe(pnl, window=30)
        assert len(result) == len(pnl)

    def test_first_window_minus_one_are_nan(self) -> None:
        pnl = self._make_daily_pnl()
        result = rolling_sharpe(pnl, window=30)
        assert all(pd.isna(result.iloc[:29]))

    def test_values_after_window_are_finite(self) -> None:
        pnl = self._make_daily_pnl()
        result = rolling_sharpe(pnl, window=30)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        assert all(np.isfinite(non_nan))

    def test_annualisation_factor(self) -> None:
        # Constant daily PnL -> Sharpe = mean/std * sqrt(252) -> inf (std=0)
        # Use almost-constant for finite result
        pnl = pd.Series(
            [1.0] * 30,
            index=[date(2024, 1, 1) + timedelta(days=i) for i in range(30)],
        )
        # std is 0 -> result is inf
        result = rolling_sharpe(pnl, window=30)
        assert result.iloc[-1] == float("inf") or pd.isna(result.iloc[-1])

    def test_negative_mean_gives_negative_sharpe(self) -> None:
        pnl = pd.Series(
            [-10.0 + np.random.default_rng(42).standard_normal() for _ in range(60)],
            index=[date(2024, 1, 1) + timedelta(days=i) for i in range(60)],
        )
        result = rolling_sharpe(pnl, window=30)
        non_nan = result.dropna()
        # At least some values should be negative
        assert any(non_nan < 0)
