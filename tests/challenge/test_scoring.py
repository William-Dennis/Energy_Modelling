"""Tests for scoring extensions -- market-adjusted metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_modelling.challenge.scoring import (
    compute_challenge_metrics,
    compute_market_adjusted_metrics,
    leaderboard_score,
    market_leaderboard_score,
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
