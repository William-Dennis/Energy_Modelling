"""Tests for strategy.analysis."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from energy_modelling.market_simulation.types import Settlement, Trade
from energy_modelling.strategy.analysis import compute_metrics, monthly_pnl, rolling_sharpe
from energy_modelling.strategy.runner import BacktestResult


def _make_backtest_result(
    daily_pnls: list[float],
    start_date: date = date(2024, 1, 2),
) -> BacktestResult:
    """Create a BacktestResult from a list of daily PnL values."""
    dates = [start_date + timedelta(days=i) for i in range(len(daily_pnls))]
    settlements = []
    for d, pnl in zip(dates, daily_pnls, strict=False):
        trade = Trade(delivery_date=d, entry_price=50.0, position_mw=1.0)
        settlement_price = 50.0 + pnl / 24.0  # back-derive a settlement price
        settlements.append(Settlement(trade=trade, settlement_price=settlement_price, pnl=pnl))
    daily_series = pd.Series(daily_pnls, index=dates, name="pnl")
    cumulative = daily_series.cumsum()
    return BacktestResult(
        settlements=settlements,
        daily_pnl=daily_series,
        cumulative_pnl=cumulative,
    )


class TestComputeMetrics:
    """Tests for compute_metrics()."""

    def test_returns_dict(self) -> None:
        result = _make_backtest_result([100.0, -50.0, 200.0, -30.0])
        metrics = compute_metrics(result)
        assert isinstance(metrics, dict)

    def test_total_pnl(self) -> None:
        result = _make_backtest_result([100.0, -50.0, 200.0, -30.0])
        metrics = compute_metrics(result)
        assert metrics["total_pnl"] == pytest.approx(220.0)

    def test_num_trading_days(self) -> None:
        result = _make_backtest_result([10.0, 20.0, 30.0])
        metrics = compute_metrics(result)
        assert metrics["num_trading_days"] == 3

    def test_win_rate(self) -> None:
        result = _make_backtest_result([100.0, -50.0, 200.0, -30.0])
        metrics = compute_metrics(result)
        assert metrics["win_rate"] == pytest.approx(0.5)

    def test_all_wins(self) -> None:
        result = _make_backtest_result([10.0, 20.0, 30.0])
        metrics = compute_metrics(result)
        assert metrics["win_rate"] == pytest.approx(1.0)

    def test_all_losses(self) -> None:
        result = _make_backtest_result([-10.0, -20.0, -30.0])
        metrics = compute_metrics(result)
        assert metrics["win_rate"] == pytest.approx(0.0)

    def test_sharpe_ratio_positive(self) -> None:
        """Positive mean with low vol should give positive Sharpe."""
        result = _make_backtest_result([10.0] * 100)
        metrics = compute_metrics(result)
        # Constant positive PnL: Sharpe should be very high (inf-ish)
        assert metrics["sharpe_ratio"] > 0

    def test_max_drawdown(self) -> None:
        result = _make_backtest_result([100.0, -200.0, 50.0])
        metrics = compute_metrics(result)
        # Peak = 100, trough = 100 - 200 = -100, drawdown = 200
        assert metrics["max_drawdown"] == pytest.approx(200.0)

    def test_profit_factor(self) -> None:
        result = _make_backtest_result([100.0, -50.0])
        metrics = compute_metrics(result)
        assert metrics["profit_factor"] == pytest.approx(2.0)

    def test_best_and_worst_day(self) -> None:
        result = _make_backtest_result([100.0, -200.0, 50.0])
        metrics = compute_metrics(result)
        assert metrics["best_day"] == pytest.approx(100.0)
        assert metrics["worst_day"] == pytest.approx(-200.0)

    def test_avg_win_and_loss(self) -> None:
        result = _make_backtest_result([100.0, 200.0, -50.0, -150.0])
        metrics = compute_metrics(result)
        assert metrics["avg_win"] == pytest.approx(150.0)
        assert metrics["avg_loss"] == pytest.approx(-100.0)

    def test_has_all_keys(self) -> None:
        result = _make_backtest_result([100.0, -50.0, 200.0, -30.0])
        metrics = compute_metrics(result)
        expected_keys = {
            "total_pnl",
            "num_trading_days",
            "annualized_return_pct",
            "sharpe_ratio",
            "max_drawdown",
            "max_drawdown_pct",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "best_day",
            "worst_day",
        }
        assert expected_keys.issubset(set(metrics.keys()))


class TestMonthlyPnl:
    """Tests for monthly_pnl()."""

    def test_returns_dataframe(self) -> None:
        result = _make_backtest_result([10.0] * 60, start_date=date(2024, 1, 1))
        df = monthly_pnl(result)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        result = _make_backtest_result([10.0] * 60, start_date=date(2024, 1, 1))
        df = monthly_pnl(result)
        assert "year" in df.columns
        assert "month" in df.columns
        assert "pnl" in df.columns

    def test_monthly_sums(self) -> None:
        """Monthly PnL should sum correctly."""
        # 31 days in Jan + 29 days in Feb (but we start Jan 1 with 60 days)
        result = _make_backtest_result([10.0] * 60, start_date=date(2024, 1, 1))
        df = monthly_pnl(result)
        assert df["pnl"].sum() == pytest.approx(600.0)


class TestRollingSharpe:
    """Tests for rolling_sharpe()."""

    def test_returns_series(self) -> None:
        result = _make_backtest_result([10.0] * 60)
        rs = rolling_sharpe(result, window=10)
        assert isinstance(rs, pd.Series)

    def test_first_values_are_nan(self) -> None:
        """First (window - 1) values should be NaN."""
        result = _make_backtest_result([10.0] * 60)
        rs = rolling_sharpe(result, window=10)
        assert pd.isna(rs.iloc[0])
