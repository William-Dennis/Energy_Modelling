"""Tests for Phase 10e: Sentinel Case Studies.

Tests the core functions from ``scripts/phase10e_sentinel_traces.py``
using small synthetic datasets -- no real data or pickles needed.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from phase10e_sentinel_traces import (  # noqa: E402
    SentinelCase,
    build_case_summary,
    build_iteration_trace,
    find_active_collapse_window,
    find_cluster_switching_window,
    find_early_accuracy_dates,
    find_high_volatility_window,
    generate_narrative,
    select_sentinel_cases,
)

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_dates(n: int = 5) -> list[datetime.date]:
    return [datetime.date(2024, 1, d) for d in range(1, n + 1)]


def _make_iteration(
    iteration: int,
    market_prices: list[float],
    profits: dict[str, float],
    weights: dict[str, float] | None = None,
) -> FuturesMarketIteration:
    """Create a synthetic FuturesMarketIteration."""
    if weights is None:
        raw = {n: max(p, 0.0) for n, p in profits.items()}
        total = sum(raw.values())
        weights = {n: w / total for n, w in raw.items()} if total > 0 else {n: 0.0 for n in raw}
    active = [n for n, w in weights.items() if w > 0.0]
    dates = pd.date_range("2024-01-01", periods=len(market_prices), freq="D")
    return FuturesMarketIteration(
        iteration=iteration,
        market_prices=pd.Series(market_prices, index=dates, name="market_price"),
        strategy_profits=profits,
        strategy_weights=weights,
        active_strategies=active,
    )


def _make_equilibrium(
    iterations: list[FuturesMarketIteration],
    converged: bool = False,
    delta: float = 1.0,
) -> FuturesMarketEquilibrium:
    last = iterations[-1] if iterations else None
    return FuturesMarketEquilibrium(
        iterations=iterations,
        final_market_prices=last.market_prices if last else pd.Series(dtype=float),
        final_weights=last.strategy_weights if last else {},
        converged=converged,
        convergence_delta=delta,
    )


@pytest.fixture()
def real_prices() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.Series([50.0, 52.0, 48.0, 55.0, 47.0], index=dates, name="real_price")


@pytest.fixture()
def simple_equilibrium() -> FuturesMarketEquilibrium:
    """10-iteration equilibrium with declining active count."""
    iters = []
    for k in range(10):
        active_ratio = max(0, 1.0 - k * 0.12)
        profits = {
            "StratA": 100.0 * active_ratio,
            "StratB": 50.0 * max(0, active_ratio - 0.3),
            "StratC": -10.0,
        }
        iters.append(
            _make_iteration(
                k,
                [50.0 + k * 0.5, 51.0 + k * 0.3, 49.0, 52.0, 48.0],
                profits,
            )
        )
    return _make_equilibrium(iters, converged=False, delta=2.0)


@pytest.fixture()
def collapse_equilibrium() -> FuturesMarketEquilibrium:
    """Equilibrium that collapses to zero active strategies."""
    iters = []
    for k in range(15):
        if k < 10:
            profits = {"StratA": float(10 - k), "StratB": float(5 - k)}
        else:
            profits = {"StratA": -1.0, "StratB": -2.0}

        weights = {"StratA": 0.0, "StratB": 0.0} if k >= 10 else None

        iters.append(
            _make_iteration(
                k,
                [50.0, 51.0, 49.0, 52.0, 48.0],
                profits,
                weights,
            )
        )
    return _make_equilibrium(iters, converged=True, delta=0.0)


# ---------------------------------------------------------------------------
# Test: find_high_volatility_window
# ---------------------------------------------------------------------------


class TestFindHighVolatilityWindow:
    def test_returns_valid_range(self, real_prices: pd.Series):
        start, end = find_high_volatility_window(real_prices, window_size=3)
        assert 0 <= start <= end < len(real_prices)

    def test_returns_tuple_of_ints(self, real_prices: pd.Series):
        result = find_high_volatility_window(real_prices, window_size=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_volatile_window_detected(self):
        """A series with a spike should detect the spike window."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = pd.Series([50.0] * 10 + [50.0, 80.0, 30.0, 70.0, 40.0] + [50.0] * 15, index=dates)
        start, end = find_high_volatility_window(prices, window_size=5)
        # The volatile section is around indices 10-14
        assert start >= 5  # window should overlap the spike
        assert end <= 20


# ---------------------------------------------------------------------------
# Test: find_active_collapse_window
# ---------------------------------------------------------------------------


class TestFindActiveCollapseWindow:
    def test_detects_collapse(self, collapse_equilibrium: FuturesMarketEquilibrium):
        start, end = find_active_collapse_window(collapse_equilibrium, margin=3)
        assert start >= 0
        assert end <= len(collapse_equilibrium.iterations) - 1
        # The collapse is at iter 10, so window should include it
        assert start <= 10 <= end

    def test_no_collapse_returns_tail(self, simple_equilibrium: FuturesMarketEquilibrium):
        start, end = find_active_collapse_window(simple_equilibrium, margin=3)
        assert end == len(simple_equilibrium.iterations) - 1


# ---------------------------------------------------------------------------
# Test: find_early_accuracy_dates
# ---------------------------------------------------------------------------


class TestFindEarlyAccuracyDates:
    def test_finds_degraded_dates(self):
        """Construct a case where iter-0 is better than final on some dates."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        real = pd.Series([50.0, 52.0, 48.0, 55.0, 47.0], index=dates)

        # Iter 0: close to real
        iter0 = _make_iteration(
            0,
            [50.5, 52.5, 47.5, 55.5, 46.5],
            {"A": 10.0, "B": 5.0},
        )
        # Final: far from real on some dates
        final = _make_iteration(
            9,
            [45.0, 52.5, 47.5, 60.0, 46.5],
            {"A": 10.0, "B": 5.0},
        )
        eq = _make_equilibrium([iter0] + [final], converged=False, delta=1.0)

        result = find_early_accuracy_dates(eq, real, n_dates=5)
        # date 0 (50 vs 50.5 vs 45.0) and date 3 (55 vs 55.5 vs 60.0) should be found
        assert len(result) >= 1

    def test_empty_if_no_degradation(self, real_prices: pd.Series):
        # Both iters have the same prices
        iter0 = _make_iteration(0, [50.0, 52.0, 48.0, 55.0, 47.0], {"A": 10.0})
        iter1 = _make_iteration(1, [50.0, 52.0, 48.0, 55.0, 47.0], {"A": 10.0})
        eq = _make_equilibrium([iter0, iter1])
        result = find_early_accuracy_dates(eq, real_prices)
        assert result == []


# ---------------------------------------------------------------------------
# Test: find_cluster_switching_window
# ---------------------------------------------------------------------------


class TestFindClusterSwitchingWindow:
    def test_returns_valid_range(self, simple_equilibrium: FuturesMarketEquilibrium):
        start, end = find_cluster_switching_window(simple_equilibrium, window_size=5)
        assert 0 <= start <= end < len(simple_equilibrium.iterations)

    def test_with_clusters(self, simple_equilibrium: FuturesMarketEquilibrium):
        clusters = {"StratA": 1, "StratB": 1, "StratC": 2}
        start, end = find_cluster_switching_window(
            simple_equilibrium, strategy_clusters=clusters, window_size=5
        )
        assert 0 <= start <= end


# ---------------------------------------------------------------------------
# Test: build_iteration_trace
# ---------------------------------------------------------------------------


class TestBuildIterationTrace:
    def test_trace_shape(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        case = SentinelCase(
            case_id="test_case",
            case_type="test",
            year=2024,
            description="Test case",
            iter_start=0,
            iter_end=4,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices)
        assert len(trace) == 5  # iters 0 through 4
        assert "case_id" in trace.columns
        assert "mae" in trace.columns
        assert "top1_strategy" in trace.columns
        assert "dominant_cluster" in trace.columns

    def test_trace_with_clusters(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        clusters = {"StratA": 1, "StratB": 1, "StratC": 2}
        case = SentinelCase(
            case_id="cluster_test",
            case_type="test",
            year=2024,
            description="Test with clusters",
            iter_start=0,
            iter_end=2,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices, clusters)
        assert len(trace) == 3
        # dominant_cluster should be a valid cluster ID
        assert all(trace["dominant_cluster"].apply(lambda x: x in {"1", "2", "-1"}))

    def test_trace_with_focus_dates(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        focus = [real_prices.index[0], real_prices.index[1]]
        case = SentinelCase(
            case_id="focus_test",
            case_type="test",
            year=2024,
            description="Test with focus dates",
            iter_start=0,
            iter_end=2,
            focus_dates=focus,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices)
        assert len(trace) == 3

    def test_empty_window(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        case = SentinelCase(
            case_id="empty_test",
            case_type="test",
            year=2024,
            description="Window beyond iterations",
            iter_start=100,
            iter_end=110,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices)
        assert len(trace) == 0


# ---------------------------------------------------------------------------
# Test: generate_narrative
# ---------------------------------------------------------------------------


class TestGenerateNarrative:
    def test_narrative_contains_case_info(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        case = SentinelCase(
            case_id="narr_test",
            case_type="high_volatility_non_convergence",
            year=2024,
            description="Test narrative",
            iter_start=0,
            iter_end=4,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices)
        narrative = generate_narrative(case, trace)
        assert "narr_test" in narrative
        assert "2024" in narrative
        assert "MAE" in narrative

    def test_narrative_for_zero_active(
        self, collapse_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        case = SentinelCase(
            case_id="zero_test",
            case_type="zero_active_convergence",
            year=2025,
            description="Test zero-active narrative",
            iter_start=8,
            iter_end=12,
        )
        trace = build_iteration_trace(case, collapse_equilibrium, real_prices)
        narrative = generate_narrative(case, trace)
        assert "absorbing" in narrative.lower() or "truncation" in narrative.lower()

    def test_narrative_for_cluster_switching(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        case = SentinelCase(
            case_id="clsw_test",
            case_type="cluster_switching",
            year=2024,
            description="Test cluster switching",
            iter_start=0,
            iter_end=4,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices)
        narrative = generate_narrative(case, trace)
        assert "leadership" in narrative.lower()

    def test_narrative_for_early_accuracy(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        case = SentinelCase(
            case_id="ea_test",
            case_type="early_accuracy_lost",
            year=2024,
            description="Test early accuracy",
            iter_start=0,
            iter_end=4,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices)
        narrative = generate_narrative(case, trace)
        assert "accuracy" in narrative.lower() or "MAE" in narrative

    def test_narrative_empty_trace(self):
        case = SentinelCase(
            case_id="empty",
            case_type="test",
            year=2024,
            description="Empty case",
            iter_start=100,
            iter_end=110,
        )
        trace = pd.DataFrame()
        narrative = generate_narrative(case, trace)
        assert "No trace data" in narrative


# ---------------------------------------------------------------------------
# Test: build_case_summary
# ---------------------------------------------------------------------------


class TestBuildCaseSummary:
    def test_summary_fields(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        case = SentinelCase(
            case_id="sum_test",
            case_type="test",
            year=2024,
            description="Summary test",
            iter_start=0,
            iter_end=4,
        )
        trace = build_iteration_trace(case, simple_equilibrium, real_prices)
        summary = build_case_summary(case, trace)
        assert summary["case_id"] == "sum_test"
        assert summary["n_trace_rows"] == 5
        assert "start_mae" in summary
        assert "end_mae" in summary
        assert "leadership_changes" in summary

    def test_summary_empty_trace(self):
        case = SentinelCase(
            case_id="empty_sum",
            case_type="test",
            year=2024,
            description="Empty",
            iter_start=100,
            iter_end=110,
        )
        trace = pd.DataFrame()
        summary = build_case_summary(case, trace)
        assert summary["n_trace_rows"] == 0


# ---------------------------------------------------------------------------
# Test: select_sentinel_cases
# ---------------------------------------------------------------------------


class TestSelectSentinelCases:
    def test_2024_cases(self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series):
        cases = select_sentinel_cases(2024, real_prices, simple_equilibrium, strategy_clusters=None)
        # Should have high-vol + cluster-switching + early-accuracy
        case_types = [c.case_type for c in cases]
        assert "high_volatility_non_convergence" in case_types
        assert "cluster_switching" in case_types

    def test_2025_cases(
        self, collapse_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        cases = select_sentinel_cases(
            2025, real_prices, collapse_equilibrium, strategy_clusters=None
        )
        case_types = [c.case_type for c in cases]
        assert "zero_active_convergence" in case_types

    def test_all_cases_have_valid_fields(
        self, simple_equilibrium: FuturesMarketEquilibrium, real_prices: pd.Series
    ):
        cases = select_sentinel_cases(2024, real_prices, simple_equilibrium)
        for case in cases:
            assert case.case_id
            assert case.case_type
            assert case.year == 2024
            assert case.iter_start >= 0
            assert case.iter_end >= case.iter_start
