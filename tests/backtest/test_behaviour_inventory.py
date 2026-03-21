"""Tests for Phase 10b: Behaviour Inventory metric extraction and classification.

Tests the core functions from ``scripts/phase10b_behaviour_inventory.py``
using small synthetic market iterations.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts/ to path so we can import from the phase10b script
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from phase10b_behaviour_inventory import (  # noqa: E402
    classify_behaviour,
    compute_iteration_metrics,
    extract_iteration_panel,
)

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _make_real_prices(values: list[float]) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates, name="settlement_price")


# ---------------------------------------------------------------------------
# compute_iteration_metrics tests
# ---------------------------------------------------------------------------


class TestComputeIterationMetrics:
    """Tests for compute_iteration_metrics()."""

    def test_first_iteration_has_inf_delta(self):
        it = _make_iteration(0, [50.0, 60.0], {"A": 10.0, "B": -5.0})
        real = _make_real_prices([55.0, 65.0])
        metrics = compute_iteration_metrics(it, None, real)
        assert metrics["convergence_delta"] == float("inf")
        assert metrics["iteration"] == 0

    def test_delta_computed_from_prev(self):
        prev = _make_iteration(0, [50.0, 60.0], {"A": 10.0})
        curr = _make_iteration(1, [52.0, 58.0], {"A": 8.0})
        real = _make_real_prices([55.0, 65.0])
        metrics = compute_iteration_metrics(curr, prev, real)
        # max(|52-50|, |58-60|) = max(2, 2) = 2.0
        assert metrics["convergence_delta"] == pytest.approx(2.0)

    def test_mae_rmse_bias(self):
        it = _make_iteration(0, [50.0, 60.0], {"A": 10.0})
        real = _make_real_prices([55.0, 65.0])
        metrics = compute_iteration_metrics(it, None, real)
        # residuals = [50-55, 60-65] = [-5, -5]
        assert metrics["mae"] == pytest.approx(5.0)
        assert metrics["rmse"] == pytest.approx(5.0)
        assert metrics["bias"] == pytest.approx(-5.0)

    def test_active_count(self):
        it = _make_iteration(
            0,
            [50.0],
            {"A": 10.0, "B": -5.0, "C": 3.0},
        )
        real = _make_real_prices([55.0])
        metrics = compute_iteration_metrics(it, None, real)
        # A (w>0), C (w>0), B (w=0)
        assert metrics["active_count"] == 2

    def test_weight_entropy(self):
        # Two equal-weight strategies: entropy = -2*(0.5 * ln(0.5)) = ln(2)
        weights = {"A": 0.5, "B": 0.5}
        it = _make_iteration(0, [50.0], {"A": 10.0, "B": 10.0}, weights=weights)
        real = _make_real_prices([55.0])
        metrics = compute_iteration_metrics(it, None, real)
        assert metrics["weight_entropy"] == pytest.approx(math.log(2), abs=1e-9)

    def test_entropy_zero_when_no_active(self):
        weights = {"A": 0.0, "B": 0.0}
        it = _make_iteration(0, [50.0], {"A": -5.0, "B": -3.0}, weights=weights)
        real = _make_real_prices([55.0])
        metrics = compute_iteration_metrics(it, None, real)
        assert metrics["weight_entropy"] == 0.0
        assert metrics["active_count"] == 0

    def test_top1_and_top5_concentration(self):
        weights = {"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.05, "E": 0.03, "F": 0.02}
        it = _make_iteration(
            0,
            [50.0],
            {n: 10.0 for n in weights},
            weights=weights,
        )
        real = _make_real_prices([55.0])
        metrics = compute_iteration_metrics(it, None, real)
        assert metrics["top1_weight"] == pytest.approx(0.6)
        # top 5 = 0.6 + 0.2 + 0.1 + 0.05 + 0.03 = 0.98
        assert metrics["top5_concentration"] == pytest.approx(0.98)

    def test_profit_metrics(self):
        it = _make_iteration(0, [50.0], {"A": 100.0, "B": -50.0, "C": 30.0})
        real = _make_real_prices([55.0])
        metrics = compute_iteration_metrics(it, None, real)
        assert metrics["total_profit_spread"] == pytest.approx(150.0)
        assert metrics["median_profit"] == pytest.approx(30.0)
        assert metrics["max_profit"] == pytest.approx(100.0)
        assert metrics["min_profit"] == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# extract_iteration_panel tests
# ---------------------------------------------------------------------------


class TestExtractIterationPanel:
    """Tests for extract_iteration_panel()."""

    def test_panel_shape(self):
        iters = [
            _make_iteration(0, [50.0, 60.0], {"A": 10.0, "B": 5.0}),
            _make_iteration(1, [52.0, 58.0], {"A": 8.0, "B": 4.0}),
            _make_iteration(2, [53.0, 57.0], {"A": 6.0, "B": 3.0}),
        ]
        eq = FuturesMarketEquilibrium(
            iterations=iters,
            final_market_prices=iters[-1].market_prices,
            final_weights=iters[-1].strategy_weights,
            converged=True,
            convergence_delta=0.5,
        )
        real = _make_real_prices([55.0, 65.0])
        panel = extract_iteration_panel(eq, real)
        assert len(panel) == 3
        assert "iteration" in panel.columns
        assert "mae" in panel.columns
        assert "convergence_delta" in panel.columns

    def test_first_delta_is_inf(self):
        iters = [_make_iteration(0, [50.0], {"A": 10.0})]
        eq = FuturesMarketEquilibrium(
            iterations=iters,
            final_market_prices=iters[-1].market_prices,
            final_weights=iters[-1].strategy_weights,
            converged=False,
            convergence_delta=1.0,
        )
        real = _make_real_prices([55.0])
        panel = extract_iteration_panel(eq, real)
        assert panel["convergence_delta"].iloc[0] == float("inf")


# ---------------------------------------------------------------------------
# classify_behaviour tests
# ---------------------------------------------------------------------------


class TestClassifyBehaviour:
    """Tests for classify_behaviour()."""

    def _make_panel(
        self,
        n: int,
        *,
        active_pattern: str = "constant",
        delta_pattern: str = "constant",
    ) -> pd.DataFrame:
        """Create a synthetic iteration panel for classification testing."""
        rows = []
        for i in range(n):
            if active_pattern == "constant":
                active = 30
            elif active_pattern == "collapse":
                active = max(0, 30 - i)
            elif active_pattern == "collapse_to_zero":
                active = max(0, 30 - i * 2) if i < 16 else 0
            else:
                active = 30

            if delta_pattern == "constant":
                delta = 1.0 if i > 0 else float("inf")
            elif delta_pattern == "decreasing":
                delta = max(0.001, 10.0 / (i + 1)) if i > 0 else float("inf")
            elif delta_pattern == "oscillating":
                delta = (1.0 + 0.5 * ((-1) ** i)) if i > 0 else float("inf")
            else:
                delta = 1.0 if i > 0 else float("inf")

            rows.append(
                {
                    "iteration": i,
                    "convergence_delta": delta,
                    "mae": 20.0 - i * 0.1,
                    "rmse": 25.0 - i * 0.1,
                    "bias": -1.0,
                    "active_count": active,
                    "weight_entropy": max(0.0, math.log(max(active, 1))),
                    "top1_weight": 1.0 / max(active, 1),
                    "top5_concentration": min(1.0, 5.0 / max(active, 1)),
                    "total_profit_spread": 100.0,
                    "median_profit": 10.0,
                    "max_profit": 50.0,
                    "min_profit": -50.0,
                }
            )
        return pd.DataFrame(rows)

    def test_absorbing_collapse(self):
        panel = self._make_panel(50, active_pattern="collapse_to_zero")
        cls = classify_behaviour(panel, 2025, converged=True, final_delta=0.0)
        assert cls.behaviour_label == "absorbing_collapse"
        assert cls.absorbing_state is True

    def test_healthy_convergence(self):
        panel = self._make_panel(50, active_pattern="constant")
        cls = classify_behaviour(panel, 2024, converged=True, final_delta=0.005)
        assert cls.behaviour_label == "healthy_convergence"
        assert cls.converged is True
        assert cls.active_collapse is False

    def test_non_convergence_oscillating(self):
        panel = self._make_panel(100, delta_pattern="oscillating")
        cls = classify_behaviour(panel, 2024, converged=False, final_delta=1.5)
        assert cls.behaviour_label == "oscillating_non_convergence"
        assert cls.oscillating

    def test_slow_damped_non_convergence(self):
        panel = self._make_panel(100, delta_pattern="decreasing")
        cls = classify_behaviour(panel, 2024, converged=False, final_delta=0.05)
        assert cls.behaviour_label == "slow_damped_non_convergence"
        assert cls.monotone_damped

    def test_classification_returns_correct_year(self):
        panel = self._make_panel(20)
        cls = classify_behaviour(panel, 2024, converged=True, final_delta=0.001)
        assert cls.year == 2024

    def test_mae_degradation_detected(self):
        panel = self._make_panel(50)
        # Override MAE to have a clear best then degradation
        panel.loc[:24, "mae"] = [20.0 - i * 0.5 for i in range(25)]
        panel.loc[25:, "mae"] = [8.5 + i * 0.2 for i in range(25)]
        cls = classify_behaviour(panel, 2024, converged=True, final_delta=0.001)
        assert cls.best_mae < cls.final_mae
        assert cls.mae_degraded_after_best

    def test_initial_and_final_metrics(self):
        panel = self._make_panel(30, active_pattern="collapse")
        cls = classify_behaviour(panel, 2024, converged=False, final_delta=1.0)
        assert cls.initial_active == 30
        assert cls.final_active == max(0, 30 - 29)
        assert cls.n_iterations == 30
