"""Tests for Phase 10c: Mechanism Attribution.

Tests the core functions from ``scripts/phase10c_mechanism_attribution.py``
using small synthetic datasets — no real data or pickles needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts/ to path so we can import from the phase10c script
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from phase10c_mechanism_attribution import (  # noqa: E402
    ABLATION_GROUPS,
    _apply_ablation,
    _apply_init_mode,
    build_ablation_experiments,
    build_ema_sweep_experiments,
    build_init_experiments,
    classify_outcome,
    run_market,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_forecasts() -> dict[str, dict]:
    """Three strategies with known forecasts over 5 dates."""
    import datetime

    dates = [datetime.date(2024, 1, d) for d in range(1, 6)]
    return {
        "StratA": {d: 50.0 + i for i, d in enumerate(dates)},
        "StratB": {d: 45.0 - i for i, d in enumerate(dates)},
        "StratC": {d: 48.0 for d in dates},
    }


@pytest.fixture()
def simple_real_prices() -> pd.Series:
    import datetime

    dates = [datetime.date(2024, 1, d) for d in range(1, 6)]
    return pd.Series([50.0, 51.0, 52.0, 53.0, 54.0], index=dates, name="real_price")


@pytest.fixture()
def simple_initial_prices() -> pd.Series:
    import datetime

    dates = [datetime.date(2024, 1, d) for d in range(1, 6)]
    return pd.Series([49.0, 50.0, 51.0, 52.0, 53.0], index=dates, name="initial_price")


# ---------------------------------------------------------------------------
# Test: ABLATION_GROUPS
# ---------------------------------------------------------------------------


class TestAblationGroups:
    def test_all_groups_have_strategies(self) -> None:
        for name, strategies in ABLATION_GROUPS.items():
            assert len(strategies) > 0, f"Group {name} is empty"

    def test_compound_all_ml_is_union(self) -> None:
        expected = set(ABLATION_GROUPS["ml_regression"] + ABLATION_GROUPS["ml_classification"])
        assert set(ABLATION_GROUPS["all_ml"]) == expected

    def test_compound_all_rule_based_mostly_excludes_ml(self) -> None:
        rb = set(ABLATION_GROUPS["all_rule_based"])
        ml = set(ABLATION_GROUPS["all_ml"])
        ens = set(ABLATION_GROUPS["ensemble_meta"])
        # Lasso Calendar Augmented spans both calendar_temporal and ml_regression
        overlap = rb & ml
        assert overlap <= {"Lasso Calendar Augmented"}, (
            f"Unexpected overlap: {overlap - {'Lasso Calendar Augmented'}}"
        )
        assert rb & ens == set(), "Rule-based should not overlap with ensemble"


# ---------------------------------------------------------------------------
# Test: _apply_ablation
# ---------------------------------------------------------------------------


class TestApplyAblation:
    def test_none_returns_all(self, simple_forecasts: dict) -> None:
        result = _apply_ablation(simple_forecasts, "none")
        assert result == simple_forecasts

    def test_remove_known_group(self) -> None:
        forecasts = {
            "Always Long": {1: 50.0},
            "Always Short": {1: 40.0},
            "StratC": {1: 45.0},
        }
        result = _apply_ablation(forecasts, "remove_naive_baselines")
        assert list(result.keys()) == ["StratC"]

    def test_keep_only_known_group(self) -> None:
        forecasts = {
            "Always Long": {1: 50.0},
            "Always Short": {1: 40.0},
            "Lasso Regression": {1: 52.0},
            "Ridge Regression": {1: 48.0},
        }
        result = _apply_ablation(forecasts, "keep_only_ml")
        assert set(result.keys()) == {"Lasso Regression", "Ridge Regression"}

    def test_remove_unknown_group_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ablation group"):
            _apply_ablation({}, "remove_nonexistent")

    def test_keep_only_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown keep_only group"):
            _apply_ablation({}, "keep_only_nonexistent")

    def test_unknown_ablation_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ablation"):
            _apply_ablation({}, "invalid_prefix")


# ---------------------------------------------------------------------------
# Test: _apply_init_mode
# ---------------------------------------------------------------------------


class TestApplyInitMode:
    def test_default_returns_initial(
        self,
        simple_initial_prices: pd.Series,
        simple_real_prices: pd.Series,
        simple_forecasts: dict,
    ) -> None:
        result = _apply_init_mode(
            "default",
            simple_initial_prices,
            simple_real_prices,
            simple_forecasts,
        )
        pd.testing.assert_series_equal(result, simple_initial_prices)

    def test_constant_50(
        self,
        simple_initial_prices: pd.Series,
        simple_real_prices: pd.Series,
        simple_forecasts: dict,
    ) -> None:
        result = _apply_init_mode(
            "constant_50",
            simple_initial_prices,
            simple_real_prices,
            simple_forecasts,
        )
        assert (result == 50.0).all()
        assert len(result) == len(simple_initial_prices)

    def test_real_prices_returns_real(
        self,
        simple_initial_prices: pd.Series,
        simple_real_prices: pd.Series,
        simple_forecasts: dict,
    ) -> None:
        result = _apply_init_mode(
            "real_prices",
            simple_initial_prices,
            simple_real_prices,
            simple_forecasts,
        )
        pd.testing.assert_series_equal(
            result, simple_real_prices.reindex(simple_initial_prices.index)
        )

    def test_forecast_mean(
        self,
        simple_initial_prices: pd.Series,
        simple_real_prices: pd.Series,
        simple_forecasts: dict,
    ) -> None:
        result = _apply_init_mode(
            "forecast_mean",
            simple_initial_prices,
            simple_real_prices,
            simple_forecasts,
        )
        # StratA=50+i, StratB=45-i, StratC=48 → mean = (50+i + 45-i + 48)/3
        # = 143/3 ≈ 47.67 for all dates
        assert abs(result.iloc[0] - 47.6667) < 0.01

    def test_unknown_init_mode_raises(
        self,
        simple_initial_prices: pd.Series,
        simple_real_prices: pd.Series,
        simple_forecasts: dict,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown init_mode"):
            _apply_init_mode(
                "invalid",
                simple_initial_prices,
                simple_real_prices,
                simple_forecasts,
            )


# ---------------------------------------------------------------------------
# Test: classify_outcome
# ---------------------------------------------------------------------------


class TestClassifyOutcome:
    def test_no_strategies(self) -> None:
        assert (
            classify_outcome({"n_strategies": 0, "converged": False, "final_active_count": 0})
            == "no_strategies"
        )

    def test_non_converged(self) -> None:
        assert (
            classify_outcome({"n_strategies": 10, "converged": False, "final_active_count": 5})
            == "non_converged"
        )

    def test_absorbing_collapse(self) -> None:
        assert (
            classify_outcome({"n_strategies": 10, "converged": True, "final_active_count": 0})
            == "absorbing_collapse"
        )

    def test_near_collapse(self) -> None:
        assert (
            classify_outcome({"n_strategies": 10, "converged": True, "final_active_count": 2})
            == "near_collapse"
        )

    def test_healthy_convergence(self) -> None:
        assert (
            classify_outcome({"n_strategies": 10, "converged": True, "final_active_count": 5})
            == "healthy_convergence"
        )


# ---------------------------------------------------------------------------
# Test: run_market (integration on tiny data)
# ---------------------------------------------------------------------------


class TestRunMarket:
    def test_returns_expected_keys(
        self,
        simple_forecasts: dict,
        simple_real_prices: pd.Series,
        simple_initial_prices: pd.Series,
    ) -> None:
        result = run_market(
            simple_forecasts,
            simple_real_prices,
            simple_initial_prices,
            max_iterations=10,
            ema_alpha=0.1,
        )
        expected_keys = {
            "converged",
            "n_iterations",
            "final_delta",
            "mae",
            "rmse",
            "bias",
            "n_strategies",
            "final_active_count",
            "final_weight_entropy",
            "final_top1_weight",
        }
        assert set(result.keys()) == expected_keys

    def test_n_strategies_matches_input(
        self,
        simple_forecasts: dict,
        simple_real_prices: pd.Series,
        simple_initial_prices: pd.Series,
    ) -> None:
        result = run_market(
            simple_forecasts,
            simple_real_prices,
            simple_initial_prices,
            max_iterations=10,
            ema_alpha=0.1,
        )
        assert result["n_strategies"] == 3

    def test_oracle_init_converges_immediately(
        self,
        simple_forecasts: dict,
        simple_real_prices: pd.Series,
    ) -> None:
        """When initial prices = real prices, market should converge in 1 iter."""
        result = run_market(
            simple_forecasts,
            simple_real_prices,
            simple_real_prices,
            max_iterations=100,
            ema_alpha=0.1,
        )
        assert result["converged"] is True
        assert result["n_iterations"] <= 2

    def test_very_small_alpha_converges(
        self,
        simple_forecasts: dict,
        simple_real_prices: pd.Series,
        simple_initial_prices: pd.Series,
    ) -> None:
        """Very small alpha should eventually converge for simple data."""
        result = run_market(
            simple_forecasts,
            simple_real_prices,
            simple_initial_prices,
            max_iterations=1000,
            ema_alpha=0.01,
        )
        # For simple 3-strategy data this should converge
        assert result["converged"] is True


# ---------------------------------------------------------------------------
# Test: experiment builders
# ---------------------------------------------------------------------------


class TestExperimentBuilders:
    def test_ema_sweep_has_seven_configs(self) -> None:
        exps = build_ema_sweep_experiments()
        assert len(exps) == 7
        alphas = [e["ema_alpha"] for e in exps]
        assert 0.01 in alphas
        assert 1.0 in alphas

    def test_init_experiments_has_four_modes(self) -> None:
        exps = build_init_experiments()
        assert len(exps) == 4
        modes = {e["init_mode"] for e in exps}
        assert modes == {"default", "forecast_mean", "constant_50", "real_prices"}

    def test_ablation_experiments_have_baseline(self) -> None:
        exps = build_ablation_experiments()
        ids = [e["experiment_id"] for e in exps]
        assert "ablation_baseline" in ids

    def test_ablation_experiments_have_remove_and_keep(self) -> None:
        exps = build_ablation_experiments()
        ids = [e["experiment_id"] for e in exps]
        assert any(i.startswith("remove_") for i in ids)
        assert any(i.startswith("keep_only_") for i in ids)

    def test_all_experiments_have_required_fields(self) -> None:
        all_exps = (
            build_ema_sweep_experiments() + build_init_experiments() + build_ablation_experiments()
        )
        required = {
            "experiment_type",
            "experiment_id",
            "label",
            "ema_alpha",
            "init_mode",
            "ablation",
        }
        for exp in all_exps:
            assert required.issubset(set(exp.keys())), (
                f"Missing keys in {exp['experiment_id']}: {required - set(exp.keys())}"
            )
