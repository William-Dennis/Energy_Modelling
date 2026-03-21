"""Phase 10b: Behaviour Inventory — extract per-iteration market metrics.

Loads the saved market artifacts (``market_2024.pkl``, ``market_2025.pkl``)
and computes a comprehensive per-iteration metric panel for each year.
Classifies each run into behaviour modes and saves the combined results.

Metrics per iteration
---------------------
- convergence_delta: max|P_k - P_{k-1}|
- mae: mean|P_market - P_real|
- rmse: sqrt(mean((P_market - P_real)^2))
- bias: mean(P_market - P_real)
- active_count: number of strategies with positive weight
- weight_entropy: -sum(w * log(w)) for w > 0
- top1_weight: max strategy weight
- top5_concentration: sum of top 5 weights
- total_profit_spread: max(profit) - min(profit)
- median_profit: median of all strategy profits

Outputs
-------
- ``data/results/phase10/behaviour_inventory.csv`` — per-iteration metrics
- ``data/results/phase10/behaviour_summary.csv`` — per-year run-level summary
- Prints a behaviour classification for each year to stdout

Usage::

    uv run python scripts/phase10b_behaviour_inventory.py
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
)
from energy_modelling.backtest.futures_market_runner import FuturesMarketResult  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE10_DIR = RESULTS_DIR / "phase10"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def compute_iteration_metrics(
    iteration: FuturesMarketIteration,
    prev_iteration: FuturesMarketIteration | None,
    real_prices: pd.Series,
) -> dict:
    """Compute all metrics for a single iteration.

    Parameters
    ----------
    iteration:
        The current iteration snapshot.
    prev_iteration:
        The previous iteration snapshot (None for the first iteration).
    real_prices:
        Ground-truth settlement prices.

    Returns
    -------
    dict with metric name -> value.
    """
    market = iteration.market_prices
    aligned_real = real_prices.reindex(market.index)
    residuals = market - aligned_real

    # Convergence delta
    if prev_iteration is not None:
        prev_market = prev_iteration.market_prices
        delta = float((market - prev_market).abs().max())
    else:
        delta = float("inf")

    # Accuracy metrics
    mae = float(residuals.abs().mean())
    rmse = float(np.sqrt((residuals**2).mean()))
    bias = float(residuals.mean())

    # Active strategy metrics
    weights = iteration.strategy_weights
    active_count = len(iteration.active_strategies)

    # Weight entropy: -sum(w * ln(w)) for w > 0
    active_weights = [w for w in weights.values() if w > 0.0]
    entropy = -sum(w * math.log(w) for w in active_weights) if active_weights else 0.0

    # Concentration metrics
    sorted_weights = sorted(weights.values(), reverse=True)
    top1_weight = sorted_weights[0] if sorted_weights else 0.0
    top5_concentration = sum(sorted_weights[:5]) if sorted_weights else 0.0

    # Profit metrics
    profits = list(iteration.strategy_profits.values())
    total_profit_spread = max(profits) - min(profits) if profits else 0.0
    median_profit = float(np.median(profits)) if profits else 0.0
    max_profit = max(profits) if profits else 0.0
    min_profit = min(profits) if profits else 0.0

    return {
        "iteration": iteration.iteration,
        "convergence_delta": delta,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "active_count": active_count,
        "weight_entropy": entropy,
        "top1_weight": top1_weight,
        "top5_concentration": top5_concentration,
        "total_profit_spread": total_profit_spread,
        "median_profit": median_profit,
        "max_profit": max_profit,
        "min_profit": min_profit,
    }


def extract_iteration_panel(
    equilibrium: FuturesMarketEquilibrium,
    real_prices: pd.Series,
) -> pd.DataFrame:
    """Extract a DataFrame of per-iteration metrics from an equilibrium result.

    Parameters
    ----------
    equilibrium:
        The market equilibrium result.
    real_prices:
        Ground-truth settlement prices.

    Returns
    -------
    DataFrame with one row per iteration and columns for each metric.
    """
    rows: list[dict] = []
    iterations = equilibrium.iterations

    for i, it in enumerate(iterations):
        prev = iterations[i - 1] if i > 0 else None
        metrics = compute_iteration_metrics(it, prev, real_prices)
        rows.append(metrics)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Behaviour classification
# ---------------------------------------------------------------------------


@dataclass
class BehaviourClassification:
    """Run-level behaviour classification for a market simulation."""

    year: int
    converged: bool
    n_iterations: int
    final_delta: float

    # Accuracy trajectory
    initial_mae: float
    best_mae: float
    best_mae_iteration: int
    final_mae: float
    mae_degraded_after_best: bool  # True if final MAE > best MAE

    # Active strategy dynamics
    initial_active: int
    final_active: int
    min_active: int
    active_collapse: bool  # True if final_active <= 2
    active_collapse_iteration: int | None  # First iter where active <= 2

    # Convergence pattern
    monotone_damped: bool  # True if delta generally decreasing
    oscillating: bool  # True if delta has significant oscillation
    absorbing_state: bool  # True if converges via zero active strategies

    # Weight concentration
    initial_entropy: float
    final_entropy: float
    max_top1_weight: float

    # Summary label
    behaviour_label: str


def classify_behaviour(
    panel: pd.DataFrame,
    year: int,
    converged: bool,
    final_delta: float,
) -> BehaviourClassification:
    """Classify the run-level behaviour from the iteration metric panel.

    Parameters
    ----------
    panel:
        Per-iteration metrics DataFrame.
    year:
        The data year.
    converged:
        Whether the engine declared convergence.
    final_delta:
        The final convergence delta.

    Returns
    -------
    BehaviourClassification with the detected behaviour label.
    """
    n = len(panel)

    # Accuracy trajectory
    initial_mae = panel["mae"].iloc[0]
    best_mae = panel["mae"].min()
    best_mae_iter = int(panel["mae"].idxmin())
    final_mae = panel["mae"].iloc[-1]
    mae_degraded = final_mae > best_mae * 1.05  # 5% tolerance

    # Active strategy dynamics
    initial_active = int(panel["active_count"].iloc[0])
    final_active = int(panel["active_count"].iloc[-1])
    min_active = int(panel["active_count"].min())
    active_collapse = final_active <= 2
    collapse_mask = panel["active_count"] <= 2
    active_collapse_iter = int(collapse_mask.idxmax()) if collapse_mask.any() else None

    # Convergence pattern — look at delta series (skip first inf)
    deltas = panel["convergence_delta"].iloc[1:].values if n > 1 else np.array([])

    # Check monotone damped: is the rolling average of delta generally decreasing?
    monotone_damped = False
    oscillating = False
    if len(deltas) >= 10:
        # Use a rolling window to smooth noise
        window = max(10, len(deltas) // 20)
        rolling = pd.Series(deltas).rolling(window, min_periods=1).mean()
        # Check if the second half average is less than the first half average
        mid = len(rolling) // 2
        first_half_mean = rolling.iloc[:mid].mean()
        second_half_mean = rolling.iloc[mid:].mean()
        monotone_damped = second_half_mean < first_half_mean * 0.8

        # Check oscillation: count sign changes in delta differences
        delta_diffs = np.diff(deltas)
        sign_changes = np.sum(np.diff(np.sign(delta_diffs)) != 0)
        oscillation_ratio = sign_changes / max(len(delta_diffs) - 1, 1)
        oscillating = oscillation_ratio > 0.4  # >40% of steps change direction

    # Absorbing state
    absorbing_state = converged and final_active == 0

    # Weight concentration
    initial_entropy = panel["weight_entropy"].iloc[0]
    final_entropy = panel["weight_entropy"].iloc[-1]
    max_top1 = panel["top1_weight"].max()

    # Determine primary behaviour label
    if absorbing_state:
        label = "absorbing_collapse"
    elif converged and not active_collapse:
        label = "healthy_convergence"
    elif converged and active_collapse:
        label = "convergence_via_collapse"
    elif not converged and monotone_damped:
        label = "slow_damped_non_convergence"
    elif not converged and oscillating:
        label = "oscillating_non_convergence"
    elif not converged:
        label = "stalled_non_convergence"
    else:
        label = "unclassified"

    return BehaviourClassification(
        year=year,
        converged=converged,
        n_iterations=n,
        final_delta=final_delta,
        initial_mae=initial_mae,
        best_mae=best_mae,
        best_mae_iteration=best_mae_iter,
        final_mae=final_mae,
        mae_degraded_after_best=mae_degraded,
        initial_active=initial_active,
        final_active=final_active,
        min_active=min_active,
        active_collapse=active_collapse,
        active_collapse_iteration=active_collapse_iter,
        monotone_damped=monotone_damped,
        oscillating=oscillating,
        absorbing_state=absorbing_state,
        initial_entropy=initial_entropy,
        final_entropy=final_entropy,
        max_top1_weight=max_top1,
        behaviour_label=label,
    )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def format_classification_report(cls: BehaviourClassification) -> str:
    """Format a human-readable report for a behaviour classification."""
    lines = [
        f"{'=' * 70}",
        f"  YEAR {cls.year}  —  {cls.behaviour_label.upper().replace('_', ' ')}",
        f"{'=' * 70}",
        "",
        "  Convergence:",
        f"    converged:        {cls.converged}",
        f"    iterations:       {cls.n_iterations}",
        f"    final delta:      {cls.final_delta:.6f}",
        "",
        "  Accuracy (MAE):",
        f"    initial:          {cls.initial_mae:.4f} EUR/MWh",
        f"    best:             {cls.best_mae:.4f} EUR/MWh  (iter {cls.best_mae_iteration})",
        f"    final:            {cls.final_mae:.4f} EUR/MWh",
        f"    degraded:         {cls.mae_degraded_after_best}",
        "",
        "  Active Strategies:",
        f"    initial:          {cls.initial_active}",
        f"    final:            {cls.final_active}",
        f"    minimum:          {cls.min_active}",
        f"    collapse (<=2):   {cls.active_collapse}",
    ]
    if cls.active_collapse_iteration is not None:
        lines.append(f"    collapse at iter: {cls.active_collapse_iteration}")
    lines.extend(
        [
            "",
            "  Dynamics:",
            f"    monotone damped:  {cls.monotone_damped}",
            f"    oscillating:      {cls.oscillating}",
            f"    absorbing state:  {cls.absorbing_state}",
            "",
            "  Weight Concentration:",
            f"    initial entropy:  {cls.initial_entropy:.4f}",
            f"    final entropy:    {cls.final_entropy:.4f}",
            f"    max top-1 weight: {cls.max_top1_weight:.4f}",
            "",
        ]
    )
    return "\n".join(lines)


def classification_to_dict(cls: BehaviourClassification) -> dict:
    """Convert a BehaviourClassification to a flat dict for CSV export."""
    return {
        "year": cls.year,
        "converged": cls.converged,
        "n_iterations": cls.n_iterations,
        "final_delta": cls.final_delta,
        "initial_mae": cls.initial_mae,
        "best_mae": cls.best_mae,
        "best_mae_iteration": cls.best_mae_iteration,
        "final_mae": cls.final_mae,
        "mae_degraded_after_best": cls.mae_degraded_after_best,
        "initial_active": cls.initial_active,
        "final_active": cls.final_active,
        "min_active": cls.min_active,
        "active_collapse": cls.active_collapse,
        "active_collapse_iteration": cls.active_collapse_iteration,
        "monotone_damped": cls.monotone_damped,
        "oscillating": cls.oscillating,
        "absorbing_state": cls.absorbing_state,
        "initial_entropy": cls.initial_entropy,
        "final_entropy": cls.final_entropy,
        "max_top1_weight": cls.max_top1_weight,
        "behaviour_label": cls.behaviour_label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load market artifacts, compute metrics, classify behaviour, and save."""
    parser = argparse.ArgumentParser(
        description="Phase 10b: Behaviour Inventory — extract per-iteration metrics"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024, 2025],
        help="Years to process (default: 2024 2025)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    PHASE10_DIR.mkdir(parents=True, exist_ok=True)

    all_panels: list[pd.DataFrame] = []
    all_summaries: list[dict] = []

    for year in args.years:
        pkl_path = RESULTS_DIR / f"market_{year}.pkl"
        if not pkl_path.exists():
            logger.warning("Skipping year %d: %s not found", year, pkl_path)
            continue

        logger.info("Loading %s", pkl_path)
        with open(pkl_path, "rb") as f:
            result: FuturesMarketResult = pickle.load(f)

        eq = result.equilibrium

        # Load real prices from the Phase 8 forecast pickles (canonical source)
        forecast_pkl = PHASE8_DIR / f"forecasts_{year}.pkl"
        if not forecast_pkl.exists():
            logger.error(
                "Phase 8 forecast pickle not found: %s — cannot get real_prices",
                forecast_pkl,
            )
            continue

        with open(forecast_pkl, "rb") as f:
            phase8_data: dict = pickle.load(f)

        real_prices: pd.Series = phase8_data["real_prices"]

        logger.info(
            "Year %d: %d iterations, %d dates, converged=%s",
            year,
            len(eq.iterations),
            len(real_prices),
            eq.converged,
        )

        # Extract per-iteration metrics
        panel = extract_iteration_panel(eq, real_prices)
        panel.insert(0, "year", year)
        all_panels.append(panel)

        # Classify behaviour
        cls = classify_behaviour(
            panel=panel,
            year=year,
            converged=eq.converged,
            final_delta=eq.convergence_delta,
        )
        all_summaries.append(classification_to_dict(cls))

        # Print report
        print(format_classification_report(cls))

    # Save iteration-level panel
    if all_panels:
        combined_panel = pd.concat(all_panels, ignore_index=True)
        panel_path = PHASE10_DIR / "behaviour_inventory.csv"
        combined_panel.to_csv(panel_path, index=False)
        logger.info("Saved iteration panel: %s (%d rows)", panel_path, len(combined_panel))

    # Save run-level summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = PHASE10_DIR / "behaviour_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved behaviour summary: %s", summary_path)

    # Print high-priority behaviours
    print("\n" + "=" * 70)
    print("  HIGH-PRIORITY BEHAVIOURS FOR PHASE 10c-10e")
    print("=" * 70)
    for s in all_summaries:
        year = s["year"]
        label = s["behaviour_label"]
        print(f"\n  Year {year}: {label}")
        if s["absorbing_state"]:
            print("    -> Absorbing state: all strategies eliminated")
        if s["active_collapse"]:
            print(f"    -> Active strategy collapse: {s['initial_active']} -> {s['final_active']}")
        if s["mae_degraded_after_best"]:
            print(
                f"    -> MAE degradation: best {s['best_mae']:.2f} "
                f"(iter {s['best_mae_iteration']}) -> final {s['final_mae']:.2f}"
            )
        if not s["converged"]:
            print(
                f"    -> Non-convergence: delta={s['final_delta']:.4f} "
                f"after {s['n_iterations']} iters"
            )

    print()


if __name__ == "__main__":
    main()
