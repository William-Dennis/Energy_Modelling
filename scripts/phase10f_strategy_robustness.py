"""Phase 10f: Strategy Robustness Analysis.

Compares standalone backtest performance against market-adjusted performance
for all 67 strategies.  Computes market-contribution metrics via leave-one-out
experiments, and classifies each strategy as robust, standalone-only, redundant,
or destabilising.

Analyses:
  1. **Standalone PnL** -- total profit under the sign-rule at initial prices.
  2. **Market-adjusted PnL** -- total profit at converged/final market prices.
  3. **Leave-one-out MAE** -- change in final market MAE when the strategy is
     removed.  Positive delta_mae = strategy was helping (MAE gets worse).
  4. **Weight stability** -- variance of the strategy's weight across iterations.
  5. **Redundancy** -- max correlation of this strategy's forecasts with any
     other strategy in the pool.
  6. **Classification** -- robust / standalone-only / redundant / destabilising.

Results are saved to ``data/results/phase10/strategy_robustness.csv``.

Usage::

    uv run python scripts/phase10f_strategy_robustness.py
    uv run python scripts/phase10f_strategy_robustness.py --year 2024
    uv run python scripts/phase10f_strategy_robustness.py --skip-loo
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
    _build_forecast_matrix,
    _vec_iteration,
)
from energy_modelling.backtest.futures_market_runner import FuturesMarketResult  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE10_DIR = RESULTS_DIR / "phase10"

CONVERGENCE_THRESHOLD = 0.01
DEFAULT_EMA_ALPHA = 0.1
DEFAULT_MAX_ITERS = 500


# ---------------------------------------------------------------------------
# Market simulation (reused from Phase 10c pattern)
# ---------------------------------------------------------------------------


def run_market_fast(
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    max_iterations: int = DEFAULT_MAX_ITERS,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> dict:
    """Run a market sim and return compact metrics dict.

    Uses the vectorised ``_vec_iteration`` path for speed.
    Returns dict with: converged, n_iterations, mae, rmse, bias, final_delta,
    n_strategies, final_active_count, final_weight_entropy, final_top1_weight,
    plus per-strategy final weights and profits.
    """
    current_prices = initial_prices.copy().astype(float)
    index = current_prices.index

    fm = _build_forecast_matrix(strategy_forecasts, index, real_prices)
    strategy_names = fm.strategy_names

    converged = False
    delta = float("inf")
    last_profits: dict[str, float] = {}
    last_weights: dict[str, float] = {}
    n_iters = 0

    for k in range(max_iterations):
        market_vec = current_prices.to_numpy(dtype=np.float64)
        new_vec, profits_arr, weights_arr = _vec_iteration(market_vec, fm)

        blended_vec = (
            ema_alpha * new_vec + (1.0 - ema_alpha) * market_vec if ema_alpha < 1.0 else new_vec
        )

        delta = float(np.abs(blended_vec - market_vec).max())
        current_prices = pd.Series(blended_vec, index=index, name="market_price")

        last_profits = dict(zip(strategy_names, profits_arr.tolist(), strict=True))
        last_weights = dict(zip(strategy_names, weights_arr.tolist(), strict=True))
        n_iters = k + 1

        if delta < CONVERGENCE_THRESHOLD:
            converged = True
            break

    # Final accuracy metrics
    aligned_real = real_prices.reindex(index)
    errors = current_prices - aligned_real
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors**2).mean()))
    bias = float(errors.mean())

    pos_w = np.array([w for w in last_weights.values() if w > 0.0])
    entropy = float(-np.sum(pos_w * np.log(pos_w))) if len(pos_w) > 0 else 0.0
    top1 = float(pos_w.max()) if len(pos_w) > 0 else 0.0
    active_count = int(np.sum(np.array(list(last_weights.values())) > 0))

    return {
        "converged": converged,
        "n_iterations": n_iters,
        "final_delta": round(delta, 6),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "n_strategies": len(strategy_names),
        "final_active_count": active_count,
        "final_weight_entropy": round(entropy, 4),
        "final_top1_weight": round(top1, 4),
        "final_profits": last_profits,
        "final_weights": last_weights,
    }


# ---------------------------------------------------------------------------
# Standalone PnL computation
# ---------------------------------------------------------------------------


def compute_standalone_pnl(
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
) -> dict[str, float]:
    """Compute per-strategy standalone PnL using sign-rule at initial prices.

    For each strategy and each date:
      q = sign(forecast - initial_price)
      pnl += q * (real_price - initial_price)

    Returns dict mapping strategy name to total PnL.
    """
    pnl: dict[str, float] = {}
    for name, forecasts in strategy_forecasts.items():
        total = 0.0
        for date, forecast in forecasts.items():
            init_p = initial_prices.get(date, None)
            real_p = real_prices.get(date, None)
            if init_p is None or real_p is None:
                continue
            q = 1.0 if forecast > init_p else (-1.0 if forecast < init_p else 0.0)
            total += q * (real_p - init_p)
        pnl[name] = round(total, 4)
    return pnl


# ---------------------------------------------------------------------------
# Weight stability from saved equilibrium
# ---------------------------------------------------------------------------


def compute_weight_stability(
    equilibrium: FuturesMarketEquilibrium,
    strategy_names: list[str],
) -> dict[str, float]:
    """Compute per-strategy weight variance across iterations.

    Lower variance = more stable weight = more consistent strategy.
    """
    weight_series: dict[str, list[float]] = {n: [] for n in strategy_names}
    for it in equilibrium.iterations:
        for name in strategy_names:
            weight_series[name].append(it.strategy_weights.get(name, 0.0))

    return {name: round(float(np.var(ws)), 8) if ws else 0.0 for name, ws in weight_series.items()}


# ---------------------------------------------------------------------------
# Forecast redundancy
# ---------------------------------------------------------------------------


def compute_forecast_redundancy(
    strategy_forecasts: dict[str, dict],
) -> dict[str, float]:
    """For each strategy, compute max absolute correlation with any other strategy.

    Higher redundancy = more similar to another strategy in the pool.
    """
    names = sorted(strategy_forecasts.keys())
    if len(names) <= 1:
        return {n: 0.0 for n in names}

    # Build forecast matrix
    rows = {}
    for name in names:
        rows[name] = pd.Series(strategy_forecasts[name])
    fm = pd.DataFrame(rows).T.sort_index(axis=1)

    # Pairwise correlation
    corr = fm.T.corr()

    redundancy: dict[str, float] = {}
    for name in names:
        if name not in corr.index:
            redundancy[name] = 0.0
            continue
        row = corr.loc[name].drop(name, errors="ignore")
        # NaN correlations (e.g. constant strategy) are treated as 0
        abs_row = row.abs().dropna()
        redundancy[name] = round(float(abs_row.max()), 4) if not abs_row.empty else 0.0

    return redundancy


# ---------------------------------------------------------------------------
# Strategy classification
# ---------------------------------------------------------------------------


def classify_strategy(
    standalone_pnl: float,
    market_adjusted_pnl: float,
    loo_mae_delta: float | None,
    redundancy: float,
    standalone_median: float,
    market_median: float,
) -> str:
    """Classify a strategy into one of four categories.

    Parameters
    ----------
    standalone_pnl:
        Total standalone PnL.
    market_adjusted_pnl:
        Total market-adjusted PnL (from final iteration profits).
    loo_mae_delta:
        MAE change when strategy is removed (positive = MAE got worse = strategy helps).
        None if leave-one-out was skipped.
    redundancy:
        Max absolute forecast correlation with another strategy.
    standalone_median:
        Median standalone PnL across all strategies.
    market_median:
        Median market-adjusted PnL across all strategies.
    """
    strong_standalone = standalone_pnl > standalone_median
    strong_market = market_adjusted_pnl > market_median

    # Destabilising: removing the strategy improves MAE
    if loo_mae_delta is not None and loo_mae_delta < -0.01:
        return "destabilising"

    # Redundant: high correlation with another strategy
    if redundancy > 0.95:
        return "redundant"

    # Robust: strong in both standalone and market-adjusted
    if strong_standalone and strong_market:
        return "robust"

    # Standalone-only: strong standalone but weak market-adjusted
    if strong_standalone and not strong_market:
        return "standalone_only"

    # Weak: below median in both
    return "weak"


# ---------------------------------------------------------------------------
# Leave-one-out analysis
# ---------------------------------------------------------------------------


def run_leave_one_out(
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    baseline_mae: float,
    max_iterations: int = DEFAULT_MAX_ITERS,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> dict[str, float]:
    """Run leave-one-out: remove each strategy, measure MAE change.

    Returns dict mapping strategy name to delta_MAE (positive = strategy was helping).
    """
    strategy_names = sorted(strategy_forecasts.keys())
    loo_results: dict[str, float] = {}

    for i, name in enumerate(strategy_names):
        # Remove this strategy
        reduced = {k: v for k, v in strategy_forecasts.items() if k != name}
        if not reduced:
            loo_results[name] = 0.0
            continue

        result = run_market_fast(reduced, real_prices, initial_prices, max_iterations, ema_alpha)
        loo_mae = result["mae"]
        # Positive delta = removing strategy made MAE worse = strategy was helping
        delta_mae = loo_mae - baseline_mae
        loo_results[name] = round(delta_mae, 4)

        if (i + 1) % 10 == 0:
            logger.info("  LOO progress: %d/%d strategies", i + 1, len(strategy_names))

    return loo_results


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyse_year(
    year: int,
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    equilibrium: FuturesMarketEquilibrium,
    skip_loo: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERS,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> pd.DataFrame:
    """Full robustness analysis for one year. Returns a DataFrame."""
    strategy_names = sorted(strategy_forecasts.keys())

    # 1. Standalone PnL
    logger.info("Computing standalone PnL...")
    standalone = compute_standalone_pnl(strategy_forecasts, real_prices, initial_prices)

    # 2. Market-adjusted PnL (from the saved equilibrium's final iteration)
    last_it = equilibrium.iterations[-1] if equilibrium.iterations else None
    market_adjusted = (
        {n: last_it.strategy_profits.get(n, 0.0) for n in strategy_names}
        if last_it
        else {n: 0.0 for n in strategy_names}
    )

    # 3. Market-adjusted weights
    final_weights = (
        {n: last_it.strategy_weights.get(n, 0.0) for n in strategy_names}
        if last_it
        else {n: 0.0 for n in strategy_names}
    )

    # 4. Weight stability
    logger.info("Computing weight stability...")
    weight_stability = compute_weight_stability(equilibrium, strategy_names)

    # 5. Forecast redundancy
    logger.info("Computing forecast redundancy...")
    redundancy = compute_forecast_redundancy(strategy_forecasts)

    # 6. Leave-one-out MAE (most expensive step)
    loo_mae_deltas: dict[str, float | None] = {}
    if not skip_loo:
        # First, get baseline MAE from running the full market
        logger.info("Running baseline market for LOO comparison...")
        baseline = run_market_fast(
            strategy_forecasts, real_prices, initial_prices, max_iterations, ema_alpha
        )
        baseline_mae = baseline["mae"]
        logger.info("Baseline MAE: %.4f", baseline_mae)

        logger.info("Running leave-one-out analysis (67 runs)...")
        t0 = time.time()
        loo_mae_deltas = run_leave_one_out(
            strategy_forecasts,
            real_prices,
            initial_prices,
            baseline_mae,
            max_iterations,
            ema_alpha,
        )
        logger.info("LOO completed in %.1fs", time.time() - t0)
    else:
        logger.info("Skipping leave-one-out analysis (--skip-loo)")
        loo_mae_deltas = {n: None for n in strategy_names}

    # 7. Classification
    standalone_median = float(np.median(list(standalone.values())))
    market_median = float(np.median(list(market_adjusted.values())))

    rows: list[dict] = []
    for name in strategy_names:
        sa_pnl = standalone.get(name, 0.0)
        ma_pnl = market_adjusted.get(name, 0.0)
        loo_delta = loo_mae_deltas.get(name)
        red = redundancy.get(name, 0.0)

        classification = classify_strategy(
            sa_pnl, ma_pnl, loo_delta, red, standalone_median, market_median
        )

        # Rank
        sa_rank = (
            sorted(standalone.values(), reverse=True).index(sa_pnl) + 1
            if sa_pnl in standalone.values()
            else -1
        )
        ma_rank = (
            sorted(market_adjusted.values(), reverse=True).index(ma_pnl) + 1
            if ma_pnl in market_adjusted.values()
            else -1
        )

        rows.append(
            {
                "year": year,
                "strategy": name,
                "standalone_pnl": round(sa_pnl, 4),
                "standalone_rank": sa_rank,
                "market_adjusted_pnl": round(ma_pnl, 4),
                "market_adjusted_rank": ma_rank,
                "rank_change": sa_rank - ma_rank,
                "final_weight": round(final_weights.get(name, 0.0), 6),
                "weight_variance": weight_stability.get(name, 0.0),
                "max_forecast_correlation": red,
                "loo_mae_delta": round(loo_delta, 4) if loo_delta is not None else None,
                "classification": classification,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_summary(results: pd.DataFrame) -> None:
    """Print a human-readable summary of strategy robustness analysis."""
    for year in sorted(results["year"].unique()):
        yr = results[results["year"] == year]
        print(f"\n{'=' * 80}")
        print(f"  YEAR {year}: STRATEGY ROBUSTNESS ANALYSIS ({len(yr)} strategies)")
        print(f"{'=' * 80}")

        # Classification counts
        print("\n  Classification Distribution:")
        for cls in ["robust", "standalone_only", "redundant", "destabilising", "weak"]:
            count = len(yr[yr["classification"] == cls])
            if count > 0:
                print(f"    {cls:<20s}: {count}")

        # Top-5 most robust
        robust = yr[yr["classification"] == "robust"].sort_values("standalone_pnl", ascending=False)
        if not robust.empty:
            print("\n  Top-5 Most Robust Strategies:")
            hdr = (
                f"  {'Strategy':<30s} {'SA PnL':>10s} {'MA PnL':>10s}"
                f" {'SA Rank':>8s} {'MA Rank':>8s} {'LOO dMAE':>10s}"
            )
            print(hdr)
            for _, r in robust.head(5).iterrows():
                loo = f"{r['loo_mae_delta']:+.4f}" if r["loo_mae_delta"] is not None else "N/A"
                print(
                    f"  {r['strategy']:<30s} {r['standalone_pnl']:>10.2f} "
                    f"{r['market_adjusted_pnl']:>10.2f} {r['standalone_rank']:>8d} "
                    f"{r['market_adjusted_rank']:>8d} {loo:>10s}"
                )

        # Top-5 most problematic (destabilising first, then redundant)
        problematic = yr[yr["classification"].isin(["destabilising", "redundant"])].sort_values(
            "loo_mae_delta", ascending=True, na_position="last"
        )
        if not problematic.empty:
            print("\n  Top-5 Most Problematic Strategies:")
            hdr2 = (
                f"  {'Strategy':<30s} {'Class':<16s} {'SA PnL':>10s}"
                f" {'LOO dMAE':>10s} {'MaxCorr':>10s}"
            )
            print(hdr2)
            for _, r in problematic.head(5).iterrows():
                loo = f"{r['loo_mae_delta']:+.4f}" if r["loo_mae_delta"] is not None else "N/A"
                print(
                    f"  {r['strategy']:<30s} {r['classification']:<16s} "
                    f"{r['standalone_pnl']:>10.2f} {loo:>10s} "
                    f"{r['max_forecast_correlation']:>10.4f}"
                )

        # Biggest rank shifts
        yr_ranked = yr.sort_values("rank_change", key=abs, ascending=False)
        print("\n  Biggest Rank Shifts (standalone -> market-adjusted):")
        print(f"  {'Strategy':<30s} {'SA Rank':>8s} {'MA Rank':>8s} {'Change':>8s}")
        for _, r in yr_ranked.head(5).iterrows():
            print(
                f"  {r['strategy']:<30s} {r['standalone_rank']:>8d} "
                f"{r['market_adjusted_rank']:>8d} {r['rank_change']:>+8d}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10f: strategy robustness analysis.")
    parser.add_argument(
        "--year",
        type=int,
        choices=[2024, 2025],
        default=None,
        help="Run only one year (default: both).",
    )
    parser.add_argument(
        "--skip-loo",
        action="store_true",
        help="Skip leave-one-out analysis (faster, no market contribution metrics).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=DEFAULT_MAX_ITERS,
        help="Maximum iterations for market simulation.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    PHASE10_DIR.mkdir(parents=True, exist_ok=True)
    years = [args.year] if args.year else [2024, 2025]

    all_results: list[pd.DataFrame] = []

    for year in years:
        # Load forecasts
        fpath = PHASE8_DIR / f"forecasts_{year}.pkl"
        if not fpath.exists():
            logger.error("Missing %s", fpath)
            continue
        with open(fpath, "rb") as f:
            data = pickle.load(f)

        strategy_forecasts: dict[str, dict] = data["strategy_forecasts"]
        real_prices: pd.Series = data["real_prices"]
        initial_prices: pd.Series = data["initial_prices"]

        # Load market result
        mpath = RESULTS_DIR / f"market_{year}.pkl"
        if not mpath.exists():
            logger.error("Missing %s", mpath)
            continue
        with open(mpath, "rb") as f:
            market_result: FuturesMarketResult = pickle.load(f)

        equilibrium = market_result.equilibrium

        logger.info(
            "Year %d: %d strategies, %d dates, %d iterations",
            year,
            len(strategy_forecasts),
            len(real_prices),
            len(equilibrium.iterations),
        )

        result = analyse_year(
            year,
            strategy_forecasts,
            real_prices,
            initial_prices,
            equilibrium,
            skip_loo=args.skip_loo,
            max_iterations=args.max_iters,
        )
        all_results.append(result)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = PHASE10_DIR / "strategy_robustness.csv"
        combined.to_csv(out_path, index=False)
        logger.info("Saved strategy robustness results: %s", out_path)
        print_summary(combined)

    print(f"\n{'=' * 80}")
    print("  STRATEGY ROBUSTNESS ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
