"""Phase 10: EMA price-update experiment sweep.

Tests the idea of blending the spec-computed new price with the previous
market price across iterations:

    P_{k+1} = alpha * P_weighted_k + (1 - alpha) * P_k

where P_weighted_k is the profit-weighted forecast average (the raw spec
output) and P_k is the current market price.  alpha=1.0 is the unmodified
spec; smaller alpha values damp the per-iteration price jump.

The strategy weights are computed exactly as per the spec (linear
max-profit weighting) — only the *published* price is blended.

This was explored in Phase 8 (B1 series) under the old engine with removed
parameters (weight_mode, weight_cap, price_mode, convergence_window).  This
script re-runs a clean sweep against the current reverted spec engine, using
the same forecast pickles from data/results/phase8/.

Configurations tested:
  BASE          alpha=1.0   (pure spec, no blending)
  H1_a*         alpha ∈ {0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}
  H2_fine       alpha ∈ {0.95, 0.9, 0.85}  (fine grid near 1.0)
  H3_symmetric  alpha ∈ {0.5}  (the specific value requested: 50/50 blend)

Results are saved to data/results/phase10/results.csv.

Usage:
    uv run python scripts/phase10_ema_price_update.py
    uv run python scripts/phase10_ema_price_update.py --year 2024
    uv run python scripts/phase10_ema_price_update.py --max-iters 200
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE10_DIR = RESULTS_DIR / "phase10"

CONVERGENCE_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------


def _build_experiments() -> list[dict]:
    """Return list of experiment config dicts.

    Each dict has: id, label, alpha.
    alpha=1.0 means no blending (pure spec).
    alpha<1.0 means P_{k+1} = alpha * P_weighted + (1-alpha) * P_k.
    """
    experiments: list[dict] = []

    def _add(exp_id: str, label: str, alpha: float = 1.0) -> None:
        experiments.append({"id": exp_id, "label": label, "alpha": alpha})

    # Baseline: unmodified spec (alpha=1.0 = no EMA blending)
    _add("BASE", "Spec baseline (no blending)", alpha=1.0)

    # H2: fine grid near 1.0
    for a in [0.95, 0.9, 0.85]:
        tag = f"{int(a * 100):02d}"
        _add(f"H2_a{tag}", f"EMA alpha={a} (fine)", alpha=a)

    # H1: coarse sweep
    for a in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        tag = f"{int(a * 100):02d}"
        _add(f"H1_a{tag}", f"EMA alpha={a}", alpha=a)

    return experiments


# ---------------------------------------------------------------------------
# Single-experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    cfg: dict,
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    max_iterations: int,
) -> dict:
    """Run one EMA-blending configuration and return a metrics dict.

    Uses the vectorised ``_vec_iteration`` path for speed — the EMA blend
    is applied *after* the spec step-4 output, before updating current_prices.
    """
    from energy_modelling.backtest.futures_market_engine import (
        FuturesMarketEquilibrium,
        FuturesMarketIteration,
        _build_forecast_matrix,
        _vec_iteration,
    )

    alpha: float = cfg["alpha"]
    current_prices = initial_prices.copy().astype(float)
    index = current_prices.index

    # Pre-build (S × T) forecast matrix once
    fm = _build_forecast_matrix(strategy_forecasts, index, real_prices)
    strategy_names = fm.strategy_names

    iterations: list[FuturesMarketIteration] = []
    converged = False
    delta = float("inf")
    delta_history: list[float] = []

    for k in range(max_iterations):
        market_vec = current_prices.to_numpy(dtype=np.float64)

        # Spec steps 1-4: vectorised computation of profit-weighted price
        new_vec, profits_arr, weights_arr = _vec_iteration(market_vec, fm)

        # EMA blend: P_{k+1} = alpha * P_spec + (1 - alpha) * P_k
        blended_vec = alpha * new_vec + (1.0 - alpha) * market_vec if alpha < 1.0 else new_vec

        published = pd.Series(blended_vec, index=index, name="market_price")

        profits_dict = dict(zip(strategy_names, profits_arr.tolist(), strict=True))
        weights_dict = dict(zip(strategy_names, weights_arr.tolist(), strict=True))
        active = [n for n, w in zip(strategy_names, weights_arr, strict=True) if w > 0.0]

        iterations.append(
            FuturesMarketIteration(
                iteration=k,
                market_prices=published,
                strategy_profits=profits_dict,
                strategy_weights=weights_dict,
                active_strategies=active,
            )
        )

        delta = float(np.abs(blended_vec - market_vec).max())
        delta_history.append(delta)
        current_prices = published

        if delta < CONVERGENCE_THRESHOLD:
            converged = True
            break

    eq = FuturesMarketEquilibrium(
        iterations=iterations,
        final_market_prices=current_prices,
        final_weights=iterations[-1].strategy_weights if iterations else {},
        converged=converged,
        convergence_delta=delta,
    )

    # MAE of converged price vs real prices
    aligned_real = real_prices.reindex(eq.final_market_prices.index)
    mae = float((eq.final_market_prices - aligned_real).abs().mean())

    # MAE at iteration 0 (baseline before any update)
    mae_iter0 = float((initial_prices.reindex(aligned_real.index) - aligned_real).abs().mean())

    # Monotone convergence quality: count consecutive strictly decreasing tail
    monotone_tail = 0
    for i in range(len(delta_history) - 1, 0, -1):
        if delta_history[i] < delta_history[i - 1]:
            monotone_tail += 1
        else:
            break

    return {
        "id": cfg["id"],
        "label": cfg["label"],
        "alpha": cfg["alpha"],
        "converged": converged,
        "n_iterations": len(iterations),
        "final_delta": round(delta, 6),
        "mae": round(mae, 4),
        "mae_iter0": round(mae_iter0, 4),
        "monotone_tail": monotone_tail,
        # Full delta history — saved separately as delta_history_{id}_{year}.csv
        "_delta_history": delta_history,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(df: pd.DataFrame) -> None:
    """Print ranked results table."""
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year].copy()
        converged = sub[sub["converged"]].sort_values("mae")
        not_conv = sub[~sub["converged"]].sort_values("final_delta")

        print(f"\n{'=' * 80}")
        print(f"  Year {year} — {len(converged)} converged / {len(sub)} total")
        print(f"{'=' * 80}")

        if converged.empty:
            print("  No experiments converged.")
        else:
            print(
                f"  {'ID':<18} {'alpha':>6} {'MAE':>8} {'Iters':>6} "
                f"{'Delta':>10} {'MonoTail':>9}  Label"
            )
            print(f"  {'-' * 18} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 10} {'-' * 9}  {'-' * 30}")
            for _, r in converged.iterrows():
                print(
                    f"  {r['id']:<18} {r['alpha']:>6.2f} {r['mae']:>8.2f} "
                    f"{int(r['n_iterations']):>6} {r['final_delta']:>10.4f} "
                    f"{int(r['monotone_tail']):>9}  {r['label']}"
                )

        if not not_conv.empty:
            print("\n  Non-converged (top 5 by final delta):")
            print(
                f"  {'ID':<18} {'alpha':>6} {'MAE':>8} {'Iters':>6} {'Delta':>10} {'MonoTail':>9}"
            )
            for _, r in not_conv.head(5).iterrows():
                print(
                    f"  {r['id']:<18} {r['alpha']:>6.2f} {r['mae']:>8.2f} "
                    f"{int(r['n_iterations']):>6} {r['final_delta']:>10.4f} "
                    f"{int(r['monotone_tail']):>9}"
                )

    # Cross-year summary: which alphas converge on BOTH years?
    converged_ids = set()
    for year in df["year"].unique():
        ids_this_year = set(df[(df["year"] == year) & df["converged"]]["id"])
        if not converged_ids:
            converged_ids = ids_this_year
        else:
            converged_ids &= ids_this_year

    print(f"\n{'=' * 80}")
    if converged_ids:
        both = df[df["id"].isin(converged_ids)].sort_values(["id", "year"])
        print(f"  CONVERGED ON BOTH YEARS ({len(converged_ids)} configs):")
        print(f"  {'ID':<18} {'alpha':>6} {'year':>6} {'MAE':>8} {'Iters':>6} {'Delta':>10}")
        print(f"  {'-' * 18} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 10}")
        for _, r in both.iterrows():
            print(
                f"  {r['id']:<18} {r['alpha']:>6.2f} {int(r['year']):>6} "
                f"{r['mae']:>8.2f} {int(r['n_iterations']):>6} "
                f"{r['final_delta']:>10.4f}"
            )
    else:
        print("  No configuration converged on BOTH years.")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10: EMA price-update sweep.")
    parser.add_argument(
        "--year",
        type=int,
        choices=[2024, 2025],
        default=None,
        help="Run only one year (default: both).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=200,
        help="Max iterations per experiment (default: 200).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    PHASE10_DIR.mkdir(parents=True, exist_ok=True)
    years = [args.year] if args.year else [2024, 2025]
    experiments = _build_experiments()
    logger.info("Phase 10 EMA sweep: %d configs × %d year(s)", len(experiments), len(years))

    all_rows: list[dict] = []

    for year in years:
        fpath = PHASE8_DIR / f"forecasts_{year}.pkl"
        if not fpath.exists():
            logger.error("Missing %s — run phase8_collect_forecasts.py first", fpath)
            sys.exit(1)

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        strategy_forecasts: dict[str, dict] = data["strategy_forecasts"]
        real_prices: pd.Series = data["real_prices"]
        initial_prices: pd.Series = data["initial_prices"]

        logger.info(
            "Year %d: %d strategies, %d eval days",
            year,
            len(strategy_forecasts),
            len(real_prices),
        )

        for i, cfg in enumerate(experiments, 1):
            t0 = time.perf_counter()
            try:
                row = run_experiment(
                    cfg=cfg,
                    strategy_forecasts=strategy_forecasts,
                    real_prices=real_prices,
                    initial_prices=initial_prices,
                    max_iterations=args.max_iters,
                )
                row["year"] = year
                row["elapsed_s"] = round(time.perf_counter() - t0, 2)
                all_rows.append(row)

                status = "CONV" if row["converged"] else "----"
                logger.info(
                    "[%d/%d] %s (%d) %s  iters=%d  delta=%.4f  MAE=%.2f  mono_tail=%d",
                    i,
                    len(experiments),
                    cfg["id"],
                    year,
                    status,
                    row["n_iterations"],
                    row["final_delta"],
                    row["mae"],
                    row["monotone_tail"],
                )
            except Exception:
                logger.exception("Experiment %s year %d failed", cfg["id"], year)
                all_rows.append(
                    {
                        "id": cfg["id"],
                        "label": cfg["label"],
                        "alpha": cfg["alpha"],
                        "year": year,
                        "converged": False,
                        "n_iterations": 0,
                        "final_delta": float("inf"),
                        "mae": float("inf"),
                        "mae_iter0": float("inf"),
                        "monotone_tail": 0,
                        "elapsed_s": 0.0,
                        "error": "FAILED",
                    }
                )

    df = pd.DataFrame(all_rows)

    # Save summary CSV (drop the internal delta history column)
    summary_df = df.drop(columns=["_delta_history"], errors="ignore")
    out_path = PHASE10_DIR / "results.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # Save tidy long-form delta history for visualisation
    delta_rows: list[dict] = []
    for row in all_rows:
        hist = row.get("_delta_history", [])
        for iteration, d in enumerate(hist):
            delta_rows.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "alpha": row["alpha"],
                    "year": row["year"],
                    "iteration": iteration,
                    "delta": d,
                }
            )
    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        delta_path = PHASE10_DIR / "delta_history.csv"
        delta_df.to_csv(delta_path, index=False)
        logger.info("Delta history saved to %s", delta_path)

    _print_summary(summary_df)


if __name__ == "__main__":
    main()
