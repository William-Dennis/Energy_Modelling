"""Phase 8b: Extended experiment sweep with stricter convergence criterion.

Re-runs all Phase 8a experiments under the stricter window=3 convergence
criterion, plus new F-series experiments.  Results saved to
data/results/phase8/results_extended.csv.

New ideas (F series):
  F1  : K=5 running-avg + window=3 (new production baseline)
  F2  : Larger running-avg windows: K=7, 10, 15, 20
  F3a : Adaptive alpha — alpha proportional to 1/(1+delta), decays as delta shrinks
  F3b : Adaptive alpha — step-size rule: alpha = min(1, threshold/delta)
  F4  : K=7/10 + alpha=0.3 combination
  F5  : Original alpha sweep re-run with window=3 (may now converge)
  F6  : K=5 + alpha sweep (combining the two mechanisms)

Usage:
    uv run python scripts/phase8b_experiments.py
    uv run python scripts/phase8b_experiments.py --year 2024
    uv run python scripts/phase8b_experiments.py --max-iters 100
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------


def _build_experiments() -> list[dict]:
    """Return list of experiment config dicts."""
    experiments: list[dict] = []

    def _add(
        exp_id: str,
        label: str,
        alpha: float = 1.0,
        weight_mode: str = "linear",
        weight_cap: float = 1.0,
        price_mode: str = "mean",
        init_strategy: str = "last_settlement",
        running_avg_k: int | None = None,
        convergence_window: int = 3,
        adaptive_alpha: str | None = None,
    ) -> None:
        experiments.append(
            {
                "id": exp_id,
                "label": label,
                "alpha": alpha,
                "weight_mode": weight_mode,
                "weight_cap": weight_cap,
                "price_mode": price_mode,
                "init_strategy": init_strategy,
                "running_avg_k": running_avg_k,
                "convergence_window": convergence_window,
                "adaptive_alpha": adaptive_alpha,
            }
        )

    # ------------------------------------------------------------------
    # Reproduce Phase 8a results under window=3
    # ------------------------------------------------------------------
    _add("BASE_w3", "Baseline window=3", convergence_window=3)

    for alpha_val in [0.1, 0.2, 0.3]:
        _add(
            f"B1_a{int(alpha_val * 10):02d}_w3",
            f"B1 alpha={alpha_val:.1f} window=3",
            alpha=alpha_val,
            convergence_window=3,
        )

    _add("E1_ravg5_w3", "E1 K=5 window=3", running_avg_k=5, convergence_window=3)

    # ------------------------------------------------------------------
    # F2: Larger running-average windows
    # ------------------------------------------------------------------
    for k in [7, 10, 15, 20]:
        _add(
            f"F2_ravg{k}_w3",
            f"F2 K={k} window=3",
            running_avg_k=k,
            convergence_window=3,
        )

    # Also try K=10, 15, 20 with window=1 to compare
    for k in [7, 10, 15, 20]:
        _add(
            f"F2_ravg{k}_w1",
            f"F2 K={k} window=1",
            running_avg_k=k,
            convergence_window=1,
        )

    # ------------------------------------------------------------------
    # F3: Adaptive alpha
    # F3a: alpha = 1 / (1 + delta/scale) — decays as delta grows,
    #       approaches 1 as delta → 0 (full update near convergence)
    # F3b: Polyak step — alpha = min(1, threshold / delta) — clips large steps
    # ------------------------------------------------------------------
    _add(
        "F3a_adaptive",
        "F3a adaptive-alpha (1/(1+d/5))",
        adaptive_alpha="inverse",
        convergence_window=3,
    )
    _add(
        "F3b_polyak",
        "F3b Polyak step (min(1, 0.01/d))",
        adaptive_alpha="polyak",
        convergence_window=3,
    )
    _add("F3b_polyak_w1", "F3b Polyak step window=1", adaptive_alpha="polyak", convergence_window=1)

    # ------------------------------------------------------------------
    # F4: Running-average + alpha combinations
    # ------------------------------------------------------------------
    for k in [5, 7, 10]:
        for a in [0.3, 0.5]:
            _add(
                f"F4_ravg{k}_a{int(a * 10)}_w3",
                f"F4 K={k} alpha={a:.1f} window=3",
                running_avg_k=k,
                alpha=a,
                convergence_window=3,
            )

    # ------------------------------------------------------------------
    # F5: Alpha sweep with window=3
    # ------------------------------------------------------------------
    for alpha_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _add(
            f"F5_a{int(alpha_val * 10):02d}_w3",
            f"F5 alpha={alpha_val:.1f} window=3",
            alpha=alpha_val,
            convergence_window=3,
        )

    # ------------------------------------------------------------------
    # F6: K=5 running-avg + alpha sweep
    # ------------------------------------------------------------------
    for alpha_val in [0.3, 0.5, 0.7]:
        _add(
            f"F6_ravg5_a{int(alpha_val * 10)}_w3",
            f"F6 K=5 alpha={alpha_val:.1f} window=3",
            running_avg_k=5,
            alpha=alpha_val,
            convergence_window=3,
        )

    # ------------------------------------------------------------------
    # F7: Log weights + running-average (revisit with window=3)
    # ------------------------------------------------------------------
    for k in [5, 7, 10]:
        _add(
            f"F7_log_ravg{k}_w3",
            f"F7 log K={k} window=3",
            weight_mode="log",
            running_avg_k=k,
            convergence_window=3,
        )

    return experiments


# ---------------------------------------------------------------------------
# Running a single experiment
# ---------------------------------------------------------------------------


def _make_initial_prices(
    init_strategy: str,
    default_prices: pd.Series,
    strategy_forecasts: dict[str, dict],
) -> pd.Series:
    if init_strategy == "forecast_mean":
        index = default_prices.index
        means = {}
        for t in index:
            vals = [
                float(fcs[t])
                for fcs in strategy_forecasts.values()
                if t in fcs and fcs[t] is not None
            ]
            means[t] = float(np.mean(vals)) if vals else float(default_prices.loc[t])
        return pd.Series(means, name="init_price")
    return default_prices.copy()


def run_experiment(
    cfg: dict,
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    max_iterations: int = 100,
) -> dict:
    """Run one experiment configuration and return metrics dict."""
    from energy_modelling.backtest.futures_market_engine import (
        FuturesMarketEquilibrium,
        FuturesMarketIteration,
        run_futures_market,
        run_futures_market_iteration,
    )

    init_prices = _make_initial_prices(
        cfg["init_strategy"],
        initial_prices,
        strategy_forecasts,
    )

    adaptive_alpha = cfg.get("adaptive_alpha")

    if adaptive_alpha is not None:
        # Custom loop with adaptive alpha
        current_prices = init_prices.copy().astype(float)
        iterations: list[FuturesMarketIteration] = []
        converged = False
        delta = float("inf")
        window = cfg["convergence_window"]
        recent_deltas: deque[float] = deque(maxlen=window)

        for k in range(max_iterations):
            # Compute adaptive alpha from current delta
            if adaptive_alpha == "inverse":
                # alpha = 1 / (1 + delta/5) — gentler near large deltas
                a = 1.0 / (1.0 + delta / 5.0) if delta < float("inf") else 0.2
            elif adaptive_alpha == "polyak":
                # alpha = min(1, threshold / delta) — clips large steps
                a = min(1.0, 0.01 / delta) if delta > 0 else 1.0
            else:
                a = 1.0

            raw = run_futures_market_iteration(
                market_prices=current_prices,
                real_prices=real_prices,
                iteration=k,
                strategy_forecasts=strategy_forecasts,
                weight_mode=cfg["weight_mode"],
                weight_cap=cfg["weight_cap"],
                price_mode=cfg["price_mode"],
            )
            published = a * raw.market_prices + (1.0 - a) * current_prices

            result = FuturesMarketIteration(
                iteration=k,
                market_prices=published,
                strategy_profits=raw.strategy_profits,
                strategy_weights=raw.strategy_weights,
                active_strategies=raw.active_strategies,
            )
            delta = float((published - current_prices).abs().max())
            iterations.append(result)
            current_prices = published

            if delta < 0.01:
                recent_deltas.append(delta)
            else:
                recent_deltas.clear()
            if len(recent_deltas) >= window:
                converged = True
                break

        result_obj = FuturesMarketEquilibrium(
            iterations=iterations,
            final_market_prices=current_prices,
            final_weights=iterations[-1].strategy_weights if iterations else {},
            converged=converged,
            convergence_delta=delta,
        )
    else:
        result_obj = run_futures_market(
            initial_market_prices=init_prices,
            real_prices=real_prices,
            strategy_forecasts=strategy_forecasts,
            max_iterations=max_iterations,
            convergence_threshold=0.01,
            convergence_window=cfg["convergence_window"],
            alpha=cfg["alpha"],
            weight_mode=cfg["weight_mode"],
            weight_cap=cfg["weight_cap"],
            price_mode=cfg["price_mode"],
            running_avg_k=cfg.get("running_avg_k"),
        )

    final_prices = result_obj.final_market_prices
    mae = float((final_prices - real_prices.reindex(final_prices.index)).abs().mean())
    iter0_prices = result_obj.iterations[0].market_prices if result_obj.iterations else init_prices
    mae_iter0 = float((iter0_prices - real_prices.reindex(iter0_prices.index)).abs().mean())

    # Compute per-iteration deltas for diagnostics
    iters = result_obj.iterations
    iter_deltas = [
        round(float((iters[i].market_prices - iters[i - 1].market_prices).abs().max()), 4)
        for i in range(1, len(iters))
    ]

    # Count how many times delta spikes above 1.0 (oscillation proxy)
    n_spikes = sum(1 for d in iter_deltas if d > 1.0)

    return {
        "id": cfg["id"],
        "label": cfg["label"],
        "converged": result_obj.converged,
        "n_iterations": len(result_obj.iterations),
        "final_delta": result_obj.convergence_delta,
        "mae": mae,
        "mae_iter0": mae_iter0,
        "n_spikes": n_spikes,
        "alpha": cfg["alpha"],
        "weight_mode": cfg["weight_mode"],
        "weight_cap": cfg["weight_cap"],
        "price_mode": cfg["price_mode"],
        "init_strategy": cfg["init_strategy"],
        "running_avg_k": cfg.get("running_avg_k"),
        "convergence_window": cfg["convergence_window"],
        "adaptive_alpha": cfg.get("adaptive_alpha"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8b extended experiment sweep.")
    parser.add_argument(
        "--year",
        type=int,
        choices=[2024, 2025],
        default=None,
        help="Run only for one year (default: both).",
    )
    parser.add_argument(
        "--max-iters", type=int, default=100, help="Max iterations per experiment (default: 100)."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    years = [args.year] if args.year else [2024, 2025]
    experiments = _build_experiments()
    logger.info("Running %d experiments across %d year(s)", len(experiments), len(years))

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
            "Year %d: %d strategies, %d eval days", year, len(strategy_forecasts), len(real_prices)
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
                elapsed = time.perf_counter() - t0
                row["elapsed_s"] = round(elapsed, 2)
                all_rows.append(row)

                status = "CONV" if row["converged"] else "----"
                logger.info(
                    "[%d/%d] %s (%d) %s iters=%d delta=%.3f MAE=%.2f spikes=%d",
                    i,
                    len(experiments),
                    cfg["id"],
                    year,
                    status,
                    row["n_iterations"],
                    row["final_delta"],
                    row["mae"],
                    row["n_spikes"],
                )
            except Exception:
                logger.exception("Experiment %s year %d failed", cfg["id"], year)
                all_rows.append(
                    {
                        "id": cfg["id"],
                        "label": cfg["label"],
                        "year": year,
                        "converged": False,
                        "n_iterations": 0,
                        "final_delta": float("inf"),
                        "mae": float("inf"),
                        "mae_iter0": float("inf"),
                        "n_spikes": -1,
                        "error": "FAILED",
                    }
                )

    df = pd.DataFrame(all_rows)
    out_path = PHASE8_DIR / "results_extended.csv"
    df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    _print_summary(df)


def _print_summary(df: pd.DataFrame) -> None:
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year].copy()
        converged = sub[sub["converged"]].sort_values("mae")
        not_conv = sub[~sub["converged"]].sort_values("mae")

        print(f"\n{'=' * 80}")
        print(f"  Year {year} — {len(converged)} converged / {len(sub)} total")
        print(f"{'=' * 80}")

        if converged.empty:
            print("  No experiments converged!")
        else:
            print(f"  {'ID':<30} {'MAE':>7} {'Iters':>6} {'Delta':>8} {'Spikes':>7}  Label")
            print(f"  {'-' * 30} {'-' * 7} {'-' * 6} {'-' * 8} {'-' * 7}  {'-' * 30}")
            for _, r in converged.iterrows():
                print(
                    f"  {r['id']:<30} {r['mae']:>7.2f} {int(r['n_iterations']):>6}"
                    f" {r['final_delta']:>8.4f} {int(r['n_spikes']):>7}  {r['label']}"
                )

        if not not_conv.empty:
            print(f"\n  Top 5 non-converged (by MAE):")
            for _, r in not_conv.head(5).iterrows():
                print(
                    f"  {r['id']:<30} {r['mae']:>7.2f} {int(r['n_iterations']):>6}"
                    f" {r['final_delta']:>8.4f} {int(r['n_spikes']):>7}  {r['label']}"
                )


if __name__ == "__main__":
    main()
