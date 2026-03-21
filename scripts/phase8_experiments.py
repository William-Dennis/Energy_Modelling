"""Phase 8: Sweep all oscillation-remedy configurations.

Loads pre-computed strategy forecasts from data/results/phase8/forecasts_*.pkl
(produced by phase8_collect_forecasts.py) then runs the market engine under
every experiment configuration defined in the Phase 8 research documents.

Results are saved as data/results/phase8/results.csv.

Usage:
    uv run python scripts/phase8_experiments.py
    uv run python scripts/phase8_experiments.py --year 2024   # one year only
    uv run python scripts/phase8_experiments.py --max-iters 50
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


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------


def _build_experiments() -> list[dict]:
    """Return list of experiment config dicts.

    Each dict has at minimum:
        id, label, alpha, weight_mode, weight_cap, price_mode,
        init_strategy, ema_beta, running_avg_k
    """
    experiments: list[dict] = []

    def _add(
        exp_id: str,
        label: str,
        alpha: float = 1.0,
        weight_mode: str = "linear",
        weight_cap: float = 1.0,
        price_mode: str = "mean",
        init_strategy: str = "last_settlement",
        ema_beta: float | None = None,
        running_avg_k: int | None = None,
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
                "ema_beta": ema_beta,
                "running_avg_k": running_avg_k,
            }
        )

    # Baseline (spec-compliant, no fixes)
    _add("BASE", "Baseline (spec, no fix)")

    # ------------------------------------------------------------------
    # Track B: Dampening mechanisms (alpha sweep)
    # ------------------------------------------------------------------
    for alpha_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        _add(
            f"B1_a{int(alpha_val * 10):02d}",
            f"B1 fixed-alpha={alpha_val:.1f}",
            alpha=alpha_val,
        )

    # ------------------------------------------------------------------
    # Track C: Weighting reforms
    # ------------------------------------------------------------------
    # C2: Log-profit weighting
    _add("C2_log", "C2 log-profit weights", weight_mode="log")

    # C3: Capped weights at various cap values
    for cap in [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
        _add(
            f"C3_cap{int(cap * 100):02d}",
            f"C3 weight-cap={cap:.2f}",
            weight_mode="capped",
            weight_cap=cap,
        )

    # ------------------------------------------------------------------
    # Track C: Price mode — weighted median
    # ------------------------------------------------------------------
    _add("C2b_median", "C2b weighted-median price", price_mode="median")

    # ------------------------------------------------------------------
    # Track D: Initialisation strategies
    # ------------------------------------------------------------------
    # D2: forecast mean (handled in run_experiment by passing init_strategy)
    _add("D2_forecast_mean", "D2 init=forecast-mean", init_strategy="forecast_mean")

    # ------------------------------------------------------------------
    # Track E: Iteration smoothing
    # ------------------------------------------------------------------
    # E2: EMA smoothing on market prices across iterations
    for beta in [0.1, 0.2, 0.3, 0.5, 0.7]:
        _add(
            f"E2_ema{int(beta * 10):02d}",
            f"E2 EMA-beta={beta:.1f}",
            ema_beta=beta,
        )

    # E1: Running average over last K iterations
    for k in [2, 3, 5]:
        _add(
            f"E1_ravg{k}",
            f"E1 running-avg K={k}",
            running_avg_k=k,
        )

    # ------------------------------------------------------------------
    # Combinations (X series): most promising joint configs
    # ------------------------------------------------------------------
    # X1: moderate dampening + log weights
    _add("X1_a05_log", "X1 alpha=0.5 + log-weights", alpha=0.5, weight_mode="log")

    # X2: moderate dampening + weight cap
    _add(
        "X2_a05_cap20", "X2 alpha=0.5 + cap=0.20", alpha=0.5, weight_mode="capped", weight_cap=0.20
    )

    # X3: log weights + weighted median
    _add("X3_log_median", "X3 log-weights + median", weight_mode="log", price_mode="median")

    # X4: dampening + log + median
    _add(
        "X4_a03_log_median",
        "X4 alpha=0.3 + log + median",
        alpha=0.3,
        weight_mode="log",
        price_mode="median",
    )

    # X5: dampening + cap + forecast-mean init
    _add(
        "X5_a05_cap20_fcinit",
        "X5 alpha=0.5 + cap=0.20 + fc-init",
        alpha=0.5,
        weight_mode="capped",
        weight_cap=0.20,
        init_strategy="forecast_mean",
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
    """Build initial market prices from the chosen init strategy.

    Parameters
    ----------
    init_strategy:
        ``"last_settlement"`` — use last_settlement_price (spec default).
        ``"forecast_mean"``  — use unweighted mean of all strategy forecasts.
    default_prices:
        last_settlement_price series (always available).
    strategy_forecasts:
        All strategy forecast dicts.
    """
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
    max_iterations: int = 50,
) -> dict:
    """Run one experiment configuration and return metrics dict."""
    from energy_modelling.backtest.futures_market_engine import run_futures_market

    init_prices = _make_initial_prices(
        cfg["init_strategy"],
        initial_prices,
        strategy_forecasts,
    )

    ema_beta = cfg.get("ema_beta")
    running_avg_k = cfg.get("running_avg_k")

    if ema_beta is not None or running_avg_k is not None:
        # Custom iteration loop with smoothing applied across iterations
        result = _run_with_smoothing(
            strategy_forecasts=strategy_forecasts,
            real_prices=real_prices,
            init_prices=init_prices,
            max_iterations=max_iterations,
            alpha=cfg["alpha"],
            weight_mode=cfg["weight_mode"],
            weight_cap=cfg["weight_cap"],
            price_mode=cfg["price_mode"],
            ema_beta=ema_beta,
            running_avg_k=running_avg_k,
        )
    else:
        result = run_futures_market(
            initial_market_prices=init_prices,
            real_prices=real_prices,
            strategy_forecasts=strategy_forecasts,
            max_iterations=max_iterations,
            convergence_threshold=0.01,
            alpha=cfg["alpha"],
            weight_mode=cfg["weight_mode"],
            weight_cap=cfg["weight_cap"],
            price_mode=cfg["price_mode"],
        )

    final_prices = result.final_market_prices
    mae = float((final_prices - real_prices.reindex(final_prices.index)).abs().mean())
    # Compute iter-0 MAE: prices after first iteration vs real
    iter0_prices = result.iterations[0].market_prices if result.iterations else init_prices
    mae_iter0 = float((iter0_prices - real_prices.reindex(iter0_prices.index)).abs().mean())

    return {
        "id": cfg["id"],
        "label": cfg["label"],
        "converged": result.converged,
        "n_iterations": len(result.iterations),
        "final_delta": result.convergence_delta,
        "mae": mae,
        "mae_iter0": mae_iter0,
        "alpha": cfg["alpha"],
        "weight_mode": cfg["weight_mode"],
        "weight_cap": cfg["weight_cap"],
        "price_mode": cfg["price_mode"],
        "init_strategy": cfg["init_strategy"],
        "ema_beta": cfg.get("ema_beta"),
        "running_avg_k": cfg.get("running_avg_k"),
    }


def _run_with_smoothing(
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    init_prices: pd.Series,
    max_iterations: int,
    alpha: float,
    weight_mode: str,
    weight_cap: float,
    price_mode: str,
    ema_beta: float | None,
    running_avg_k: int | None,
) -> object:
    """Custom iteration loop applying EMA or running-average smoothing.

    EMA: after each iteration, the published market price is blended
    with the previous EMA:  ema_t = beta * ema_{t-1} + (1 - beta) * raw_t
    (where raw_t is the dampened candidate for that iteration).

    Running-average: publish the mean of the last K raw iteration prices.
    """
    from energy_modelling.backtest.futures_market_engine import (
        FuturesMarketEquilibrium,
        FuturesMarketIteration,
        run_futures_market_iteration,
    )

    current_prices = init_prices.copy().astype(float)
    iterations: list[FuturesMarketIteration] = []
    converged = False
    delta = float("inf")
    price_history: list[pd.Series] = []
    ema_prices = current_prices.copy()

    for k in range(max_iterations):
        raw_iter = run_futures_market_iteration(
            market_prices=current_prices,
            real_prices=real_prices,
            iteration=k,
            strategy_forecasts=strategy_forecasts,
            weight_mode=weight_mode,
            weight_cap=weight_cap,
            price_mode=price_mode,
        )

        # Apply within-iteration alpha dampening first
        if alpha < 1.0:
            candidate = alpha * raw_iter.market_prices + (1.0 - alpha) * current_prices
        else:
            candidate = raw_iter.market_prices.copy()

        # Apply cross-iteration smoothing
        if ema_beta is not None:
            ema_prices = ema_beta * ema_prices + (1.0 - ema_beta) * candidate
            published = ema_prices.copy()
        elif running_avg_k is not None:
            price_history.append(candidate.copy())
            if len(price_history) > running_avg_k:
                price_history = price_history[-running_avg_k:]
            published = pd.concat(price_history, axis=1).mean(axis=1)
        else:
            published = candidate

        smoothed_iter = FuturesMarketIteration(
            iteration=k,
            market_prices=published,
            strategy_profits=raw_iter.strategy_profits,
            strategy_weights=raw_iter.strategy_weights,
            active_strategies=raw_iter.active_strategies,
        )

        delta = float((published - current_prices).abs().max())
        iterations.append(smoothed_iter)
        current_prices = published

        if delta < 0.01:
            converged = True
            break

    return FuturesMarketEquilibrium(
        iterations=iterations,
        final_market_prices=current_prices,
        final_weights=iterations[-1].strategy_weights if iterations else {},
        converged=converged,
        convergence_delta=delta,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8 experiment sweep.")
    parser.add_argument(
        "--year",
        type=int,
        choices=[2024, 2025],
        default=None,
        help="Run only for one year (default: both).",
    )
    parser.add_argument(
        "--max-iters", type=int, default=50, help="Max iterations per experiment (default: 50)."
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
                    "[%d/%d] %s (%d) %s iters=%d delta=%.2f MAE=%.2f",
                    i,
                    len(experiments),
                    cfg["id"],
                    year,
                    status,
                    row["n_iterations"],
                    row["final_delta"],
                    row["mae"],
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
                        "error": "FAILED",
                    }
                )

    df = pd.DataFrame(all_rows)
    out_path = PHASE8_DIR / "results.csv"
    df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    # Summary: converged runs sorted by MAE
    _print_summary(df)


def _print_summary(df: pd.DataFrame) -> None:
    """Print a ranked summary of experiment results."""
    # Separate by year
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year].copy()
        converged = sub[sub["converged"] == True].sort_values("mae")  # noqa: E712
        not_conv = sub[sub["converged"] == False].sort_values("mae")  # noqa: E712

        print(f"\n{'=' * 72}")
        print(f"  Year {year} — {len(converged)} converged / {len(sub)} total")
        print(f"{'=' * 72}")

        if converged.empty:
            print("  No experiments converged!")
        else:
            print(f"  {'ID':<25} {'MAE':>8} {'Iters':>6} {'Delta':>10}  Label")
            print(f"  {'-' * 25} {'-' * 8} {'-' * 6} {'-' * 10}  {'-' * 30}")
            for _, r in converged.iterrows():
                print(
                    f"  {r['id']:<25} {r['mae']:>8.2f} {int(r['n_iterations']):>6}"
                    f" {r['final_delta']:>10.4f}  {r['label']}"
                )

        if not not_conv.empty:
            print("\n  Top 5 non-converged (by MAE):")
            print(f"  {'ID':<25} {'MAE':>8} {'Iters':>6} {'Delta':>10}  Label")
            for _, r in not_conv.head(5).iterrows():
                print(
                    f"  {r['id']:<25} {r['mae']:>8.2f} {int(r['n_iterations']):>6}"
                    f" {r['final_delta']:>10.4f}  {r['label']}"
                )

    # Overall best converged
    conv_all = df[df["converged"] == True].sort_values("mae")  # noqa: E712
    if not conv_all.empty:
        best = conv_all.iloc[0]
        print(f"\n{'=' * 72}")
        print(f"  BEST OVERALL CONVERGED: {best['id']} (year={int(best['year'])})")
        print(
            f"  MAE={best['mae']:.2f}  delta={best['final_delta']:.4f}"
            f"  iters={int(best['n_iterations'])}"
        )
        print(
            f"  alpha={best['alpha']}  weight_mode={best['weight_mode']}  "
            f"weight_cap={best['weight_cap']}  price_mode={best['price_mode']}"
        )
        print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
