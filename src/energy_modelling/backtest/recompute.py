"""Regenerate all backtest and futures market results.

CLI entry point: ``recompute-all``
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

from energy_modelling.backtest.benchmarks import ALL_BENCHMARKS, get_benchmark
from energy_modelling.backtest.io import (
    RESULTS_DIR,
    load_backtest_results,
    save_backtest_results,
)
from energy_modelling.backtest.runner import run_backtest

logger = logging.getLogger(__name__)

# Default date ranges
TRAINING_END = date(2023, 12, 31)
EVAL_START = date(2024, 1, 1)
EVAL_END = date(2024, 12, 31)


def _parse_date(s: str | None) -> date | None:
    """Parse an ISO-format date string, or return *None*."""
    if s is None:
        return None
    return date.fromisoformat(s)


def recompute_all(
    strategies: list[str] | None = None,
    benchmarks: list[str] | None = None,
    training_end: date | None = None,
    evaluation_start: date | None = None,
    evaluation_end: date | None = None,
) -> None:
    """Regenerate all backtest results.

    Parameters
    ----------
    strategies:
        Strategy display names to include (default: all discovered strategies).
    benchmarks:
        Benchmark IDs to run (default: all from ``ALL_BENCHMARKS``).
    training_end, evaluation_start, evaluation_end:
        Override the default date window.
    """
    from energy_modelling.dashboard._backtest import (
        STRATEGY_FACTORIES,
        _resolve_path,
        load_daily,
    )

    t_end = training_end or TRAINING_END
    e_start = evaluation_start or EVAL_START
    e_end = evaluation_end or EVAL_END

    pub_path = _resolve_path("data/backtest/daily_public.csv")
    if not pub_path.exists():
        logger.error("Dataset not found: %s", pub_path)
        return

    daily = load_daily(pub_path)
    logger.info("Loaded %d rows from %s", len(daily), pub_path)

    strat_factories = STRATEGY_FACTORIES
    if strategies:
        strat_factories = {k: v for k, v in strat_factories.items() if k in strategies}
    logger.info("Strategies: %s", list(strat_factories.keys()))

    bench_ids = list(ALL_BENCHMARKS.keys()) if benchmarks is None else benchmarks
    logger.info("Benchmarks: %s", bench_ids)

    t0 = time.perf_counter()
    for bench_id in bench_ids:
        entry_prices = get_benchmark(bench_id, daily)
        results = {}
        for name, factory in strat_factories.items():
            strategy = factory()
            result = run_backtest(
                strategy=strategy,
                daily_data=daily,
                training_end=t_end,
                evaluation_start=e_start,
                evaluation_end=e_end,
                entry_prices=entry_prices if bench_id != "baseline" else None,
            )
            results[name] = result
            logger.info("  %s × %s: PnL=%.0f", name, bench_id, result.metrics["total_pnl"])
        save_backtest_results(results, RESULTS_DIR / f"benchmark_{bench_id}.pkl")
        logger.info("Saved benchmark_%s.pkl", bench_id)

    if "baseline" in bench_ids:
        baseline_path = RESULTS_DIR / "benchmark_baseline.pkl"
        baseline = load_backtest_results(baseline_path)
        if baseline:
            save_backtest_results(baseline, RESULTS_DIR / "backtest_val_2024.pkl")

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1f seconds. Results saved to %s", elapsed, RESULTS_DIR)


def main() -> None:
    """CLI entry point for ``recompute-all``."""
    parser = argparse.ArgumentParser(
        description="Regenerate all backtest and futures market results.",
    )
    parser.add_argument(
        "--strategies", nargs="*", default=None, help="Strategy names (default: all)",
    )
    parser.add_argument(
        "--benchmarks", nargs="*", default=None, help="Benchmark IDs (default: all)",
    )
    parser.add_argument(
        "--training-end", type=str, default=None, help="Training end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--evaluation-start", type=str, default=None, help="Eval start (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--evaluation-end", type=str, default=None, help="Eval end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    recompute_all(
        strategies=args.strategies,
        benchmarks=args.benchmarks,
        training_end=_parse_date(args.training_end),
        evaluation_start=_parse_date(args.evaluation_start),
        evaluation_end=_parse_date(args.evaluation_end),
    )
