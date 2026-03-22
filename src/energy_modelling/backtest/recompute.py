"""Regenerate all backtest and futures market results.

CLI entry point: ``recompute-all``

Caching strategy
----------------
**Two-tier caching** ensures maximum speed:

1. **Pickle fingerprint cache** (legacy, retained): Backtests are deterministic.
   We compute a SHA-256 fingerprint of all strategy source files plus the CSV
   dataset; if a pickle was written with the same fingerprint we skip it.

2. **SQLite forecast cache** (new, Phase 12): Per-strategy forecasts and
   backtest results are stored in ``data/results/forecast_cache.db``.
   Each strategy is fingerprinted individually so that changing one strategy
   only invalidates that strategy's cache.  The market simulation loads
   forecasts directly from the database, bypassing the expensive
   ``fit()`` + ``forecast()`` cycle for all cached strategies.

   This brings ``recompute-all`` from minutes to seconds when the cache is warm.

Market simulation parameters
-----------------------------
- ``ema_alpha=0.003``: production default (Phase 14 fine sweep).  Lowest value
  achieving genuine cold-start convergence for both 2024 (1195 iters, ~0.4 s)
  and 2025 (1103 iters, ~0.3 s).  Previously 0.01 (Phase 13) which appeared to
  converge only due to warm-start initialisation; now confirmed with cold starts.
  Passed explicitly here for clarity; also the engine function default.
- ``max_iterations=10_000``: engine default (Phase 14).  Ample headroom; early
  exit on convergence keeps cost low.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from energy_modelling.backtest.benchmarks import ALL_BENCHMARKS, get_benchmark
from energy_modelling.backtest.forecast_cache import (
    _connect,
    clear_cache,
    is_cached,
    load_all_backtest_results,
    load_all_forecasts,
    store_forecasts,
)
from energy_modelling.backtest.io import (
    RESULTS_DIR,
    load_backtest_results,
    save_backtest_results,
    save_market_results,
)
from energy_modelling.backtest.runner import BacktestResult, run_backtest
from energy_modelling.backtest.types import BacktestStrategy

logger = logging.getLogger(__name__)

# Default date ranges
TRAINING_END = date(2023, 12, 31)
EVAL_START = date(2024, 1, 1)
EVAL_END = date(2024, 12, 31)

# Reduced set of benchmarks run by default (covers representative spread without
# running all 8 which would take 8x longer for diminishing insight).
DEFAULT_BENCHMARKS = ["baseline", "noise_5", "oracle"]

_FINGERPRINT_FILE = RESULTS_DIR / ".fingerprint.json"


# ---------------------------------------------------------------------------
# Worker (top-level so it is picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _run_backtest_worker(
    name: str,
    factory: Callable[[], BacktestStrategy],
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
    entry_prices: pd.Series | None,
) -> tuple[str, BacktestResult]:
    """Subprocess worker: instantiate, fit and evaluate one strategy.

    Returns ``(name, BacktestResult)``.
    """
    strategy = factory()
    result = run_backtest(
        strategy=strategy,
        daily_data=daily_data,
        training_end=training_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        entry_prices=entry_prices,
    )
    return name, result


def _run_and_cache_worker(
    name: str,
    factory: Callable[[], BacktestStrategy],
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
) -> tuple[str, BacktestResult, dict]:
    """Subprocess worker: fit, evaluate, AND collect forecasts.

    Returns ``(name, BacktestResult, forecasts_dict)``.
    """
    from energy_modelling.backtest.feature_engineering import (  # noqa: PLC0415
        add_derived_features,
    )
    from energy_modelling.backtest.futures_market_runner import (  # noqa: PLC0415
        _collect_forecasts,
    )
    from energy_modelling.backtest.runner import _normalise_daily_data  # noqa: PLC0415

    strategy = factory()
    result = run_backtest(
        strategy=strategy,
        daily_data=daily_data,
        training_end=training_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )

    # Collect forecasts from the already-fitted strategy
    data = _normalise_daily_data(daily_data)
    data = add_derived_features(data)
    eval_mask = (data.index >= evaluation_start) & (data.index <= evaluation_end)
    eval_data = data.loc[eval_mask]

    forecasts = _collect_forecasts(strategy, eval_data, data)
    return name, result, forecasts


# ---------------------------------------------------------------------------
# Fingerprinting helpers (legacy pickle cache)
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    """Return hex SHA-256 of a single file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _strategies_fingerprint(strategies_dir: Path) -> str:
    """SHA-256 over the concatenated content of every ``*.py`` in *strategies/*."""
    h = hashlib.sha256()
    for py in sorted(strategies_dir.glob("*.py")):
        h.update(py.read_bytes())
    return h.hexdigest()


def _data_fingerprint(csv_path: Path) -> str:
    """SHA-256 of the CSV dataset file."""
    if csv_path.exists():
        return _hash_file(csv_path)
    return "MISSING"


def _compute_fingerprint(strategies_dir: Path, csv_path: Path) -> str:
    """Combined fingerprint of strategy code + dataset."""
    return _strategies_fingerprint(strategies_dir) + "|" + _data_fingerprint(csv_path)


def _load_saved_fingerprints() -> dict:
    """Load the fingerprint cache from disk."""
    if _FINGERPRINT_FILE.exists():
        try:
            return json.loads(_FINGERPRINT_FILE.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_fingerprints(fp: dict) -> None:
    """Persist the fingerprint cache to disk."""
    _FINGERPRINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FINGERPRINT_FILE.write_text(json.dumps(fp, indent=2))


def _should_skip(pkl_name: str, fingerprint: str, *, force: bool) -> bool:
    """Return True if the pkl is up-to-date and can be skipped."""
    if force:
        return False
    pkl_path = RESULTS_DIR / pkl_name
    if not pkl_path.exists():
        return False
    saved = _load_saved_fingerprints()
    return saved.get(pkl_name) == fingerprint


def _record_fingerprint(pkl_name: str, fingerprint: str) -> None:
    """Record a successful build in the fingerprint cache."""
    saved = _load_saved_fingerprints()
    saved[pkl_name] = fingerprint
    _save_fingerprints(saved)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _parse_date(s: str | None) -> date | None:
    """Parse an ISO-format date string, or return *None*."""
    if s is None:
        return None
    return date.fromisoformat(s)


def _populate_forecast_cache(
    strat_factories: dict[str, Callable[[], BacktestStrategy]],
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
    year: int,
    strategies_dir: Path,
    csv_path: Path,
    *,
    force: bool = False,
    max_workers: int | None = None,
) -> tuple[dict[str, dict], dict[str, BacktestResult]]:
    """Ensure the forecast cache is populated for all strategies.

    Returns ``(all_forecasts, all_backtest_results)`` loaded from the cache.
    Only strategies whose source code or data has changed are re-computed.
    """
    conn = _connect()
    try:
        # Identify which strategies need recomputing
        uncached_names: list[str] = []
        for name in strat_factories:
            if force or not is_cached(name, year, strategies_dir, csv_path, conn):
                uncached_names.append(name)

        if uncached_names:
            logger.info(
                "Year %d: %d/%d strategies need recomputing: %s",
                year,
                len(uncached_names),
                len(strat_factories),
                uncached_names[:10],
            )
            uncached_factories = {n: strat_factories[n] for n in uncached_names}

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures_map = {
                    executor.submit(
                        _run_and_cache_worker,
                        name,
                        factory,
                        daily_data,
                        training_end,
                        evaluation_start,
                        evaluation_end,
                    ): name
                    for name, factory in uncached_factories.items()
                }
                for future in tqdm(
                    futures_map,
                    desc=f"  {year} forecast cache",
                    unit="strat",
                ):
                    name, result, forecasts = future.result()
                    store_forecasts(
                        name,
                        year,
                        forecasts,
                        result,
                        strategies_dir,
                        csv_path,
                        conn,
                    )
                    logger.debug("Cached %s (%d forecasts)", name, len(forecasts))
        else:
            logger.info("Year %d: all %d strategies cached", year, len(strat_factories))

        # Load everything from the cache
        all_forecasts = load_all_forecasts(year, conn)
        all_results = load_all_backtest_results(year, conn)
        return all_forecasts, all_results
    finally:
        conn.close()


def recompute_all(
    strategies: list[str] | None = None,
    benchmarks: list[str] | None = None,
    training_end: date | None = None,
    evaluation_start: date | None = None,
    evaluation_end: date | None = None,
    *,
    force: bool = False,
    max_workers: int | None = None,
) -> None:
    """Regenerate all backtest results.

    Parameters
    ----------
    strategies:
        Strategy display names to include (default: all discovered strategies).
    benchmarks:
        Benchmark IDs to run (default: ``DEFAULT_BENCHMARKS``).
    training_end, evaluation_start, evaluation_end:
        Override the default date window.
    force:
        If True, regenerate everything regardless of cache state.
    max_workers:
        Number of worker processes. ``None`` lets :class:`ProcessPoolExecutor`
        choose (one per CPU). Set to 1 for serial execution.
    """
    from energy_modelling.backtest.futures_market_runner import (
        run_futures_market_evaluation,
    )
    from energy_modelling.dashboard._backtest import (
        STRATEGY_FACTORIES,
        _resolve_path,
        combine_public_hidden,
        load_daily,
    )

    t_end = training_end or TRAINING_END
    e_start = evaluation_start or EVAL_START
    e_end = evaluation_end or EVAL_END

    pub_path = _resolve_path("data/backtest/daily_public.csv")
    if not pub_path.exists():
        logger.error("Dataset not found: %s", pub_path)
        return

    # Resolve strategies directory for fingerprinting
    import strategies as _strat_pkg

    strategies_dir = Path(_strat_pkg.__file__).parent
    fp_val = _compute_fingerprint(strategies_dir, pub_path)

    daily = load_daily(pub_path)
    logger.info("Loaded %d rows from %s", len(daily), pub_path)

    strat_factories = STRATEGY_FACTORIES
    if strategies:
        strat_factories = {k: v for k, v in strat_factories.items() if k in strategies}
    logger.info("Strategies (%d): %s", len(strat_factories), list(strat_factories.keys()))

    bench_ids = DEFAULT_BENCHMARKS if benchmarks is None else benchmarks
    logger.info("Benchmarks: %s", bench_ids)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: Benchmark backtests (deterministic -- respect pickle cache)
    # ------------------------------------------------------------------
    for bench_id in tqdm(bench_ids, desc="Benchmarks", unit="bench"):
        pkl_name = f"benchmark_{bench_id}.pkl"
        if _should_skip(pkl_name, fp_val, force=force):
            logger.info("SKIP %s (unchanged)", pkl_name)
            continue

        entry_prices = get_benchmark(bench_id, daily)
        ep = entry_prices if bench_id != "baseline" else None

        results: dict[str, BacktestResult] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {
                executor.submit(
                    _run_backtest_worker,
                    name,
                    factory,
                    daily,
                    t_end,
                    e_start,
                    e_end,
                    ep,
                ): name
                for name, factory in tqdm(
                    strat_factories.items(),
                    desc=f"  {bench_id} (submitting)",
                    unit="strat",
                    leave=False,
                )
            }
            for future in tqdm(
                futures_map,
                desc=f"  {bench_id} (collecting)",
                unit="strat",
                leave=False,
            ):
                name, result = future.result()
                results[name] = result

        save_backtest_results(results, RESULTS_DIR / pkl_name)
        _record_fingerprint(pkl_name, fp_val)
        logger.info("Saved %s", pkl_name)

    # Copy baseline -> backtest_val_2024.pkl
    if "baseline" in bench_ids:
        baseline_path = RESULTS_DIR / "benchmark_baseline.pkl"
        baseline = load_backtest_results(baseline_path)
        if baseline:
            val_name = "backtest_val_2024.pkl"
            save_backtest_results(baseline, RESULTS_DIR / val_name)
            _record_fingerprint(val_name, fp_val)
            logger.info("Saved %s (copy of baseline)", val_name)

    # ------------------------------------------------------------------
    # Phase 2: Hidden-test period (2025) -- deterministic
    # ------------------------------------------------------------------
    hid_path = _resolve_path("data/backtest/daily_hidden_test_full.csv")
    combined = None
    if hid_path.exists():
        fp_hid = _compute_fingerprint(strategies_dir, hid_path) + "|" + fp_val
        hid_pkl = "backtest_hid_2025.pkl"
        if _should_skip(hid_pkl, fp_hid, force=force):
            logger.info("SKIP %s (unchanged)", hid_pkl)
            # Still need combined for market 2025
            hidden = load_daily(hid_path)
            combined = combine_public_hidden(daily, hidden)
        else:
            hidden = load_daily(hid_path)
            combined = combine_public_hidden(daily, hidden)
            logger.info("Hidden data (%d rows). Running 2025 backtests...", len(hidden))
            hid_results: dict[str, BacktestResult] = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures_map = {
                    executor.submit(
                        _run_backtest_worker,
                        name,
                        factory,
                        combined,
                        date(2024, 12, 31),
                        date(2025, 1, 1),
                        date(2025, 12, 31),
                        None,
                    ): name
                    for name, factory in tqdm(
                        strat_factories.items(),
                        desc="2025 backtest (submitting)",
                        unit="strat",
                    )
                }
                for future in tqdm(
                    futures_map,
                    desc="2025 backtest (collecting)",
                    unit="strat",
                ):
                    name, result = future.result()
                    hid_results[name] = result
            save_backtest_results(hid_results, RESULTS_DIR / hid_pkl)
            _record_fingerprint(hid_pkl, fp_hid)
            logger.info("Saved %s", hid_pkl)
    else:
        logger.info("No hidden data at %s -- skipping 2025 evaluation.", hid_path)

    # ------------------------------------------------------------------
    # Phase 3: Populate forecast cache (per-strategy, incremental)
    # This is the expensive step on first run, but free on subsequent runs.
    # ------------------------------------------------------------------
    logger.info("Populating forecast cache (2024)...")
    forecasts_24, results_24 = _populate_forecast_cache(
        strat_factories,
        daily,
        t_end,
        e_start,
        e_end,
        year=2024,
        strategies_dir=strategies_dir,
        csv_path=pub_path,
        force=force,
        max_workers=max_workers,
    )

    forecasts_25: dict[str, dict] = {}
    results_25: dict[str, BacktestResult] = {}
    if combined is not None:
        logger.info("Populating forecast cache (2025)...")
        # Determine CSV path for 2025 fingerprinting
        forecasts_25, results_25 = _populate_forecast_cache(
            strat_factories,
            combined,
            date(2024, 12, 31),
            date(2025, 1, 1),
            date(2025, 12, 31),
            year=2025,
            strategies_dir=strategies_dir,
            csv_path=hid_path,
            force=force,
            max_workers=max_workers,
        )

    # ------------------------------------------------------------------
    # Phase 4: Futures market simulation (fast -- uses cached forecasts)
    # ------------------------------------------------------------------
    logger.info("Running futures market simulation (2024) with cached forecasts...")
    try:
        m24 = run_futures_market_evaluation(
            strategy_factories=strat_factories,
            daily_data=daily,
            training_end=t_end,
            evaluation_start=e_start,
            evaluation_end=e_end,
            max_workers=max_workers,
            cached_forecasts=forecasts_24,
            cached_results=results_24,
            ema_alpha=0.003,
        )
        save_market_results(m24, RESULTS_DIR / "market_2024.pkl")
        logger.info("Saved market_2024.pkl")
    except Exception:
        logger.exception("Futures market 2024 failed -- skipping.")

    if combined is not None:
        logger.info("Running futures market simulation (2025) with cached forecasts...")
        try:
            m25 = run_futures_market_evaluation(
                strategy_factories=strat_factories,
                daily_data=combined,
                training_end=date(2024, 12, 31),
                evaluation_start=date(2025, 1, 1),
                evaluation_end=date(2025, 12, 31),
                max_workers=max_workers,
                cached_forecasts=forecasts_25,
                cached_results=results_25,
                ema_alpha=0.003,
            )
            save_market_results(m25, RESULTS_DIR / "market_2025.pkl")
            logger.info("Saved market_2025.pkl")
        except Exception:
            logger.exception("Futures market 2025 failed -- skipping.")

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.1f seconds. Results saved to %s", elapsed, RESULTS_DIR)


def main() -> None:
    """CLI entry point for ``recompute-all``."""
    parser = argparse.ArgumentParser(
        description="Regenerate all backtest and futures market results.",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Strategy names (default: all)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help=f"Benchmark IDs (default: {DEFAULT_BENCHMARKS})",
    )
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Run all 8 benchmarks instead of the default reduced set",
    )
    parser.add_argument(
        "--training-end",
        type=str,
        default=None,
        help="Training end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--evaluation-start",
        type=str,
        default=None,
        help="Eval start (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--evaluation-end",
        type=str,
        default=None,
        help="Eval end (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration, ignoring cache",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: number of CPUs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the SQLite forecast cache before running",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.clear_cache:
        logger.info("Clearing forecast cache...")
        clear_cache()

    bench_ids = None
    if args.all_benchmarks:
        bench_ids = list(ALL_BENCHMARKS.keys())
    elif args.benchmarks is not None:
        bench_ids = args.benchmarks

    recompute_all(
        strategies=args.strategies,
        benchmarks=bench_ids,
        training_end=_parse_date(args.training_end),
        evaluation_start=_parse_date(args.evaluation_start),
        evaluation_end=_parse_date(args.evaluation_end),
        force=args.force,
        max_workers=args.max_workers,
    )
