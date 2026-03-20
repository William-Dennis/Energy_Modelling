"""Regenerate all backtest and futures market results.

CLI entry point: ``recompute-all``

Caching strategy
----------------
Backtests are deterministic — given the same strategy source code and the same
dataset, they produce identical results.  We compute a SHA-256 fingerprint of
all strategy source files plus the CSV dataset; if a pickle file was written
with the same fingerprint we skip it.

The futures market simulation is dynamic (it depends on the mix of all
strategies and the market convergence engine), so ``market_*.pkl`` files are
always regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path

from tqdm import tqdm

from energy_modelling.backtest.benchmarks import ALL_BENCHMARKS, get_benchmark
from energy_modelling.backtest.io import (
    RESULTS_DIR,
    load_backtest_results,
    save_backtest_results,
    save_market_results,
)
from energy_modelling.backtest.runner import run_backtest

logger = logging.getLogger(__name__)

# Default date ranges
TRAINING_END = date(2023, 12, 31)
EVAL_START = date(2024, 1, 1)
EVAL_END = date(2024, 12, 31)

_FINGERPRINT_FILE = RESULTS_DIR / ".fingerprint.json"


# ---------------------------------------------------------------------------
# Fingerprinting helpers
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


def recompute_all(
    strategies: list[str] | None = None,
    benchmarks: list[str] | None = None,
    training_end: date | None = None,
    evaluation_start: date | None = None,
    evaluation_end: date | None = None,
    *,
    force: bool = False,
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
    force:
        If True, regenerate everything regardless of cache state.
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

    bench_ids = list(ALL_BENCHMARKS.keys()) if benchmarks is None else benchmarks
    logger.info("Benchmarks: %s", bench_ids)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Phase 1: Benchmark backtests (deterministic — respect cache)
    # ------------------------------------------------------------------
    for bench_id in tqdm(bench_ids, desc="Benchmarks", unit="bench"):
        pkl_name = f"benchmark_{bench_id}.pkl"
        if _should_skip(pkl_name, fp_val, force=force):
            logger.info("SKIP %s (unchanged)", pkl_name)
            continue

        entry_prices = get_benchmark(bench_id, daily)
        results = {}
        for name, factory in tqdm(
            strat_factories.items(),
            desc=f"  {bench_id}",
            unit="strat",
            leave=False,
        ):
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
        save_backtest_results(results, RESULTS_DIR / pkl_name)
        _record_fingerprint(pkl_name, fp_val)
        logger.info("Saved %s", pkl_name)

    # Copy baseline → backtest_val_2024.pkl
    if "baseline" in bench_ids:
        baseline_path = RESULTS_DIR / "benchmark_baseline.pkl"
        baseline = load_backtest_results(baseline_path)
        if baseline:
            val_name = "backtest_val_2024.pkl"
            save_backtest_results(baseline, RESULTS_DIR / val_name)
            _record_fingerprint(val_name, fp_val)
            logger.info("Saved %s (copy of baseline)", val_name)

    # ------------------------------------------------------------------
    # Phase 2: Hidden-test period (2025) — deterministic
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
            hid_results: dict = {}
            for name, factory in tqdm(strat_factories.items(), desc="2025 backtest", unit="strat"):
                strategy = factory()
                result = run_backtest(
                    strategy=strategy,
                    daily_data=combined,
                    training_end=date(2024, 12, 31),
                    evaluation_start=date(2025, 1, 1),
                    evaluation_end=date(2025, 12, 31),
                )
                hid_results[name] = result
            save_backtest_results(hid_results, RESULTS_DIR / hid_pkl)
            _record_fingerprint(hid_pkl, fp_hid)
            logger.info("Saved %s", hid_pkl)
    else:
        logger.info("No hidden data at %s — skipping 2025 evaluation.", hid_path)

    # ------------------------------------------------------------------
    # Phase 3: Futures market simulation (always re-run — dynamic)
    # ------------------------------------------------------------------
    logger.info("Running futures market simulation (2024)...")
    try:
        m24 = run_futures_market_evaluation(
            strategy_factories=strat_factories,
            daily_data=daily,
            training_end=t_end,
            evaluation_start=e_start,
            evaluation_end=e_end,
        )
        save_market_results(m24, RESULTS_DIR / "market_2024.pkl")
        logger.info("Saved market_2024.pkl")
    except Exception:
        logger.exception("Futures market 2024 failed — skipping.")

    if combined is not None:
        logger.info("Running futures market simulation (2025)...")
        try:
            m25 = run_futures_market_evaluation(
                strategy_factories=strat_factories,
                daily_data=combined,
                training_end=date(2024, 12, 31),
                evaluation_start=date(2025, 1, 1),
                evaluation_end=date(2025, 12, 31),
            )
            save_market_results(m25, RESULTS_DIR / "market_2025.pkl")
            logger.info("Saved market_2025.pkl")
        except Exception:
            logger.exception("Futures market 2025 failed — skipping.")

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
        help="Benchmark IDs (default: all)",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
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
        force=args.force,
    )
