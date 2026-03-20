"""Per-strategy timing profiler for a single benchmark run.

Runs every strategy through a full fit + act loop (366 eval days) and a
forecast collection loop, then prints a sorted table.

Usage
-----
    uv run python scripts/profile_benchmark.py
    uv run python scripts/profile_benchmark.py --csv scripts/profile_results.csv
    uv run python scripts/profile_benchmark.py --benchmark noise_5
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from datetime import date
from pathlib import Path

# Suppress sklearn ConvergenceWarnings so they don't pollute timing output
warnings.filterwarnings("ignore")


def _build_state_list(eval_data, full_data, state_exclude):
    """Pre-build all BacktestState objects once (amortised across strategies)."""
    from energy_modelling.backtest.types import BacktestState

    states = []
    for delivery_date, row in eval_data.iterrows():
        features = row.drop(labels=list(state_exclude), errors="ignore").copy()
        state = BacktestState(
            delivery_date=delivery_date,
            last_settlement_price=float(row["last_settlement_price"]),
            features=features,
            history=full_data.loc[full_data.index < delivery_date].copy(),
        )
        states.append(state)
    return states


def profile(benchmark_id: str = "baseline", csv_path: Path | None = None) -> None:
    from energy_modelling.backtest.feature_engineering import add_derived_features
    from energy_modelling.backtest.runner import _normalise_daily_data
    from energy_modelling.dashboard._backtest import (
        STRATEGY_FACTORIES,
        _resolve_path,
        load_daily,
    )

    _STATE_EXCLUDE = {
        "delivery_date",
        "split",
        "settlement_price",
        "price_change_eur_mwh",
        "target_direction",
        "pnl_long_eur",
        "pnl_short_eur",
    }

    TRAINING_END = date(2023, 12, 31)
    EVAL_START = date(2024, 1, 1)
    EVAL_END = date(2024, 12, 31)

    pub = _resolve_path("data/backtest/daily_public.csv")
    daily = load_daily(pub)

    # Prepare normalised data once
    data = _normalise_daily_data(daily)
    data = add_derived_features(data)

    train_mask = data.index <= TRAINING_END
    eval_mask = (data.index >= EVAL_START) & (data.index <= EVAL_END)
    train_data = data.loc[train_mask].copy()
    eval_data = data.loc[eval_mask].copy()

    print(
        f"Profiling benchmark='{benchmark_id}' | "
        f"train={len(train_data)} rows | eval={len(eval_data)} rows | "
        f"strategies={len(STRATEGY_FACTORIES)}"
    )
    print()

    # Pre-build BacktestState objects (shared across all strategies)
    t0 = time.perf_counter()
    states = _build_state_list(eval_data, data, _STATE_EXCLUDE)
    state_build_ms = (time.perf_counter() - t0) * 1000
    print(f"State build time (amortised): {state_build_ms:.0f}ms for {len(states)} days")
    print()

    rows = []
    for name, factory in STRATEGY_FACTORIES.items():
        # --- setup ---
        t_setup = time.perf_counter()
        strategy = factory()
        setup_ms = (time.perf_counter() - t_setup) * 1000

        # --- fit ---
        t_fit = time.perf_counter()
        strategy.fit(train_data)
        strategy.reset()
        fit_ms = (time.perf_counter() - t_fit) * 1000

        # --- act loop (eval) ---
        t_act = time.perf_counter()
        for delivery_date, row in eval_data.iterrows():
            from energy_modelling.backtest.types import BacktestState

            state = BacktestState(
                delivery_date=delivery_date,
                last_settlement_price=float(row["last_settlement_price"]),
                features=row.drop(labels=list(_STATE_EXCLUDE), errors="ignore").copy(),
                history=data.loc[data.index < delivery_date].copy(),
            )
            strategy.act(state)
        act_ms = (time.perf_counter() - t_act) * 1000

        # --- forecast loop (Phase 2b proxy) ---
        t_fc = time.perf_counter()
        for delivery_date, row in eval_data.iterrows():
            from energy_modelling.backtest.types import BacktestState

            state = BacktestState(
                delivery_date=delivery_date,
                last_settlement_price=float(row["last_settlement_price"]),
                features=row.drop(labels=list(_STATE_EXCLUDE), errors="ignore").copy(),
                history=data.loc[data.index < delivery_date].copy(),
            )
            strategy.forecast(state)
        fc_ms = (time.perf_counter() - t_fc) * 1000

        total_ms = setup_ms + fit_ms + act_ms + fc_ms
        rows.append(
            {
                "strategy": name,
                "setup_ms": setup_ms,
                "fit_ms": fit_ms,
                "act_ms": act_ms,
                "forecast_ms": fc_ms,
                "total_ms": total_ms,
            }
        )

    rows.sort(key=lambda r: r["total_ms"], reverse=True)

    # Print table
    header = f"{'Strategy':<36} {'setup':>7} {'fit':>9} {'act×366':>9} {'fc×366':>9} {'TOTAL':>9}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['strategy']:<36} "
            f"{r['setup_ms']:>6.0f}ms "
            f"{r['fit_ms']:>8.0f}ms "
            f"{r['act_ms']:>8.0f}ms "
            f"{r['forecast_ms']:>8.0f}ms "
            f"{r['total_ms']:>8.0f}ms"
        )
    print(sep)
    grand = sum(r["total_ms"] for r in rows)
    fit_total = sum(r["fit_ms"] for r in rows)
    act_total = sum(r["act_ms"] for r in rows)
    fc_total = sum(r["forecast_ms"] for r in rows)
    print(
        f"{'TOTAL (serial)':<36} "
        f"{'':>7} "
        f"{fit_total:>8.0f}ms "
        f"{act_total:>8.0f}ms "
        f"{fc_total:>8.0f}ms "
        f"{grand:>8.0f}ms"
    )

    if csv_path:
        csv_path = Path(csv_path)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile per-strategy timing")
    parser.add_argument("--benchmark", default="baseline", help="Benchmark ID to use")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    args = parser.parse_args()
    profile(benchmark_id=args.benchmark, csv_path=args.csv)


if __name__ == "__main__":
    main()
