#!/usr/bin/env python
"""Run all 9 challenge strategies through backtest and market evaluation.

Usage::

    python scripts/run_full_backtest.py

Produces two tables:
1. Standard backtest leaderboard (vs last_settlement_price)
2. Market-adjusted leaderboard (vs converged equilibrium price)

Results are printed to stdout and saved to ``data/backtest/backtest_results.csv``
and ``data/backtest/market_results.csv``.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path so ``strategies`` package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from energy_modelling.backtest.futures_market_runner import (  # noqa: E402
    run_futures_market_evaluation,
)
from energy_modelling.backtest.io import (  # noqa: E402
    RESULTS_DIR,
    save_backtest_results,
    save_market_results,
)
from energy_modelling.backtest.runner import run_backtest  # noqa: E402
from energy_modelling.backtest.scoring import leaderboard_score  # noqa: E402
from strategies import (  # noqa: E402
    AlwaysLongStrategy,
    AlwaysShortStrategy,
    CompositeSignalStrategy,
    DayOfWeekStrategy,
    FossilDispatchStrategy,
    Lag2ReversionStrategy,
    LoadForecastStrategy,
    WeeklyCycleStrategy,
    WindForecastStrategy,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_FACTORIES: dict[str, type] = {
    "Always Long": AlwaysLongStrategy,
    "Always Short": AlwaysShortStrategy,
    "Composite Signal": CompositeSignalStrategy,
    "Day Of Week": DayOfWeekStrategy,
    "Fossil Dispatch": FossilDispatchStrategy,
    "Lag2 Reversion": Lag2ReversionStrategy,
    "Load Forecast": LoadForecastStrategy,
    "Weekly Cycle": WeeklyCycleStrategy,
    "Wind Forecast": WindForecastStrategy,
}

PUBLIC_CSV = Path("data/backtest/daily_public.csv")
HIDDEN_CSV = Path("data/backtest/daily_hidden_test_full.csv")
OUTPUT_DIR = Path("data/backtest")

# Evaluation on 2024 validation split
TRAINING_END = date(2023, 12, 31)
EVAL_START = date(2024, 1, 1)
EVAL_END = date(2024, 12, 31)


def _load_combined() -> pd.DataFrame:
    """Load public + hidden test data into a single DataFrame."""
    pub = pd.read_csv(PUBLIC_CSV, parse_dates=["delivery_date"])
    hid = pd.read_csv(HIDDEN_CSV, parse_dates=["delivery_date"])
    combined = pd.concat([pub, hid], ignore_index=True)
    combined["delivery_date"] = combined["delivery_date"].dt.date
    combined = combined.set_index("delivery_date").sort_index()
    return combined


def _load_public() -> pd.DataFrame:
    pub = pd.read_csv(PUBLIC_CSV, parse_dates=["delivery_date"])
    pub["delivery_date"] = pub["delivery_date"].dt.date
    pub = pub.set_index("delivery_date").sort_index()
    return pub


# ---------------------------------------------------------------------------
# Standard backtest
# ---------------------------------------------------------------------------


def run_standard_backtests(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run each strategy through standard backtest, return metrics table and raw results."""
    rows = []
    raw_results: dict = {}
    for name, factory in STRATEGY_FACTORIES.items():
        print(f"  Backtesting: {name} ... ", end="", flush=True)
        t0 = time.perf_counter()
        strategy = factory()
        result = run_backtest(
            strategy=strategy,
            daily_data=daily,
            training_end=TRAINING_END,
            evaluation_start=EVAL_START,
            evaluation_end=EVAL_END,
        )
        raw_results[name] = result
        elapsed = time.perf_counter() - t0
        m = result.metrics
        lb = leaderboard_score(m)
        rows.append(
            {
                "Strategy": name,
                "Total PnL": m["total_pnl"],
                "Sharpe": m["sharpe_ratio"],
                "Max DD": m["max_drawdown"],
                "Win Rate": m["win_rate"],
                "Trades": m["trade_count"],
                "Profit Factor": m["profit_factor"],
                "Best Day": m["best_day"],
                "Worst Day": m["worst_day"],
                "LB Score (PnL)": lb[0],
                "LB Score (Sharpe)": lb[1],
                "Time (s)": round(elapsed, 2),
            }
        )
        print(f"PnL={m['total_pnl']:,.0f}  Sharpe={m['sharpe_ratio']:.2f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df = df.sort_values("Total PnL", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    df.index.name = "Rank"
    return df, raw_results


# ---------------------------------------------------------------------------
# Market evaluation
# ---------------------------------------------------------------------------


def run_market_sim(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict, object]:
    """Run market evaluation with all strategies, return metrics table + info + raw result."""
    print("\n  Running market simulation (may take a few minutes) ...")
    t0 = time.perf_counter()
    market_result = run_futures_market_evaluation(
        strategy_factories=STRATEGY_FACTORIES,
        daily_data=daily,
        training_end=TRAINING_END,
        evaluation_start=EVAL_START,
        evaluation_end=EVAL_END,
        max_iterations=50,
        convergence_threshold=0.01,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Market simulation complete ({elapsed:.1f}s)")

    eq = market_result.equilibrium
    info = {
        "iterations": len(eq.iterations),
        "converged": eq.converged,
        "final_delta": eq.convergence_delta,
    }

    rows = []
    for name in STRATEGY_FACTORIES:
        mkt = market_result.market_results[name]
        orig = market_result.original_results[name]
        m = mkt.metrics
        om = orig.metrics
        rows.append(
            {
                "Strategy": name,
                "Market PnL": m["total_pnl"],
                "Market Sharpe": m["sharpe_ratio"],
                "Market Max DD": m["max_drawdown"],
                "Market Win Rate": m["win_rate"],
                "Original PnL": om["total_pnl"],
                "PnL Delta": m["total_pnl"] - om["total_pnl"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("Market PnL", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df, info, market_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("FULL BACKTEST RE-RUN (with cleaned data pipeline)")
    print("=" * 70)

    print(f"\nLoading data from {PUBLIC_CSV} ...")
    daily = _load_public()
    print(f"  Public rows: {len(daily)}, columns: {daily.shape[1]}")
    print(f"  NaN count: {daily.isna().sum().sum()}")
    print(f"  Date range: {daily.index.min()} to {daily.index.max()}")

    # --- Standard backtest ---
    print("\n--- Standard Backtest (2024 Validation) ---\n")
    std_df, backtest_results = run_standard_backtests(daily)
    print("\n" + "=" * 70)
    print("STANDARD LEADERBOARD")
    print("=" * 70)
    print(std_df.to_string())

    std_path = OUTPUT_DIR / "backtest_results.csv"
    std_df.to_csv(std_path)
    print(f"\nSaved to {std_path}")

    # Persist raw results for dashboard
    save_backtest_results(backtest_results, RESULTS_DIR / "backtest_val_2024.pkl")
    print(f"Saved backtest results to {RESULTS_DIR / 'backtest_val_2024.pkl'}")

    # --- Market evaluation ---
    print("\n--- Market Evaluation (2024 Validation) ---\n")
    # Need combined data for market sim? Let's check if hidden is needed
    # For 2024 validation we only need public data
    mkt_df, mkt_info, market_result = run_market_sim(daily)
    print(f"\n  Iterations: {mkt_info['iterations']}")
    print(f"  Converged: {mkt_info['converged']}")
    print(f"  Final delta: {mkt_info['final_delta']:.4f} EUR/MWh")
    print("\n" + "=" * 70)
    print("MARKET-ADJUSTED LEADERBOARD")
    print("=" * 70)
    print(mkt_df.to_string())

    mkt_path = OUTPUT_DIR / "market_results.csv"
    mkt_df.to_csv(mkt_path)
    print(f"\nSaved to {mkt_path}")

    # Persist raw market result for dashboard
    save_market_results(market_result, RESULTS_DIR / "market_2024.pkl")
    print(f"Saved market results to {RESULTS_DIR / 'market_2024.pkl'}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
