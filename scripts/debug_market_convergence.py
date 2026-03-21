"""Diagnostic: trace market convergence with the spec-compliant engine.

Runs the market engine with various strategy combinations and prints
per-iteration MAE, weights, and price update internals.

Usage:
    uv run python scripts/debug_market_convergence.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Ensure repo root and src are on the path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd  # noqa: E402

from energy_modelling.backtest.feature_engineering import add_derived_features  # noqa: E402
from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    compute_market_prices,
    compute_strategy_profits,
    compute_weights,
    run_futures_market,
)
from energy_modelling.backtest.runner import _normalise_daily_data  # noqa: E402
from energy_modelling.dashboard._backtest import _resolve_path, load_daily  # noqa: E402


def _build_pf_forecasts(real_prices: pd.Series) -> dict:
    """PF forecast dict: every date maps to the real settlement price."""
    return {t: float(real_prices.loc[t]) for t in real_prices.index}


def _build_long_forecasts(initial_prices: pd.Series, bias: float = 10.0) -> dict:
    """Always-Long forecast: initial price + positive bias."""
    return {t: float(initial_prices.loc[t]) + bias for t in initial_prices.index}


def _build_short_forecasts(initial_prices: pd.Series, bias: float = 10.0) -> dict:
    """Always-Short forecast: initial price - negative bias."""
    return {t: float(initial_prices.loc[t]) - bias for t in initial_prices.index}


def run_experiment(
    label: str,
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    max_iterations: int = 20,
    convergence_threshold: float = 0.001,
) -> None:
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {label}")
    print(f"{'=' * 70}")
    print(f"  Initial MAE vs real: {(initial_prices - real_prices).abs().mean():.4f} EUR/MWh")
    print()

    eq = run_futures_market(
        initial_market_prices=initial_prices.copy(),
        real_prices=real_prices,
        strategy_forecasts=strategy_forecasts,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
    )

    print(
        f"  Converged: {eq.converged}  |  Iterations: {len(eq.iterations)}"
        f"  |  Delta: {eq.convergence_delta:.6f}"
    )
    print()
    print(f"  {'Iter':>4}  {'MAE_real':>10}  {'Delta':>10}  {'Active strategies & weights'}")
    print(f"  {'-' * 4}  {'-' * 10}  {'-' * 10}  {'-' * 40}")

    prev = initial_prices.copy().astype(float)
    for it in eq.iterations:
        mae_real = float((it.market_prices - real_prices).abs().mean())
        delta = float((it.market_prices - prev).abs().max())
        active = {k: f"{v:.3f}" for k, v in it.strategy_weights.items() if v > 0}
        print(f"  {it.iteration:>4}  {mae_real:>10.4f}  {delta:>10.6f}  {active}")
        prev = it.market_prices

    final = eq.final_market_prices
    print()
    print(f"  FINAL MAE vs real: {(final - real_prices).abs().mean():.4f} EUR/MWh")

    # --- Deep-dive: show PF profit at iteration 0 ---
    if "Perfect Foresight" in strategy_forecasts:
        print()
        print("  --- PF profit trace (iter 0) ---")
        profits0 = compute_strategy_profits(initial_prices, real_prices, strategy_forecasts)
        weights0 = compute_weights(profits0)
        print(f"  Profits: { {k: f'{v:.1f}' for k, v in profits0.items()} }")
        print(f"  Weights: { {k: f'{v:.4f}' for k, v in weights0.items()} }")

        raw_new = compute_market_prices(weights0, strategy_forecasts, initial_prices)
        print(f"  Raw new price (first 5 days): {raw_new.head().values}")
        print(f"  Real prices   (first 5 days): {real_prices.head().values}")


def main():
    pub = _resolve_path("data/backtest/daily_public.csv")
    daily = load_daily(pub)
    data = _normalise_daily_data(daily)
    data = add_derived_features(data)

    EVAL_START = date(2024, 1, 1)
    EVAL_END = date(2024, 12, 31)

    eval_mask = (data.index >= EVAL_START) & (data.index <= EVAL_END)
    eval_data = data.loc[eval_mask]

    real_prices = eval_data["settlement_price"].astype(float)
    initial_prices = eval_data["last_settlement_price"].astype(float)

    print(f"Eval window: {EVAL_START} to {EVAL_END}  ({len(eval_data)} days)")
    print(f"Real price range:    {real_prices.min():.1f} - {real_prices.max():.1f} EUR/MWh")
    print(f"Prev-day price range:{initial_prices.min():.1f} - {initial_prices.max():.1f} EUR/MWh")
    print(
        f"Price std: {real_prices.std():.2f}  |  "
        f"MAE(real, prev-day): {(real_prices - initial_prices).abs().mean():.2f}"
    )

    pf_forecasts = _build_pf_forecasts(real_prices)
    long_forecasts = _build_long_forecasts(initial_prices)
    short_forecasts = _build_short_forecasts(initial_prices)

    # Experiment 1: PF only
    run_experiment(
        "PF only",
        strategy_forecasts={"Perfect Foresight": pf_forecasts},
        real_prices=real_prices,
        initial_prices=initial_prices,
    )

    # Experiment 2: PF + AlwaysLong
    run_experiment(
        "PF + AlwaysLong",
        strategy_forecasts={
            "Perfect Foresight": pf_forecasts,
            "Always Long": long_forecasts,
        },
        real_prices=real_prices,
        initial_prices=initial_prices,
    )

    # Experiment 3: PF + AlwaysLong + AlwaysShort
    run_experiment(
        "PF + AlwaysLong + AlwaysShort",
        strategy_forecasts={
            "Perfect Foresight": pf_forecasts,
            "Always Long": long_forecasts,
            "Always Short": short_forecasts,
        },
        real_prices=real_prices,
        initial_prices=initial_prices,
    )

    # Experiment 4: AlwaysLong only (no PF)
    run_experiment(
        "AlwaysLong only (no PF)",
        strategy_forecasts={"Always Long": long_forecasts},
        real_prices=real_prices,
        initial_prices=initial_prices,
    )

    # Experiment 5: Opposing strategies (Long vs Short, no PF)
    run_experiment(
        "AlwaysLong + AlwaysShort (no PF)",
        strategy_forecasts={
            "Always Long": long_forecasts,
            "Always Short": short_forecasts,
        },
        real_prices=real_prices,
        initial_prices=initial_prices,
    )


if __name__ == "__main__":
    main()
