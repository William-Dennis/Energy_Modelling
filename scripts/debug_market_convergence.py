"""Diagnostic: trace why MAE vs real price doesn't converge.

Runs the market engine with 1, 2, and 3 strategies (including PF) and
prints per-iteration MAE, PF weight, and raw price update internals.

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

import numpy as np
import pandas as pd

from energy_modelling.backtest.feature_engineering import add_derived_features
from energy_modelling.backtest.futures_market_engine import (
    compute_market_prices,
    compute_strategy_profits,
    compute_weights,
    run_futures_market,
)
from energy_modelling.backtest.runner import _normalise_daily_data
from energy_modelling.dashboard._backtest import _resolve_path, load_daily


def _build_pf_direction_and_forecast(real_prices: pd.Series, market_prices: pd.Series):
    """PF always knows real price — direction is sign(real - market)."""
    direction = pd.Series(
        {t: int(np.sign(real_prices[t] - market_prices[t])) or 1 for t in real_prices.index},
        name="PF",
    )
    # forecast = real price
    forecasts = real_prices.to_dict()
    return direction, forecasts


def _build_always_long(real_prices: pd.Series):
    direction = pd.Series(1, index=real_prices.index, name="AlwaysLong")
    # legacy strategy — no forecast
    return direction, {}


def run_experiment(label: str, directions, forecasts_dict, real_prices, initial_prices):
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {label}")
    print(f"{'=' * 70}")
    print(f"  Initial MAE vs real:     {(initial_prices - real_prices).abs().mean():.4f} EUR/MWh")
    print(f"  Initial MAE vs prev-day: {0:.4f} EUR/MWh (= 0 by definition)")
    print()

    eq = run_futures_market(
        directions=directions,
        initial_market_prices=initial_prices.copy(),
        real_prices=real_prices,
        max_iterations=20,
        convergence_threshold=0.001,
        dampening=0.5,
        strategy_forecasts=forecasts_dict if forecasts_dict else None,
    )

    print(
        f"  Converged: {eq.converged}  |  Iterations: {len(eq.iterations)}  |  Delta: {eq.convergence_delta:.6f}"
    )
    print()
    print(
        f"  {'Iter':>4}  {'MAE_real':>10}  {'MAE_prev':>10}  {'Delta':>10}  {'Active strategies & weights'}"
    )
    print(f"  {'-' * 4}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 40}")

    current = initial_prices.copy().astype(float)
    for it in eq.iterations:
        # The iteration stores market_prices BEFORE dampening is applied in run_futures_market.
        # But run_futures_market applies: new = 0.5*result.market_prices + 0.5*current
        # So we need to track the actual current prices the same way the loop does.
        new = 0.5 * it.market_prices + 0.5 * current
        mae_real = float((new - real_prices).abs().mean())
        mae_prev = float((new - initial_prices).abs().mean())
        delta = float((new - current).abs().max())
        active = {k: f"{v:.3f}" for k, v in it.strategy_weights.items() if v > 0}
        print(
            f"  {it.iteration:>4}  {mae_real:>10.4f}  {mae_prev:>10.4f}  {delta:>10.6f}  {active}"
        )
        current = new

    print()
    print(f"  FINAL MAE vs real:     {(current - real_prices).abs().mean():.4f} EUR/MWh")
    print(f"  FINAL MAE vs prev-day: {(current - initial_prices).abs().mean():.4f} EUR/MWh")

    # --- Deep-dive: show what PF profit is at iteration 0 ---
    if "Perfect Foresight" in directions:
        print()
        print("  --- PF profit trace (iter 0) ---")
        p0 = initial_prices.copy()
        profits0 = compute_strategy_profits(
            directions, p0, real_prices, forecasts_dict if forecasts_dict else None
        )
        weights0 = compute_weights(profits0)
        print(f"  Profits: { {k: f'{v:.1f}' for k, v in profits0.items()} }")
        print(f"  Weights: { {k: f'{v:.4f}' for k, v in weights0.items()} }")

        # What does compute_market_prices actually produce?
        raw_new = compute_market_prices(
            directions,
            weights0,
            p0,
            forecast_spread=1.0,
            strategy_forecasts=forecasts_dict if forecasts_dict else None,
        )
        print(f"  Raw new price (first 5 days): {raw_new.head().values}")
        print(f"  Real prices   (first 5 days): {real_prices.head().values}")
        pf_fc = (forecasts_dict or {}).get("Perfect Foresight") or {}
        print(
            f"  PF forecast   (first 5 days): {[pf_fc.get(t, 'N/A') for t in real_prices.head().index]}"
        )
        print(f"  PF direction  (first 5 days): {directions['Perfect Foresight'].head().values}")


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
    print(f"Real price range:    {real_prices.min():.1f} – {real_prices.max():.1f} EUR/MWh")
    print(f"Prev-day price range:{initial_prices.min():.1f} – {initial_prices.max():.1f} EUR/MWh")
    print(
        f"Price std: {real_prices.std():.2f}  |  MAE(real, prev-day): {(real_prices - initial_prices).abs().mean():.2f}"
    )

    # Build PF direction (fixed at iter 0 initial prices, as per run_backtest)
    pf_dir_fixed, pf_forecasts = _build_pf_direction_and_forecast(real_prices, initial_prices)
    long_dir, long_fc = _build_always_long(real_prices)

    # Experiment 1: PF only, with forecasts
    run_experiment(
        "PF only — forecast mode",
        directions={"Perfect Foresight": pf_dir_fixed},
        forecasts_dict={"Perfect Foresight": pf_forecasts},
        real_prices=real_prices,
        initial_prices=initial_prices,
    )

    # Experiment 2: PF only, WITHOUT forecasts (legacy fallback)
    run_experiment(
        "PF only — NO forecasts (legacy fallback)",
        directions={"Perfect Foresight": pf_dir_fixed},
        forecasts_dict=None,
        real_prices=real_prices,
        initial_prices=initial_prices,
    )

    # Experiment 3: PF + AlwaysLong, with forecasts
    run_experiment(
        "PF + AlwaysLong — forecast mode",
        directions={"Perfect Foresight": pf_dir_fixed, "Always Long": long_dir},
        forecasts_dict={"Perfect Foresight": pf_forecasts},
        real_prices=real_prices,
        initial_prices=initial_prices,
    )

    # Experiment 4: What the REAL run does — PF direction is FIXED from run_backtest
    # (computed against last_settlement_price, NOT updated market price).
    # Show this clearly:
    print(f"\n{'=' * 70}")
    print("KEY QUESTION: Is the PF direction in the real run fixed or adaptive?")
    print(f"{'=' * 70}")
    pf_correct = int((pf_dir_fixed * (real_prices - initial_prices) > 0).sum())
    pf_total = int(pf_dir_fixed.notna().sum())
    print(f"  PF direction is computed by run_backtest against last_settlement_price.")
    print(f"  PF correct days (vs last_settlement_price): {pf_correct}/{pf_total}")
    print(f"  NOTE: PF direction is FIXED before iteration starts.")
    print(f"  If market_price drifts toward real_price, PF direction may become WRONG")
    print(f"  (e.g. market > real but PF still says +1 from iter-0 direction).")
    print()

    # Show how many days PF direction disagrees with (real - market_at_iter_k)
    # as the market approaches real
    print("  Days where PF direction conflicts with (real - market) at various price levels:")
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interpolated = initial_prices + frac * (real_prices - initial_prices)
        agrees = int((pf_dir_fixed * (real_prices - interpolated) > 0).sum())
        print(
            f"    Market at {int(frac * 100):3d}% toward real: PF correct {agrees}/{pf_total} days"
        )


if __name__ == "__main__":
    main()
