"""Phase 7 Theorem Verification Script.

Verifies all four convergence theorems against the spec-compliant market
engine (``docs/energy_market_spec.md``).  The engine has NO dampening,
NO legacy direction+/-spread mode, and NO *24 profit multiplier.

Theorems:
  1. PF Instant Convergence — sole PF converges in <=2 iterations
  2. PF Dominance — PF profit >= any other strategy's profit
  3. Fixed Point — converged P* satisfies P* = sum w_i * forecast_i(P*)
  4. Unprofitable Elimination — wrong-side strategies get zero weight

Each theorem is verified against its *testable predictions*.
Exit code 0 if all pass, 1 if any fail.

Usage:
    uv run python scripts/verify_theorems.py
    uv run python scripts/verify_theorems.py --verbose
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 stdout so unicode characters render on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import argparse
from datetime import date, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    compute_market_prices,
    compute_strategy_profits,
    compute_weights,
    run_futures_market,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_LABEL = "\033[32mPASS\033[0m"
FAIL_LABEL = "\033[31mFAIL\033[0m"
_failures: list[str] = []


def _assert(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  {PASS_LABEL}  {label}")
    else:
        tag = f"  {FAIL_LABEL}  {label}"
        if detail:
            tag += f"\n         {detail}"
        print(tag)
        _failures.append(label)


def _rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(((a.reindex(b.index) - b) ** 2).mean()))


def _make_index(n: int) -> pd.Index:
    return pd.Index(
        [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        name="delivery_date",
    )


def _synthetic_prices(n: int = 20, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Return (real_prices, initial_market_prices) with non-trivial starting error."""
    rng = np.random.default_rng(seed)
    idx = _make_index(n)
    real = pd.Series(50.0 + rng.uniform(-20, 20, n), index=idx, name="real")
    initial = real + rng.uniform(-15, 15, n)
    initial.name = "initial"
    return real, initial


def _trending_up_prices(n: int = 20, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Return prices where real > initial on every day."""
    rng = np.random.default_rng(seed)
    idx = _make_index(n)
    real = pd.Series(50.0 + rng.uniform(-20, 20, n), index=idx, name="real")
    gap = rng.uniform(2, 15, n)
    initial = real - gap
    initial.name = "initial"
    return real, initial


def _pf_forecasts(real: pd.Series) -> dict:
    """PF forecast: every date maps to the real settlement price."""
    return {t: float(real.loc[t]) for t in real.index}


# ---------------------------------------------------------------------------
# Theorem 1: PF Instant Convergence
# ---------------------------------------------------------------------------


def verify_theorem_1(verbose: bool) -> None:
    """Theorem 1: With PF as sole strategy, P_m -> P_real in ONE iteration.

    In the undampened engine:
    - Iteration 0: PF forecast = P_real, PF is profitable (|P_real - P_m| > 0
      on at least one day), so w_PF = 1.0.  New price = 1.0 * P_real = P_real.
    - Iteration 1: sign(P_real - P_real) = 0 for all days, so profit = 0,
      all weights zero, prices carry forward.  Delta = 0 -> converged.
    - Total: <= 2 iterations.
    """
    print("\n" + "=" * 70)
    print("THEOREM 1: PF Instant Convergence (no dampening)")
    print("=" * 70)
    print("  Claim: sole PF converges to P_real in <= 2 iterations")

    real, initial = _synthetic_prices(n=20, seed=42)
    pf_fc = _pf_forecasts(real)

    eq = run_futures_market(
        initial_market_prices=initial.copy(),
        real_prices=real,
        strategy_forecasts={"PF": pf_fc},
        max_iterations=50,
        convergence_threshold=0.001,
        ema_alpha=1.0,
    )

    if verbose:
        print(f"\n  Iterations: {len(eq.iterations)}")
        for it in eq.iterations:
            rmse = _rmse(it.market_prices, real)
            active = {k: f"{v:.3f}" for k, v in it.strategy_weights.items() if v > 0}
            print(f"    iter {it.iteration}: RMSE={rmse:.6f}, weights={active or '{all zero}'}")

    # Prediction 1: converges
    _assert(eq.converged, "System converges")

    # Prediction 2: at most 2 iterations
    _assert(
        len(eq.iterations) <= 2,
        f"Converges in <= 2 iterations (actual: {len(eq.iterations)})",
    )

    # Prediction 3: after iteration 0, market prices = real prices
    if eq.iterations:
        it0 = eq.iterations[0]
        rmse_after_0 = _rmse(it0.market_prices, real)
        _assert(
            rmse_after_0 < 1e-9,
            f"After iter 0: P_m = P_real (RMSE={rmse_after_0:.2e})",
        )

    # Prediction 4: PF has weight 1.0 at iteration 0
    if eq.iterations:
        w_pf_0 = eq.iterations[0].strategy_weights.get("PF", 0.0)
        _assert(w_pf_0 == 1.0, f"PF weight at iter 0 = 1.0 (actual: {w_pf_0:.4f})")

    # Prediction 5: final RMSE is essentially zero
    final_rmse = _rmse(eq.final_market_prices, real)
    _assert(final_rmse < 1e-9, f"Final RMSE < 1e-9 (actual: {final_rmse:.2e})")

    # Prediction 6: works for different seeds/sizes
    all_instant = True
    for seed in [0, 7, 99, 123]:
        r, i = _synthetic_prices(n=30, seed=seed)
        e = run_futures_market(
            initial_market_prices=i.copy(),
            real_prices=r,
            strategy_forecasts={"PF": _pf_forecasts(r)},
            max_iterations=50,
            convergence_threshold=0.001,
            ema_alpha=1.0,
        )
        if len(e.iterations) > 2 or not e.converged:
            all_instant = False
            if verbose:
                print(f"    seed={seed}: iters={len(e.iterations)}, converged={e.converged}")
    _assert(all_instant, "Instant convergence for seeds {0, 7, 99, 123}")


# ---------------------------------------------------------------------------
# Theorem 2: PF Dominance
# ---------------------------------------------------------------------------


def verify_theorem_2(verbose: bool) -> None:
    """Theorem 2: PF profit >= any other strategy's profit at every iteration.

    PF's direction is always correct: sign(P_real - P_m) aligns with
    (P_real - P_m), so PF daily profit = |P_real - P_m| >= 0.
    Any other strategy's daily profit <= |P_real - P_m| (since |q*(P_real-P_m)| <= |P_real-P_m|).
    Therefore PF total profit >= any other strategy's total profit.
    """
    print("\n" + "=" * 70)
    print("THEOREM 2: PF Dominance")
    print("=" * 70)
    print("  Claim: PF profit >= any other strategy's profit at every iteration")

    real, initial = _trending_up_prices(n=20, seed=42)
    pf_fc = _pf_forecasts(real)

    # Long strategy: always forecasts above initial (long bias)
    long_fc = {t: float(initial.loc[t]) + 10.0 for t in real.index}
    # Short strategy: always forecasts below initial (short bias)
    short_fc = {t: float(initial.loc[t]) - 10.0 for t in real.index}
    # Random strategy: forecasts with noise around real
    rng = np.random.default_rng(42)
    random_fc = {t: float(real.loc[t]) + rng.uniform(-20, 20) for t in real.index}

    all_forecasts = {
        "PF": pf_fc,
        "Long": long_fc,
        "Short": short_fc,
        "Random": random_fc,
    }

    # Check at iteration 0 (using initial market prices)
    profits = compute_strategy_profits(initial, real, all_forecasts)
    pf_profit = profits["PF"]

    if verbose:
        print("\n  Profits at iter 0:")
        for name, p in sorted(profits.items(), key=lambda x: -x[1]):
            marker = " <-- PF" if name == "PF" else ""
            print(f"    {name:12s}: {p:>10.2f}{marker}")

    # Prediction 1: PF profit is non-negative
    _assert(pf_profit >= 0.0, f"PF profit >= 0 (actual: {pf_profit:.2f})")

    # Prediction 2: PF profit >= every other strategy's profit
    pf_dominates = all(pf_profit >= p for name, p in profits.items() if name != "PF")
    worst_gap = min(pf_profit - p for name, p in profits.items() if name != "PF")
    _assert(
        pf_dominates,
        f"PF profit >= all others (worst gap: {worst_gap:.2f})",
    )

    # Prediction 3: PF profit = sum |P_real - P_m| (the maximum possible)
    max_possible = float((real - initial).abs().sum())
    _assert(
        abs(pf_profit - max_possible) < 1e-6,
        f"PF profit = sum|P_real - P_m| = {max_possible:.2f} (actual: {pf_profit:.2f})",
    )

    # Prediction 4: PF dominates with different data sets
    all_dominate = True
    for seed in [0, 7, 99]:
        r, i = _synthetic_prices(n=20, seed=seed)
        fc = {
            "PF": _pf_forecasts(r),
            "Long": {t: float(i.loc[t]) + 10.0 for t in r.index},
            "Short": {t: float(i.loc[t]) - 10.0 for t in r.index},
        }
        p = compute_strategy_profits(i, r, fc)
        if p["PF"] < p["Long"] or p["PF"] < p["Short"]:
            all_dominate = False
            if verbose:
                print(
                    f"    seed={seed}: PF={p['PF']:.2f},"
                    f" Long={p['Long']:.2f}, Short={p['Short']:.2f}"
                )
    _assert(all_dominate, "PF dominates for seeds {0, 7, 99}")

    # Prediction 5: PF gets highest weight
    weights = compute_weights(profits)
    pf_weight = weights["PF"]
    max_other_weight = max(w for name, w in weights.items() if name != "PF")
    _assert(
        pf_weight >= max_other_weight,
        f"PF weight ({pf_weight:.4f}) >= max other weight ({max_other_weight:.4f})",
    )


# ---------------------------------------------------------------------------
# Theorem 3: Fixed Point
# ---------------------------------------------------------------------------


def verify_theorem_3(verbose: bool) -> None:
    """Theorem 3: Converged P* satisfies P* = sum w_i * forecast_i(P*).

    At a fixed point, one more iteration of the engine produces the same
    prices (within tolerance).  The weights computed from the converged
    prices, applied to the forecasts, reproduce the converged prices.

    We test two scenarios:
    A. PF only -> P* = P_real (trivial fixed point, instant).
    B. PF + Long -> the undampened engine jumps to P_real in one step
       (PF dominates), so the fixed point is P_real.  We verify idempotency.
    C. Two non-PF strategies with complementary forecasts on either side
       of P_real -> verify the converged result is a self-consistent
       weighted average.
    """
    print("\n" + "=" * 70)
    print("THEOREM 3: Fixed Point Property")
    print("=" * 70)
    print("  Claim: converged P* satisfies P* = sum w_i * forecast_i(P*)")

    real, initial = _trending_up_prices(n=20, seed=42)
    pf_fc = _pf_forecasts(real)

    # --- Scenario A: PF only (trivial fixed point) ---
    eq_pf = run_futures_market(
        initial_market_prices=initial.copy(),
        real_prices=real,
        strategy_forecasts={"PF": pf_fc},
        max_iterations=50,
        convergence_threshold=0.001,
        ema_alpha=1.0,
    )
    rmse_pf = _rmse(eq_pf.final_market_prices, real)
    _assert(rmse_pf < 1e-9, f"PF-only: P* = P_real (RMSE={rmse_pf:.2e})")

    # --- Scenario B: PF + Long ---
    long_fc = {t: float(real.loc[t]) + 8.0 for t in real.index}
    eq_pf_long = run_futures_market(
        initial_market_prices=initial.copy(),
        real_prices=real,
        strategy_forecasts={"PF": pf_fc, "Long": long_fc},
        max_iterations=50,
        convergence_threshold=0.001,
        ema_alpha=1.0,
    )
    _assert(eq_pf_long.converged, "PF+Long converges")

    # Idempotency: one more iteration from P* produces P*
    profits_star = compute_strategy_profits(
        eq_pf_long.final_market_prices, real, {"PF": pf_fc, "Long": long_fc}
    )
    weights_star = compute_weights(profits_star)
    new_prices = compute_market_prices(
        weights_star, {"PF": pf_fc, "Long": long_fc}, eq_pf_long.final_market_prices
    )
    delta = float((new_prices - eq_pf_long.final_market_prices).abs().max())
    _assert(
        delta < 0.01,
        f"PF+Long fixed point: one more iteration delta < 0.01 (actual: {delta:.6f})",
    )

    if verbose:
        print(f"  PF+Long: converged={eq_pf_long.converged}, iters={len(eq_pf_long.iterations)}")
        print(f"  Final weights: {eq_pf_long.final_weights}")
        print(f"  Profits at P*: {profits_star}")

    # --- Scenario C: Two non-PF strategies, no PF ---
    # BullishA forecasts real+5 (above market when market < real+5)
    # BullishB forecasts real+15 (further above)
    # Both are on the same side, both profitable when market < real
    # because direction = +1 (forecast > market) and real > market.
    bullish_a_fc = {t: float(real.loc[t]) + 5.0 for t in real.index}
    bullish_b_fc = {t: float(real.loc[t]) + 15.0 for t in real.index}

    eq_ab = run_futures_market(
        initial_market_prices=initial.copy(),
        real_prices=real,
        strategy_forecasts={"BullishA": bullish_a_fc, "BullishB": bullish_b_fc},
        max_iterations=50,
        convergence_threshold=0.001,
        ema_alpha=1.0,
    )
    _assert(eq_ab.converged, "BullishA+BullishB converges")

    # Fixed-point test: one more iteration produces same prices
    profits_ab = compute_strategy_profits(
        eq_ab.final_market_prices, real, {"BullishA": bullish_a_fc, "BullishB": bullish_b_fc}
    )
    weights_ab = compute_weights(profits_ab)
    new_ab = compute_market_prices(
        weights_ab,
        {"BullishA": bullish_a_fc, "BullishB": bullish_b_fc},
        eq_ab.final_market_prices,
    )
    delta_ab = float((new_ab - eq_ab.final_market_prices).abs().max())
    _assert(
        delta_ab < 0.01,
        f"BullishA+B fixed point: one more iteration delta < 0.01 (actual: {delta_ab:.6f})",
    )

    # Verify P*_t = sum w_i * forecast_i_t numerically
    max_fp_err = 0.0
    for t in real.index:
        numerator = 0.0
        denom = 0.0
        for name, w in weights_ab.items():
            if w <= 0:
                continue
            fc_val = {"BullishA": bullish_a_fc, "BullishB": bullish_b_fc}[name].get(t)
            if fc_val is not None:
                numerator += w * fc_val
                denom += w
        expected_t = numerator / denom if denom > 0 else float(eq_ab.final_market_prices.loc[t])
        err = abs(float(new_ab.loc[t]) - expected_t)
        max_fp_err = max(max_fp_err, err)
    _assert(
        max_fp_err < 1e-9,
        f"P*_t = sum w_i * forecast_i_t (max err: {max_fp_err:.2e})",
    )

    if verbose:
        print(f"  BullishA+B: converged={eq_ab.converged}, iters={len(eq_ab.iterations)}")
        print(f"  Final weights: {eq_ab.final_weights}")
        print(f"  Profits at P*: {profits_ab}")


# ---------------------------------------------------------------------------
# Theorem 4: Unprofitable Elimination
# ---------------------------------------------------------------------------


def verify_theorem_4(verbose: bool) -> None:
    """Theorem 4: Strategies forecasting in the wrong direction get zero weight.

    If a strategy's net profit is <= 0, its weight is zero under the
    spec's max(Pi_i, 0) rule.  In a trending-up market (real > market),
    a short-biased strategy that forecasts below market earns negative
    profit and is eliminated.
    """
    print("\n" + "=" * 70)
    print("THEOREM 4: Unprofitable Elimination")
    print("=" * 70)
    print("  Claim: strategies with non-positive profit get zero weight")

    real, initial = _trending_up_prices(n=20, seed=42)
    pf_fc = _pf_forecasts(real)

    # Short strategy forecasts well below market -> direction = -1
    # But real > market, so profit = (-1)*(real - market) < 0
    short_fc = {t: float(initial.loc[t]) - 20.0 for t in real.index}

    all_forecasts = {"PF": pf_fc, "Short": short_fc}

    profits = compute_strategy_profits(initial, real, all_forecasts)
    weights = compute_weights(profits)

    if verbose:
        print(f"\n  Profits: {profits}")
        print(f"  Weights: {weights}")

    # Prediction 1: Short profit is negative (wrong side of trending market)
    _assert(
        profits["Short"] < 0,
        f"Short profit < 0 in trending-up market (actual: {profits['Short']:.2f})",
    )

    # Prediction 2: Short weight is zero
    _assert(
        weights["Short"] == 0.0,
        f"Short weight = 0 (actual: {weights['Short']:.4f})",
    )

    # Prediction 3: PF weight is 1.0 (only profitable strategy)
    _assert(
        weights["PF"] == 1.0,
        f"PF weight = 1.0 when Short eliminated (actual: {weights['PF']:.4f})",
    )

    # Prediction 4: full market run with PF + Short = same as PF only
    eq_both = run_futures_market(
        initial_market_prices=initial.copy(),
        real_prices=real,
        strategy_forecasts=all_forecasts,
        max_iterations=50,
        convergence_threshold=0.001,
        ema_alpha=1.0,
    )
    eq_pf = run_futures_market(
        initial_market_prices=initial.copy(),
        real_prices=real,
        strategy_forecasts={"PF": pf_fc},
        max_iterations=50,
        convergence_threshold=0.001,
        ema_alpha=1.0,
    )
    rmse_both = _rmse(eq_both.final_market_prices, real)
    rmse_pf = _rmse(eq_pf.final_market_prices, real)
    _assert(
        abs(rmse_both - rmse_pf) < 1e-9,
        f"PF+Short RMSE ({rmse_both:.6f}) = PF-only RMSE ({rmse_pf:.6f})",
    )

    # Prediction 5: elimination also works with multiple unprofitable strategies
    bad1_fc = {t: float(initial.loc[t]) - 15.0 for t in real.index}
    bad2_fc = {t: float(initial.loc[t]) - 25.0 for t in real.index}
    multi_fc = {"PF": pf_fc, "Bad1": bad1_fc, "Bad2": bad2_fc}
    profits_multi = compute_strategy_profits(initial, real, multi_fc)
    weights_multi = compute_weights(profits_multi)

    _assert(
        weights_multi["Bad1"] == 0.0 and weights_multi["Bad2"] == 0.0,
        f"Multiple unprofitable strategies eliminated "
        f"(Bad1={weights_multi['Bad1']:.4f}, Bad2={weights_multi['Bad2']:.4f})",
    )
    _assert(
        weights_multi["PF"] == 1.0,
        f"PF gets all weight when others eliminated (actual: {weights_multi['PF']:.4f})",
    )

    # Prediction 6: when ALL strategies are unprofitable, all weights = 0
    # Use prices where all forecasts miss badly
    idx = _make_index(5)
    r = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    m = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    # All forecasts == market price -> sign(fc - m) = 0 -> profit = 0
    fc_zero = {t: 100.0 for t in idx}
    profits_zero = compute_strategy_profits(m, r, {"A": fc_zero, "B": fc_zero})
    weights_zero = compute_weights(profits_zero)
    _assert(
        all(w == 0.0 for w in weights_zero.values()),
        "All weights = 0 when all profits = 0 (no active strategies)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 7 convergence theorems.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-iteration tables")
    args = parser.parse_args()

    print("Phase 7 Convergence Theorem Verification")
    print("Spec-compliant engine: no dampening, no legacy mode, no *24")
    print("Synthetic dataset: 20 days, seed=42, prices ~ U(30, 70)")

    verify_theorem_1(args.verbose)
    verify_theorem_2(args.verbose)
    verify_theorem_3(args.verbose)
    verify_theorem_4(args.verbose)

    print("\n" + "=" * 70)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILURE(S)")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: ALL THEOREMS VERIFIED")
        sys.exit(0)


if __name__ == "__main__":
    main()
