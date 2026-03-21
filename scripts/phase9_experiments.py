"""Phase 9: Monotone-convergence experiment sweep.

Goal: find a market engine configuration that achieves **strictly monotonically
decreasing deltas over 5+ consecutive iterations** on BOTH 2024 and 2025
simultaneously.  The Phase 8b winner (K=30, w=1) converges on both years but
via a lucky dip — the new criterion requires provably decreasing dynamics.

New ideas explored:
  G1  : Softmax temperature weighting (replaces hard positive-profit gating)
  G2  : Soft-sign via tanh (replaces hard sign() discontinuity)
  G3  : Weight EMA across iterations (smooth active-set transitions)
  G4  : Profit EMA across iterations (smooth profit signal)
  G5  : Annealing alpha (starts low, increases as iteration grows)
  G6  : Combinations of G1+G2, G2+K, G1+K, G3+K, G4+K, G1+G2+K
  G7  : Fine-tuned softmax temperatures with running-avg
  G8  : Soft-sign with varied sigma and running-avg

Three-round structure:
  Round 1 (broad sweep, max_iters=300): ~50 configs
  Round 2 (focused, max_iters=400): survivors from Round 1 + parameter sweeps
  Round 3 (fine-tune, max_iters=500): top candidates, tighter params

Usage:
    uv run python scripts/phase9_experiments.py
    uv run python scripts/phase9_experiments.py --round 1
    uv run python scripts/phase9_experiments.py --round 2
    uv run python scripts/phase9_experiments.py --round 3
    uv run python scripts/phase9_experiments.py --round 1 --max-iters 200
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
PHASE9_DIR = RESULTS_DIR / "phase9"

CONVERGENCE_THRESHOLD = 0.01
MONOTONE_WINDOW = 5  # strict: 5+ consecutive strictly decreasing deltas


# ---------------------------------------------------------------------------
# Convergence helpers (used in diagnostics, not in the engine call)
# ---------------------------------------------------------------------------


def monotone_converged(
    delta_history: list[float], window: int = 5, threshold: float = 0.01
) -> bool:
    """Return True if the last ``window`` deltas are strictly decreasing AND final < threshold."""
    if len(delta_history) < window:
        return False
    tail = delta_history[-window:]
    return all(tail[i] < tail[i - 1] for i in range(1, len(tail))) and tail[-1] < threshold


def extract_delta_history(result_obj) -> list[float]:
    """Extract per-iteration max-delta from a FuturesMarketEquilibrium."""
    iters = result_obj.iterations
    if len(iters) < 2:
        return []
    prices = [it.market_prices.to_numpy() for it in iters]
    deltas = [float(np.abs(prices[i] - prices[i - 1]).max()) for i in range(1, len(prices))]
    return deltas


# ---------------------------------------------------------------------------
# Experiment registries
# ---------------------------------------------------------------------------


def _build_experiments_round1() -> list[dict]:
    """Round 1: broad sweep of new ideas."""
    experiments: list[dict] = []

    def _add(exp_id: str, label: str, **kwargs) -> None:
        cfg = {
            "id": exp_id,
            "label": label,
            "alpha": 1.0,
            "weight_mode": "linear",
            "weight_cap": 1.0,
            "softmax_temp": 1.0,
            "price_mode": "mean",
            "running_avg_k": None,
            "soft_sign_sigma": None,
            "weight_ema_beta": None,
            "profit_ema_beta": None,
        }
        cfg.update(kwargs)
        experiments.append(cfg)

    # ------------------------------------------------------------------
    # Baseline for comparison
    # ------------------------------------------------------------------
    _add("BASE_K30", "Baseline K=30 (Phase 8b winner)", running_avg_k=30)

    # ------------------------------------------------------------------
    # G1: Softmax temperature weighting
    # All strategies get weight — no hard cutoff at zero profit.
    # High T → uniform, low T → winner-take-all.
    # ------------------------------------------------------------------
    for T in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        _add(f"G1_T{T}", f"Softmax T={T}", weight_mode="softmax", softmax_temp=T)

    # ------------------------------------------------------------------
    # G2: Soft-sign via tanh (replaces hard sign discontinuity)
    # sigma = scale of transition; small sigma → sharp, large → soft
    # ------------------------------------------------------------------
    for sigma in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        _add(f"G2_s{sigma}", f"Soft-sign sigma={sigma}", soft_sign_sigma=sigma)

    # ------------------------------------------------------------------
    # G3: Weight EMA across iterations
    # Smooth the active-set transitions
    # ------------------------------------------------------------------
    for beta in [0.5, 0.7, 0.85, 0.9, 0.95]:
        _add(f"G3_b{beta}", f"Weight EMA beta={beta}", weight_ema_beta=beta)

    # ------------------------------------------------------------------
    # G4: Profit EMA across iterations
    # Smooth the profit signal before computing weights
    # ------------------------------------------------------------------
    for beta in [0.5, 0.7, 0.85, 0.9]:
        _add(f"G4_b{beta}", f"Profit EMA beta={beta}", profit_ema_beta=beta)

    # ------------------------------------------------------------------
    # G5: Running-avg K with weight EMA (combine two smoothing mechanisms)
    # ------------------------------------------------------------------
    for K in [5, 10, 15, 20]:
        for beta in [0.7, 0.85, 0.9]:
            _add(
                f"G5_K{K}_b{beta}",
                f"K={K} + Weight EMA beta={beta}",
                running_avg_k=K,
                weight_ema_beta=beta,
            )

    # ------------------------------------------------------------------
    # G6: Softmax + running-avg
    # ------------------------------------------------------------------
    for T in [2.0, 5.0, 10.0, 20.0]:
        for K in [5, 10, 20]:
            _add(
                f"G6_T{T}_K{K}",
                f"Softmax T={T} + K={K}",
                weight_mode="softmax",
                softmax_temp=T,
                running_avg_k=K,
            )

    # ------------------------------------------------------------------
    # G7: Soft-sign + running-avg
    # ------------------------------------------------------------------
    for sigma in [5.0, 10.0, 20.0]:
        for K in [5, 10, 20]:
            _add(
                f"G7_s{sigma}_K{K}",
                f"Soft-sign sigma={sigma} + K={K}",
                soft_sign_sigma=sigma,
                running_avg_k=K,
            )

    # ------------------------------------------------------------------
    # G8: Softmax + soft-sign (both mechanisms at once)
    # ------------------------------------------------------------------
    for T in [5.0, 10.0]:
        for sigma in [5.0, 10.0]:
            _add(
                f"G8_T{T}_s{sigma}",
                f"Softmax T={T} + Soft-sign sigma={sigma}",
                weight_mode="softmax",
                softmax_temp=T,
                soft_sign_sigma=sigma,
            )

    # ------------------------------------------------------------------
    # G9: Profit EMA + running-avg
    # ------------------------------------------------------------------
    for beta in [0.7, 0.85, 0.9]:
        for K in [5, 10, 20]:
            _add(
                f"G9_b{beta}_K{K}",
                f"Profit EMA beta={beta} + K={K}",
                profit_ema_beta=beta,
                running_avg_k=K,
            )

    return experiments


def _build_experiments_round2(survivor_ids: list[str]) -> list[dict]:
    """Round 2: focused sweep around Round 1 survivors.

    Takes the top ideas from Round 1 and sweeps tighter parameter grids,
    plus some fresh combinations.
    """
    experiments: list[dict] = []

    def _add(exp_id: str, label: str, **kwargs) -> None:
        cfg = {
            "id": exp_id,
            "label": label,
            "alpha": 1.0,
            "weight_mode": "linear",
            "weight_cap": 1.0,
            "softmax_temp": 1.0,
            "price_mode": "mean",
            "running_avg_k": None,
            "soft_sign_sigma": None,
            "weight_ema_beta": None,
            "profit_ema_beta": None,
        }
        cfg.update(kwargs)
        experiments.append(cfg)

    # Tight softmax grid
    for T in [3.0, 4.0, 6.0, 8.0, 12.0, 15.0, 25.0, 30.0]:
        _add(f"R2_G1_T{T}", f"Softmax T={T}", weight_mode="softmax", softmax_temp=T)

    # Softmax + K fine-grid
    for T in [5.0, 8.0, 10.0, 15.0]:
        for K in [7, 12, 15, 20, 25]:
            _add(
                f"R2_G6_T{T}_K{K}",
                f"Softmax T={T} + K={K}",
                weight_mode="softmax",
                softmax_temp=T,
                running_avg_k=K,
            )

    # Weight EMA fine-grid
    for beta in [0.75, 0.80, 0.87, 0.92, 0.93, 0.95, 0.97]:
        _add(f"R2_G3_b{beta}", f"Weight EMA beta={beta}", weight_ema_beta=beta)

    # Weight EMA + K
    for beta in [0.8, 0.85, 0.9, 0.92]:
        for K in [7, 10, 12, 15]:
            _add(
                f"R2_G5_K{K}_b{beta}",
                f"K={K} + Weight EMA beta={beta}",
                running_avg_k=K,
                weight_ema_beta=beta,
            )

    # Soft-sign fine-grid
    for sigma in [3.0, 7.0, 15.0, 25.0, 30.0]:
        _add(f"R2_G2_s{sigma}", f"Soft-sign sigma={sigma}", soft_sign_sigma=sigma)

    # Profit EMA fine-grid
    for beta in [0.6, 0.75, 0.80, 0.87, 0.92]:
        _add(f"R2_G4_b{beta}", f"Profit EMA beta={beta}", profit_ema_beta=beta)

    # Triple combo: softmax + weight EMA + K
    for T in [5.0, 10.0]:
        for beta in [0.8, 0.9]:
            for K in [10, 15]:
                _add(
                    f"R2_G10_T{T}_b{beta}_K{K}",
                    f"Softmax T={T} + WEMA b={beta} + K={K}",
                    weight_mode="softmax",
                    softmax_temp=T,
                    weight_ema_beta=beta,
                    running_avg_k=K,
                )

    # Soft-sign + weight EMA + K
    for sigma in [5.0, 10.0]:
        for beta in [0.8, 0.9]:
            for K in [10, 15]:
                _add(
                    f"R2_G11_s{sigma}_b{beta}_K{K}",
                    f"Soft-sign s={sigma} + WEMA b={beta} + K={K}",
                    soft_sign_sigma=sigma,
                    weight_ema_beta=beta,
                    running_avg_k=K,
                )

    return experiments


def _build_experiments_round3(survivor_ids: list[str], survivor_cfgs: list[dict]) -> list[dict]:
    """Round 3: fine-tune the best survivors, plus targeted micro-variations."""
    experiments: list[dict] = []

    # Include all survivors as-is
    for cfg in survivor_cfgs:
        experiments.append(cfg)

    def _add(exp_id: str, label: str, **kwargs) -> None:
        cfg = {
            "id": exp_id,
            "label": label,
            "alpha": 1.0,
            "weight_mode": "linear",
            "weight_cap": 1.0,
            "softmax_temp": 1.0,
            "price_mode": "mean",
            "running_avg_k": None,
            "soft_sign_sigma": None,
            "weight_ema_beta": None,
            "profit_ema_beta": None,
        }
        cfg.update(kwargs)
        experiments.append(cfg)

    # Micro-grid around best survivors — dynamically constructed after seeing Round 2 data.
    # Placeholder for the most promising idea found in Round 2: try fine variations.
    for T in [7.0, 9.0, 11.0, 13.0]:
        for K in [8, 10, 12, 14, 16, 18]:
            _add(
                f"R3_G6_T{T}_K{K}",
                f"Softmax T={T} + K={K}",
                weight_mode="softmax",
                softmax_temp=T,
                running_avg_k=K,
            )

    for beta in [0.82, 0.84, 0.86, 0.88, 0.91, 0.94, 0.96]:
        for K in [8, 10, 12, 14]:
            _add(
                f"R3_G5_K{K}_b{beta}",
                f"K={K} + Weight EMA beta={beta}",
                running_avg_k=K,
                weight_ema_beta=beta,
            )

    return experiments


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    cfg: dict,
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    max_iterations: int = 300,
    monotone_window: int = MONOTONE_WINDOW,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
) -> dict:
    """Run one experiment configuration and return metrics dict."""
    from energy_modelling.backtest.futures_market_engine import run_futures_market

    result_obj = run_futures_market(
        initial_market_prices=initial_prices.copy(),
        real_prices=real_prices,
        strategy_forecasts=strategy_forecasts,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        monotone_window=monotone_window,
        alpha=cfg.get("alpha", 1.0),
        weight_mode=cfg.get("weight_mode", "linear"),
        weight_cap=cfg.get("weight_cap", 1.0),
        softmax_temp=cfg.get("softmax_temp", 1.0),
        price_mode=cfg.get("price_mode", "mean"),
        running_avg_k=cfg.get("running_avg_k"),
        soft_sign_sigma=cfg.get("soft_sign_sigma"),
        weight_ema_beta=cfg.get("weight_ema_beta"),
        profit_ema_beta=cfg.get("profit_ema_beta"),
    )

    final_prices = result_obj.final_market_prices
    real_aligned = real_prices.reindex(final_prices.index)
    mae = float((final_prices - real_aligned).abs().mean())

    # Compute delta history from iteration snapshots
    delta_history = extract_delta_history(result_obj)

    # Check if monotone convergence was genuinely achieved
    genuinely_monotone = False
    if result_obj.converged:
        # Verify the last monotone_window deltas are strictly decreasing
        genuinely_monotone = monotone_converged(
            delta_history, window=monotone_window, threshold=convergence_threshold
        )

    # Count spikes > 1.0 as oscillation proxy
    n_spikes = sum(1 for d in delta_history if d > 1.0)

    # Longest monotonically decreasing tail (for partial credit in pruning)
    longest_mono = 0
    current_mono = 1
    for i in range(1, len(delta_history)):
        if delta_history[i] < delta_history[i - 1]:
            current_mono += 1
            longest_mono = max(longest_mono, current_mono)
        else:
            current_mono = 1

    return {
        "id": cfg["id"],
        "label": cfg["label"],
        "converged": result_obj.converged,
        "genuinely_monotone": genuinely_monotone,
        "n_iterations": len(result_obj.iterations),
        "final_delta": round(result_obj.convergence_delta, 6),
        "mae": round(mae, 4),
        "n_spikes": n_spikes,
        "longest_mono_tail": longest_mono,
        "weight_mode": cfg.get("weight_mode", "linear"),
        "softmax_temp": cfg.get("softmax_temp", 1.0),
        "running_avg_k": cfg.get("running_avg_k"),
        "soft_sign_sigma": cfg.get("soft_sign_sigma"),
        "weight_ema_beta": cfg.get("weight_ema_beta"),
        "profit_ema_beta": cfg.get("profit_ema_beta"),
        "alpha": cfg.get("alpha", 1.0),
        # Store a compact delta tail for diagnostics (last 20)
        "delta_tail": delta_history[-20:] if len(delta_history) >= 20 else delta_history,
    }


# ---------------------------------------------------------------------------
# Pruning utilities
# ---------------------------------------------------------------------------


def prune_both_years(
    rows_2024: list[dict],
    rows_2025: list[dict],
    require_converged: bool = True,
) -> list[str]:
    """Return experiment IDs that converge (genuinely monotone) on BOTH years."""
    conv_2024 = {r["id"] for r in rows_2024 if r["converged"] and r["genuinely_monotone"]}
    conv_2025 = {r["id"] for r in rows_2025 if r["converged"] and r["genuinely_monotone"]}
    both = conv_2024 & conv_2025
    logger.info(
        "  Conv 2024: %d | Conv 2025: %d | Both: %d", len(conv_2024), len(conv_2025), len(both)
    )
    return sorted(both)


def rank_survivors(
    rows_2024: list[dict], rows_2025: list[dict], survivor_ids: list[str]
) -> list[dict]:
    """Rank surviving configs by combined score: fewer iterations + lower MAE."""
    by_id_24 = {r["id"]: r for r in rows_2024}
    by_id_25 = {r["id"]: r for r in rows_2025}
    ranked = []
    for eid in survivor_ids:
        r24 = by_id_24.get(eid, {})
        r25 = by_id_25.get(eid, {})
        n_iters = (r24.get("n_iterations", 9999) + r25.get("n_iterations", 9999)) / 2
        mae_avg = (r24.get("mae", 999) + r25.get("mae", 999)) / 2
        ranked.append(
            {
                "id": eid,
                "n_iters_avg": n_iters,
                "mae_avg": mae_avg,
                "score": n_iters + mae_avg * 10,  # iterations dominate
            }
        )
    ranked.sort(key=lambda x: x["score"])
    return ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 monotone-convergence experiment sweep.")
    parser.add_argument(
        "--round",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run specific round only (default: all rounds sequentially).",
    )
    parser.add_argument("--max-iters", type=int, default=300, help="Max iterations (default: 300).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    PHASE9_DIR.mkdir(parents=True, exist_ok=True)

    # Load pre-computed forecast data
    data_by_year: dict[int, dict] = {}
    for year in [2024, 2025]:
        fpath = PHASE8_DIR / f"forecasts_{year}.pkl"
        if not fpath.exists():
            logger.error("Missing %s — run phase8_collect_forecasts.py first", fpath)
            sys.exit(1)
        with open(fpath, "rb") as f:
            data_by_year[year] = pickle.load(f)
        logger.info(
            "Year %d: %d strategies, %d eval days",
            year,
            len(data_by_year[year]["strategy_forecasts"]),
            len(data_by_year[year]["real_prices"]),
        )

    run_rounds = [args.round] if args.round else [1, 2, 3]
    all_results: dict[int, dict[int, list[dict]]] = {}  # round -> year -> rows

    # -----------------------------------------------------------------------
    # Round 1
    # -----------------------------------------------------------------------
    if 1 in run_rounds:
        logger.info("=" * 60)
        logger.info("ROUND 1: broad sweep")
        logger.info("=" * 60)
        experiments = _build_experiments_round1()
        logger.info("  %d experiments × 2 years", len(experiments))

        rows_by_year: dict[int, list[dict]] = {2024: [], 2025: []}
        for year in [2024, 2025]:
            d = data_by_year[year]
            t_start = time.perf_counter()
            for i, cfg in enumerate(experiments, 1):
                t0 = time.perf_counter()
                row = run_experiment(
                    cfg=cfg,
                    strategy_forecasts=d["strategy_forecasts"],
                    real_prices=d["real_prices"],
                    initial_prices=d["initial_prices"],
                    max_iterations=args.max_iters,
                )
                row["year"] = year
                rows_by_year[year].append(row)
                elapsed = time.perf_counter() - t0
                flag = (
                    "✓MONO"
                    if row["genuinely_monotone"]
                    else ("✓conv" if row["converged"] else "     ")
                )
                logger.info(
                    "  R1 [%d/%d] %d %-35s %s iters=%d mae=%.1f spikes=%d %.2fs",
                    i,
                    len(experiments),
                    year,
                    cfg["id"],
                    flag,
                    row["n_iterations"],
                    row["mae"],
                    row["n_spikes"],
                    elapsed,
                )
            logger.info("  Year %d done in %.1fs", year, time.perf_counter() - t_start)

        # Save Round 1 results
        all_rows_r1: list[dict] = rows_by_year[2024] + rows_by_year[2025]
        df_r1 = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "delta_tail"} for r in all_rows_r1]
        )
        out_r1 = PHASE9_DIR / "results_round1.csv"
        df_r1.to_csv(out_r1, index=False)
        logger.info("Round 1 results saved to %s", out_r1)

        # Analyse survivors
        survivor_ids = prune_both_years(rows_by_year[2024], rows_by_year[2025])
        ranked = rank_survivors(rows_by_year[2024], rows_by_year[2025], survivor_ids)

        logger.info("\nRound 1 SURVIVORS (monotone on both years):")
        for r in ranked[:20]:
            logger.info(
                "  %-40s iters_avg=%.0f mae_avg=%.2f", r["id"], r["n_iters_avg"], r["mae_avg"]
            )

        if not survivor_ids:
            logger.warning("NO configs converged monotonically on both years in Round 1!")
            logger.info("Showing partial results (longest monotone tail on both years):")
            # Fall back to best partial convergence
            combined: list[dict] = []
            for r24 in rows_by_year[2024]:
                r25 = next((r for r in rows_by_year[2025] if r["id"] == r24["id"]), None)
                if r25:
                    combined.append(
                        {
                            "id": r24["id"],
                            "mono_24": r24["longest_mono_tail"],
                            "mono_25": r25["longest_mono_tail"],
                            "conv_24": r24["converged"],
                            "conv_25": r25["converged"],
                            "iters_24": r24["n_iterations"],
                            "iters_25": r25["n_iterations"],
                        }
                    )
            combined.sort(key=lambda x: -(x["mono_24"] + x["mono_25"]))
            for c in combined[:10]:
                logger.info(
                    "  %-40s mono_24=%d mono_25=%d conv=(%s/%s) iters=(%d/%d)",
                    c["id"],
                    c["mono_24"],
                    c["mono_25"],
                    "✓" if c["conv_24"] else "✗",
                    "✓" if c["conv_25"] else "✗",
                    c["iters_24"],
                    c["iters_25"],
                )

        all_results[1] = rows_by_year
        all_results["r1_survivors"] = survivor_ids  # type: ignore[assignment]
        all_results["r1_rows"] = rows_by_year  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Round 2
    # -----------------------------------------------------------------------
    if 2 in run_rounds:
        logger.info("=" * 60)
        logger.info("ROUND 2: focused sweep")
        logger.info("=" * 60)

        # If we ran Round 1 in this session, use its survivors; else try to load
        if "r1_survivors" in all_results:
            survivor_ids_r1 = all_results["r1_survivors"]  # type: ignore[index]
        else:
            r1_path = PHASE9_DIR / "results_round1.csv"
            if r1_path.exists():
                df_r1 = pd.read_csv(r1_path)
                # Re-derive survivors: converged + genuinely_monotone on both years
                ids_24 = set(df_r1[(df_r1.year == 2024) & df_r1.genuinely_monotone]["id"])
                ids_25 = set(df_r1[(df_r1.year == 2025) & df_r1.genuinely_monotone]["id"])
                survivor_ids_r1 = sorted(ids_24 & ids_25)
                logger.info("Loaded %d Round 1 survivors from CSV", len(survivor_ids_r1))
            else:
                survivor_ids_r1 = []
                logger.warning("No Round 1 results found; running full Round 2 grid")

        experiments_r2 = _build_experiments_round2(survivor_ids_r1)
        logger.info("  %d experiments × 2 years", len(experiments_r2))
        max_iters_r2 = max(args.max_iters, 400)

        rows_by_year_r2: dict[int, list[dict]] = {2024: [], 2025: []}
        for year in [2024, 2025]:
            d = data_by_year[year]
            t_start = time.perf_counter()
            for i, cfg in enumerate(experiments_r2, 1):
                t0 = time.perf_counter()
                row = run_experiment(
                    cfg=cfg,
                    strategy_forecasts=d["strategy_forecasts"],
                    real_prices=d["real_prices"],
                    initial_prices=d["initial_prices"],
                    max_iterations=max_iters_r2,
                )
                row["year"] = year
                rows_by_year_r2[year].append(row)
                elapsed = time.perf_counter() - t0
                flag = (
                    "✓MONO"
                    if row["genuinely_monotone"]
                    else ("✓conv" if row["converged"] else "     ")
                )
                logger.info(
                    "  R2 [%d/%d] %d %-40s %s iters=%d mae=%.1f spikes=%d %.2fs",
                    i,
                    len(experiments_r2),
                    year,
                    cfg["id"],
                    flag,
                    row["n_iterations"],
                    row["mae"],
                    row["n_spikes"],
                    elapsed,
                )
            logger.info("  Year %d done in %.1fs", year, time.perf_counter() - t_start)

        all_rows_r2 = rows_by_year_r2[2024] + rows_by_year_r2[2025]
        df_r2 = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "delta_tail"} for r in all_rows_r2]
        )
        out_r2 = PHASE9_DIR / "results_round2.csv"
        df_r2.to_csv(out_r2, index=False)
        logger.info("Round 2 results saved to %s", out_r2)

        survivor_ids_r2 = prune_both_years(rows_by_year_r2[2024], rows_by_year_r2[2025])
        ranked_r2 = rank_survivors(rows_by_year_r2[2024], rows_by_year_r2[2025], survivor_ids_r2)

        logger.info("\nRound 2 SURVIVORS:")
        for r in ranked_r2[:20]:
            logger.info(
                "  %-45s iters_avg=%.0f mae_avg=%.2f", r["id"], r["n_iters_avg"], r["mae_avg"]
            )

        all_results[2] = rows_by_year_r2  # type: ignore[assignment]
        all_results["r2_survivors"] = survivor_ids_r2  # type: ignore[assignment]
        all_results["r2_rows"] = rows_by_year_r2  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Round 3
    # -----------------------------------------------------------------------
    if 3 in run_rounds:
        logger.info("=" * 60)
        logger.info("ROUND 3: fine-tuning")
        logger.info("=" * 60)

        # Load Round 2 survivors
        if "r2_survivors" in all_results:
            survivor_ids_r2 = all_results["r2_survivors"]  # type: ignore[index]
            rows_by_year_r2 = all_results["r2_rows"]  # type: ignore[index]
        else:
            r2_path = PHASE9_DIR / "results_round2.csv"
            if r2_path.exists():
                df_r2 = pd.read_csv(r2_path)
                ids_24 = set(df_r2[(df_r2.year == 2024) & df_r2.genuinely_monotone]["id"])
                ids_25 = set(df_r2[(df_r2.year == 2025) & df_r2.genuinely_monotone]["id"])
                survivor_ids_r2 = sorted(ids_24 & ids_25)
                rows_by_year_r2 = {
                    2024: df_r2[df_r2.year == 2024].to_dict("records"),
                    2025: df_r2[df_r2.year == 2025].to_dict("records"),
                }
                logger.info("Loaded %d Round 2 survivors from CSV", len(survivor_ids_r2))
            else:
                survivor_ids_r2 = []
                rows_by_year_r2 = {2024: [], 2025: []}
                logger.warning("No Round 2 results found; using empty survivor list")

        # Reconstruct survivor cfgs from the experiment registry
        all_r2_cfgs = _build_experiments_round2(survivor_ids_r2)
        survivor_cfgs = [c for c in all_r2_cfgs if c["id"] in survivor_ids_r2]

        experiments_r3 = _build_experiments_round3(survivor_ids_r2, survivor_cfgs)
        logger.info("  %d experiments × 2 years", len(experiments_r3))
        max_iters_r3 = max(args.max_iters, 500)

        rows_by_year_r3: dict[int, list[dict]] = {2024: [], 2025: []}
        for year in [2024, 2025]:
            d = data_by_year[year]
            t_start = time.perf_counter()
            for i, cfg in enumerate(experiments_r3, 1):
                t0 = time.perf_counter()
                row = run_experiment(
                    cfg=cfg,
                    strategy_forecasts=d["strategy_forecasts"],
                    real_prices=d["real_prices"],
                    initial_prices=d["initial_prices"],
                    max_iterations=max_iters_r3,
                )
                row["year"] = year
                rows_by_year_r3[year].append(row)
                elapsed = time.perf_counter() - t0
                flag = (
                    "✓MONO"
                    if row["genuinely_monotone"]
                    else ("✓conv" if row["converged"] else "     ")
                )
                logger.info(
                    "  R3 [%d/%d] %d %-45s %s iters=%d mae=%.1f spikes=%d %.2fs",
                    i,
                    len(experiments_r3),
                    year,
                    cfg["id"],
                    flag,
                    row["n_iterations"],
                    row["mae"],
                    row["n_spikes"],
                    elapsed,
                )
            logger.info("  Year %d done in %.1fs", year, time.perf_counter() - t_start)

        all_rows_r3 = rows_by_year_r3[2024] + rows_by_year_r3[2025]
        df_r3 = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "delta_tail"} for r in all_rows_r3]
        )
        out_r3 = PHASE9_DIR / "results_round3.csv"
        df_r3.to_csv(out_r3, index=False)
        logger.info("Round 3 results saved to %s", out_r3)

        survivor_ids_r3 = prune_both_years(rows_by_year_r3[2024], rows_by_year_r3[2025])
        ranked_r3 = rank_survivors(rows_by_year_r3[2024], rows_by_year_r3[2025], survivor_ids_r3)

        logger.info("\n" + "=" * 60)
        logger.info("FINAL WINNER CANDIDATES (monotone on both years):")
        logger.info("=" * 60)
        for r in ranked_r3[:10]:
            logger.info(
                "  %-50s iters_avg=%.0f mae_avg=%.3f", r["id"], r["n_iters_avg"], r["mae_avg"]
            )

        if ranked_r3:
            winner = ranked_r3[0]
            logger.info(
                "\nBEST WINNER: %s (iters_avg=%.0f, mae_avg=%.3f)",
                winner["id"],
                winner["n_iters_avg"],
                winner["mae_avg"],
            )

            # Print winner config
            all_r3_cfgs = {c["id"]: c for c in experiments_r3}
            w_cfg = all_r3_cfgs.get(winner["id"])
            if w_cfg:
                logger.info(
                    "Winner config: %s",
                    {k: v for k, v in w_cfg.items() if k not in ("id", "label")},
                )


if __name__ == "__main__":
    main()
