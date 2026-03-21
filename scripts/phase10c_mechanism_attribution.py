"""Phase 10c: Mechanism Attribution — ablation and sensitivity experiments.

Determines which parts of the market mechanism are responsible for each
observed behaviour:
  - sign-based trading rule
  - total-profit scoring across the full window
  - positive-profit truncation
  - linear weight normalisation
  - weighted-average forecast aggregation
  - EMA dampening via ``ema_alpha``
  - initialisation from ``last_settlement_price``

Experiments:
  1. **EMA-alpha sweep** — {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
  2. **Initialisation counterfactuals** — default / forecast-mean / constant / random
  3. **Strategy-family ablations** — remove one family at a time and measure change
  4. **Keep-only ablations** — keep only ML / only rule-based / only ensembles

Results are saved to ``data/results/phase10/mechanism_attribution.csv``.

Usage::

    uv run python scripts/phase10c_mechanism_attribution.py
    uv run python scripts/phase10c_mechanism_attribution.py --year 2024
    uv run python scripts/phase10c_mechanism_attribution.py --max-iters 500
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

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
    FuturesMarketIteration,
    _build_forecast_matrix,
    _vec_iteration,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE10_DIR = RESULTS_DIR / "phase10"

CONVERGENCE_THRESHOLD = 0.01
DEFAULT_EMA_ALPHA = 0.1
DEFAULT_MAX_ITERS = 500


# ---------------------------------------------------------------------------
# Strategy-family definitions
# ---------------------------------------------------------------------------

ABLATION_GROUPS: dict[str, list[str]] = {
    "naive_baselines": [
        "Always Long",
        "Always Short",
    ],
    "calendar_temporal": [
        "Day Of Week",
        "Dow Composite",
        "Month Seasonal",
        "Monday Effect",
        "Quarter Seasonal",
        "Lasso Calendar Augmented",
    ],
    "mean_reversion": [
        "Lag2 Reversion",
        "Weekly Cycle",
        "Price ZScore Reversion",
        "Price Min Reversion",
        "Composite Signal",
    ],
    "momentum_trend": [
        "Gas Trend",
        "Carbon Trend",
        "Fuel Index Trend",
        "ZScore Momentum",
        "Gas Carbon Joint Trend",
    ],
    "supply_side": [
        "Wind Forecast",
        "Fossil Dispatch",
        "Solar Forecast",
        "Nuclear Availability",
        "Wind Forecast Error",
    ],
    "demand_side": [
        "Load Forecast",
        "Temperature Extreme",
        "Net Demand",
        "Load Surprise",
        "Net Demand Momentum",
    ],
    "commodity_cost": [
        "Commodity Cost",
    ],
    "renewables_regime": [
        "Renewables Surplus",
        "Renewables Penetration",
        "Renewable Regime",
        "Volatility Regime",
    ],
    "cross_border_spread": [
        "Cross Border Spread",
        "DEFR Spread",
        "DENL Spread",
        "Multi Spread",
        "NLFlow Signal",
        "FRFlow Signal",
    ],
    "ml_regression": [
        "Lasso Regression",
        "Ridge Regression",
        "Elastic Net",
        "Lasso Top Features",
        "Ridge Net Demand",
        "Lasso Calendar Augmented",
        "GBMNet Demand",
        "Bayesian Ridge",
        "PLSRegression",
        "Neural Net",
        "Volatility Regime ML",
    ],
    "ml_classification": [
        "Logistic Direction",
        "Random Forest",
        "Gradient Boosting",
        "KNNDirection",
        "SVMDirection",
        "Decision Tree",
    ],
    "ensemble_meta": [
        "Consensus Signal",
        "Majority Vote Rule Based",
        "Majority Vote ML",
        "Mean Forecast Regression",
        "Median Forecast Ensemble",
        "Top KEnsemble",
        "Weighted Vote Mixed",
        "Diversity Ensemble",
        "Regime Conditional Ensemble",
        "Stacked Ridge Meta",
        "Weekday Weekend Ensemble",
        "Boosted Spread ML",
    ],
}

# Compound groups for broader ablations
ABLATION_GROUPS["all_ml"] = ABLATION_GROUPS["ml_regression"] + ABLATION_GROUPS["ml_classification"]
ABLATION_GROUPS["all_rule_based"] = (
    ABLATION_GROUPS["naive_baselines"]
    + ABLATION_GROUPS["calendar_temporal"]
    + ABLATION_GROUPS["mean_reversion"]
    + ABLATION_GROUPS["momentum_trend"]
    + ABLATION_GROUPS["supply_side"]
    + ABLATION_GROUPS["demand_side"]
    + ABLATION_GROUPS["commodity_cost"]
    + ABLATION_GROUPS["renewables_regime"]
    + ABLATION_GROUPS["cross_border_spread"]
)


# ---------------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------------


def run_market(
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    initial_prices: pd.Series,
    max_iterations: int = DEFAULT_MAX_ITERS,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
) -> dict:
    """Run one market simulation and return a compact metrics dict.

    Uses the vectorised ``_vec_iteration`` path for speed.  The EMA blend
    is applied after the spec step-4 output.
    """
    current_prices = initial_prices.copy().astype(float)
    index = current_prices.index

    fm = _build_forecast_matrix(strategy_forecasts, index, real_prices)
    strategy_names = fm.strategy_names

    iterations: list[FuturesMarketIteration] = []
    converged = False
    delta = float("inf")
    n_strategies = len(strategy_names)

    for k in range(max_iterations):
        market_vec = current_prices.to_numpy(dtype=np.float64)
        new_vec, profits_arr, weights_arr = _vec_iteration(market_vec, fm)

        blended_vec = (
            ema_alpha * new_vec + (1.0 - ema_alpha) * market_vec if ema_alpha < 1.0 else new_vec
        )

        published = pd.Series(blended_vec, index=index, name="market_price")
        profits_dict = dict(zip(strategy_names, profits_arr.tolist(), strict=True))
        weights_dict = dict(zip(strategy_names, weights_arr.tolist(), strict=True))
        active = [n for n, w in zip(strategy_names, weights_arr, strict=True) if w > 0.0]

        iterations.append(
            FuturesMarketIteration(
                iteration=k,
                market_prices=published,
                strategy_profits=profits_dict,
                strategy_weights=weights_dict,
                active_strategies=active,
            )
        )

        delta = float(np.abs(blended_vec - market_vec).max())
        current_prices = published

        if delta < CONVERGENCE_THRESHOLD:
            converged = True
            break

    eq = FuturesMarketEquilibrium(
        iterations=iterations,
        final_market_prices=current_prices,
        final_weights=iterations[-1].strategy_weights if iterations else {},
        converged=converged,
        convergence_delta=delta,
    )

    # Metrics
    aligned_real = real_prices.reindex(eq.final_market_prices.index)
    errors = eq.final_market_prices - aligned_real
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt((errors**2).mean()))
    bias = float(errors.mean())

    last_it = iterations[-1] if iterations else None
    active_count = len(last_it.active_strategies) if last_it else 0
    weights_arr_final = (
        np.array(list(last_it.strategy_weights.values())) if last_it else np.array([])
    )
    pos_w = weights_arr_final[weights_arr_final > 0]
    entropy = float(-np.sum(pos_w * np.log(pos_w))) if len(pos_w) > 0 else 0.0
    top1 = float(pos_w.max()) if len(pos_w) > 0 else 0.0

    return {
        "converged": converged,
        "n_iterations": len(iterations),
        "final_delta": round(delta, 6),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "n_strategies": n_strategies,
        "final_active_count": active_count,
        "final_weight_entropy": round(entropy, 4),
        "final_top1_weight": round(top1, 4),
    }


# ---------------------------------------------------------------------------
# Experiment builders
# ---------------------------------------------------------------------------


def build_ema_sweep_experiments() -> list[dict]:
    """EMA-alpha sensitivity sweep."""
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    return [
        {
            "experiment_type": "ema_sweep",
            "experiment_id": f"ema_alpha_{a:.2f}",
            "label": f"EMA alpha={a}",
            "ema_alpha": a,
            "init_mode": "default",
            "ablation": "none",
        }
        for a in alphas
    ]


def build_init_experiments() -> list[dict]:
    """Initialisation counterfactual experiments."""
    return [
        {
            "experiment_type": "init_sensitivity",
            "experiment_id": "init_default",
            "label": "Init: last settlement (default)",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "default",
            "ablation": "none",
        },
        {
            "experiment_type": "init_sensitivity",
            "experiment_id": "init_forecast_mean",
            "label": "Init: mean of all strategy forecasts",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "forecast_mean",
            "ablation": "none",
        },
        {
            "experiment_type": "init_sensitivity",
            "experiment_id": "init_constant_50",
            "label": "Init: constant 50 EUR/MWh",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "constant_50",
            "ablation": "none",
        },
        {
            "experiment_type": "init_sensitivity",
            "experiment_id": "init_real_prices",
            "label": "Init: real prices (oracle)",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "real_prices",
            "ablation": "none",
        },
    ]


def build_ablation_experiments() -> list[dict]:
    """Strategy-family ablation experiments (remove one family at a time)."""
    # The 12 base family ablations
    family_names = [
        "naive_baselines",
        "calendar_temporal",
        "mean_reversion",
        "momentum_trend",
        "supply_side",
        "demand_side",
        "commodity_cost",
        "renewables_regime",
        "cross_border_spread",
        "ml_regression",
        "ml_classification",
        "ensemble_meta",
    ]

    experiments = []

    # Baseline: all strategies (no ablation)
    experiments.append(
        {
            "experiment_type": "ablation",
            "experiment_id": "ablation_baseline",
            "label": "Baseline: all 67 strategies",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "default",
            "ablation": "none",
        }
    )

    # Remove one family at a time
    for family in family_names:
        n = len(set(ABLATION_GROUPS[family]))
        experiments.append(
            {
                "experiment_type": "ablation",
                "experiment_id": f"remove_{family}",
                "label": f"Remove {family} ({n} strategies)",
                "ema_alpha": DEFAULT_EMA_ALPHA,
                "init_mode": "default",
                "ablation": f"remove_{family}",
            }
        )

    # Broad structural ablations
    experiments.append(
        {
            "experiment_type": "ablation",
            "experiment_id": "remove_all_ml",
            "label": "Remove all ML (regression + classification)",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "default",
            "ablation": "remove_all_ml",
        }
    )
    experiments.append(
        {
            "experiment_type": "ablation",
            "experiment_id": "keep_only_ml",
            "label": "Keep only ML strategies",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "default",
            "ablation": "keep_only_ml",
        }
    )
    experiments.append(
        {
            "experiment_type": "ablation",
            "experiment_id": "keep_only_rule_based",
            "label": "Keep only rule-based strategies",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "default",
            "ablation": "keep_only_rule_based",
        }
    )
    experiments.append(
        {
            "experiment_type": "ablation",
            "experiment_id": "keep_only_ensemble",
            "label": "Keep only ensemble strategies",
            "ema_alpha": DEFAULT_EMA_ALPHA,
            "init_mode": "default",
            "ablation": "keep_only_ensemble",
        }
    )

    return experiments


# ---------------------------------------------------------------------------
# Forecast filtering helpers
# ---------------------------------------------------------------------------


def _apply_ablation(
    strategy_forecasts: dict[str, dict],
    ablation: str,
) -> dict[str, dict]:
    """Return a filtered copy of strategy_forecasts per the ablation spec."""
    if ablation == "none":
        return strategy_forecasts

    if ablation.startswith("remove_"):
        group_key = ablation[len("remove_") :]
        if group_key in ABLATION_GROUPS:
            to_remove = set(ABLATION_GROUPS[group_key])
            return {k: v for k, v in strategy_forecasts.items() if k not in to_remove}
        raise ValueError(f"Unknown ablation group: {group_key}")

    if ablation.startswith("keep_only_"):
        group_key = ablation[len("keep_only_") :]
        if group_key in ABLATION_GROUPS:
            to_keep = set(ABLATION_GROUPS[group_key])
            return {k: v for k, v in strategy_forecasts.items() if k in to_keep}
        # Compound keep rules
        if group_key == "ml":
            to_keep = set(ABLATION_GROUPS["all_ml"])
            return {k: v for k, v in strategy_forecasts.items() if k in to_keep}
        if group_key == "rule_based":
            to_keep = set(ABLATION_GROUPS["all_rule_based"])
            return {k: v for k, v in strategy_forecasts.items() if k in to_keep}
        if group_key == "ensemble":
            to_keep = set(ABLATION_GROUPS["ensemble_meta"])
            return {k: v for k, v in strategy_forecasts.items() if k in to_keep}
        raise ValueError(f"Unknown keep_only group: {group_key}")

    raise ValueError(f"Unknown ablation: {ablation}")


def _apply_init_mode(
    init_mode: str,
    initial_prices: pd.Series,
    real_prices: pd.Series,
    strategy_forecasts: dict[str, dict],
) -> pd.Series:
    """Return modified initial prices per the init_mode spec."""
    if init_mode == "default":
        return initial_prices

    if init_mode == "forecast_mean":
        # Mean of all strategy forecasts per date
        all_forecasts: dict[object, list[float]] = {}
        for forecasts in strategy_forecasts.values():
            for dt, val in forecasts.items():
                all_forecasts.setdefault(dt, []).append(val)
        mean_prices = pd.Series({dt: np.mean(vals) for dt, vals in all_forecasts.items()})
        return mean_prices.reindex(initial_prices.index).fillna(initial_prices)

    if init_mode == "constant_50":
        return pd.Series(50.0, index=initial_prices.index)

    if init_mode == "real_prices":
        return real_prices.reindex(initial_prices.index).fillna(initial_prices)

    raise ValueError(f"Unknown init_mode: {init_mode}")


# ---------------------------------------------------------------------------
# Classification of convergence pattern
# ---------------------------------------------------------------------------


def classify_outcome(metrics: dict) -> str:
    """Classify the convergence outcome of a single experiment run."""
    if metrics["n_strategies"] == 0:
        return "no_strategies"
    if not metrics["converged"]:
        return "non_converged"
    if metrics["final_active_count"] == 0:
        return "absorbing_collapse"
    if metrics["final_active_count"] <= 2:
        return "near_collapse"
    return "healthy_convergence"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_all_experiments(
    years: list[int],
    max_iterations: int,
) -> pd.DataFrame:
    """Run all mechanism-attribution experiments and return a results DataFrame."""
    experiments = (
        build_ema_sweep_experiments() + build_init_experiments() + build_ablation_experiments()
    )

    logger.info(
        "Phase 10c: %d experiments × %d year(s) = %d total runs",
        len(experiments),
        len(years),
        len(experiments) * len(years),
    )

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
            "Year %d: %d strategies, %d eval days",
            year,
            len(strategy_forecasts),
            len(real_prices),
        )

        for i, cfg in enumerate(experiments, 1):
            t0 = time.perf_counter()
            try:
                # Apply ablation (filter strategies)
                filtered = _apply_ablation(strategy_forecasts, cfg["ablation"])

                if len(filtered) == 0:
                    logger.warning(
                        "[%d/%d] %s (%d) — no strategies after ablation, skipping",
                        i,
                        len(experiments),
                        cfg["experiment_id"],
                        year,
                    )
                    row = {
                        "experiment_type": cfg["experiment_type"],
                        "experiment_id": cfg["experiment_id"],
                        "label": cfg["label"],
                        "ema_alpha": cfg["ema_alpha"],
                        "init_mode": cfg["init_mode"],
                        "ablation": cfg["ablation"],
                        "year": year,
                        "converged": False,
                        "n_iterations": 0,
                        "final_delta": float("inf"),
                        "mae": float("inf"),
                        "rmse": float("inf"),
                        "bias": float("inf"),
                        "n_strategies": 0,
                        "final_active_count": 0,
                        "final_weight_entropy": 0.0,
                        "final_top1_weight": 0.0,
                        "outcome_class": "no_strategies",
                        "elapsed_s": 0.0,
                    }
                    all_rows.append(row)
                    continue

                # Apply init mode
                init_p = _apply_init_mode(cfg["init_mode"], initial_prices, real_prices, filtered)

                metrics = run_market(
                    strategy_forecasts=filtered,
                    real_prices=real_prices,
                    initial_prices=init_p,
                    max_iterations=max_iterations,
                    ema_alpha=cfg["ema_alpha"],
                )

                row = {
                    "experiment_type": cfg["experiment_type"],
                    "experiment_id": cfg["experiment_id"],
                    "label": cfg["label"],
                    "ema_alpha": cfg["ema_alpha"],
                    "init_mode": cfg["init_mode"],
                    "ablation": cfg["ablation"],
                    "year": year,
                    **metrics,
                    "outcome_class": classify_outcome(metrics),
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                }
                all_rows.append(row)

                status = "CONV" if metrics["converged"] else "----"
                logger.info(
                    "[%d/%d] %s (%d) %s  n=%d  iters=%d  delta=%.4f  MAE=%.2f  active=%d",
                    i,
                    len(experiments),
                    cfg["experiment_id"],
                    year,
                    status,
                    metrics["n_strategies"],
                    metrics["n_iterations"],
                    metrics["final_delta"],
                    metrics["mae"],
                    metrics["final_active_count"],
                )

            except Exception:
                logger.exception(
                    "Experiment %s year %d failed",
                    cfg["experiment_id"],
                    year,
                )
                row = {
                    "experiment_type": cfg["experiment_type"],
                    "experiment_id": cfg["experiment_id"],
                    "label": cfg["label"],
                    "ema_alpha": cfg["ema_alpha"],
                    "init_mode": cfg["init_mode"],
                    "ablation": cfg["ablation"],
                    "year": year,
                    "converged": False,
                    "n_iterations": 0,
                    "final_delta": float("inf"),
                    "mae": float("inf"),
                    "rmse": float("inf"),
                    "bias": float("inf"),
                    "n_strategies": 0,
                    "final_active_count": 0,
                    "final_weight_entropy": 0.0,
                    "final_top1_weight": 0.0,
                    "outcome_class": "error",
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "error": "FAILED",
                }
                all_rows.append(row)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_section(title: str, df: pd.DataFrame) -> None:
    """Print a formatted results section."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    if df.empty:
        print("  (no results)")
        return

    cols = [
        "experiment_id",
        "year",
        "n_strategies",
        "converged",
        "n_iterations",
        "final_delta",
        "mae",
        "final_active_count",
        "outcome_class",
    ]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of all experiments."""
    # EMA sweep
    ema_df = df[df["experiment_type"] == "ema_sweep"]
    _print_section("EMA ALPHA SWEEP", ema_df)

    # Init sensitivity
    init_df = df[df["experiment_type"] == "init_sensitivity"]
    _print_section("INITIALISATION SENSITIVITY", init_df)

    # Ablation
    abl_df = df[df["experiment_type"] == "ablation"]
    _print_section("STRATEGY-FAMILY ABLATION", abl_df)

    # Attribution summary: which mechanism changes the convergence class?
    print(f"\n{'=' * 90}")
    print("  MECHANISM ATTRIBUTION SUMMARY")
    print(f"{'=' * 90}")

    baseline = df[df["experiment_id"] == "ablation_baseline"]
    for year in sorted(df["year"].unique()):
        bl = baseline[baseline["year"] == year]
        if bl.empty:
            continue
        bl_class = bl.iloc[0]["outcome_class"]
        bl_mae = bl.iloc[0]["mae"]
        print(f"\n  Year {year} — baseline: {bl_class} (MAE={bl_mae:.2f})")
        print(f"  {'Experiment':<35} {'Outcome':<25} {'MAE':>8} {'dMAE':>8} {'Class changed?':<15}")
        print(f"  {'-' * 35} {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 15}")

        year_df = df[df["year"] == year]
        for _, row in year_df.iterrows():
            if row["experiment_id"] == "ablation_baseline":
                continue
            changed = "YES" if row["outcome_class"] != bl_class else "no"
            d_mae = row["mae"] - bl_mae if row["mae"] != float("inf") else float("inf")
            print(
                f"  {row['experiment_id']:<35} "
                f"{row['outcome_class']:<25} "
                f"{row['mae']:>8.2f} "
                f"{d_mae:>+8.2f} "
                f"{changed:<15}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10c: mechanism attribution experiments.")
    parser.add_argument(
        "--year",
        type=int,
        choices=[2024, 2025],
        default=None,
        help="Run only one year (default: both).",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=DEFAULT_MAX_ITERS,
        help=f"Max iterations per experiment (default: {DEFAULT_MAX_ITERS}).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    PHASE10_DIR.mkdir(parents=True, exist_ok=True)
    years = [args.year] if args.year else [2024, 2025]

    df = run_all_experiments(years=years, max_iterations=args.max_iters)

    # Save
    out_path = PHASE10_DIR / "mechanism_attribution.csv"
    df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    print_summary(df)


if __name__ == "__main__":
    main()
