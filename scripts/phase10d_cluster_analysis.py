"""Phase 10d: Regime and Cluster Analysis.

Clusters strategies by forecast similarity and profit-response similarity,
identifies regime-dependent behaviour, and compares 2024 vs 2025 patterns.

Analyses:
  1. **Forecast-similarity clustering** — pairwise correlation of forecast
     time series, hierarchical clustering into families.
  2. **Profit-similarity clustering** — pairwise correlation of per-iteration
     profit vectors, compare to forecast clusters.
  3. **Cluster dominance by iteration** — which cluster dominates early vs late.
  4. **Regime analysis** — low-vol vs high-vol periods.
  5. **Cross-year comparison** — 2024 vs 2025 cluster structure.

Results are saved to ``data/results/phase10/strategy_clusters.csv``.

Usage::

    uv run python scripts/phase10d_cluster_analysis.py
    uv run python scripts/phase10d_cluster_analysis.py --year 2024
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE10_DIR = RESULTS_DIR / "phase10"

N_FORECAST_CLUSTERS = 8
N_PROFIT_CLUSTERS = 8


# ---------------------------------------------------------------------------
# Forecast clustering
# ---------------------------------------------------------------------------


def build_forecast_matrix(
    strategy_forecasts: dict[str, dict],
) -> pd.DataFrame:
    """Build a (strategies × dates) DataFrame from forecast dicts."""
    rows = {}
    for name, forecasts in strategy_forecasts.items():
        rows[name] = pd.Series(forecasts)
    return pd.DataFrame(rows).T.sort_index(axis=1)


def cluster_by_correlation(
    matrix: pd.DataFrame,
    n_clusters: int,
    method: str = "average",
) -> pd.Series:
    """Hierarchical clustering of rows by correlation distance.

    Returns a Series mapping row labels to cluster IDs.
    """
    if len(matrix) <= 1:
        return pd.Series(1, index=matrix.index, name="cluster")

    # Pairwise correlation → distance
    corr = matrix.T.corr()
    # Clip to ensure valid distance range [0, 2]
    dist = (1.0 - corr).clip(0.0, 2.0)
    np.fill_diagonal(dist.values, 0.0)

    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return pd.Series(labels, index=matrix.index, name="cluster")


# ---------------------------------------------------------------------------
# Profit clustering
# ---------------------------------------------------------------------------


def build_profit_matrix(
    equilibrium: FuturesMarketEquilibrium,
    strategy_names: list[str],
) -> pd.DataFrame:
    """Build a (strategies × iterations) profit DataFrame.

    Each cell is the total profit of strategy i at iteration k.
    """
    rows: dict[str, list[float]] = {name: [] for name in strategy_names}
    for it in equilibrium.iterations:
        for name in strategy_names:
            rows[name].append(it.strategy_profits.get(name, 0.0))
    return pd.DataFrame(rows, index=range(len(equilibrium.iterations))).T


def build_weight_matrix(
    equilibrium: FuturesMarketEquilibrium,
    strategy_names: list[str],
) -> pd.DataFrame:
    """Build a (strategies × iterations) weight DataFrame."""
    rows: dict[str, list[float]] = {name: [] for name in strategy_names}
    for it in equilibrium.iterations:
        for name in strategy_names:
            rows[name].append(it.strategy_weights.get(name, 0.0))
    return pd.DataFrame(rows, index=range(len(equilibrium.iterations))).T


# ---------------------------------------------------------------------------
# Cluster dominance analysis
# ---------------------------------------------------------------------------


def compute_cluster_dominance(
    weight_matrix: pd.DataFrame,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """Compute total weight per cluster per iteration.

    Returns a (clusters × iterations) DataFrame.
    """
    result = {}
    for cid in sorted(cluster_labels.unique()):
        members = cluster_labels[cluster_labels == cid].index
        members_in_matrix = [m for m in members if m in weight_matrix.index]
        if members_in_matrix:
            result[f"cluster_{cid}"] = weight_matrix.loc[members_in_matrix].sum(axis=0)
        else:
            result[f"cluster_{cid}"] = pd.Series(0.0, index=weight_matrix.columns)
    return pd.DataFrame(result).T


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------


def identify_volatility_regimes(
    real_prices: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Label each date as 'low_vol' or 'high_vol' based on rolling std.

    Returns a Series of regime labels indexed by date.
    """
    rolling_std = real_prices.rolling(window, min_periods=5).std()
    median_std = rolling_std.median()
    return pd.Series(
        np.where(rolling_std > median_std, "high_vol", "low_vol"),
        index=real_prices.index,
        name="regime",
    )


def compute_regime_forecast_stats(
    forecast_matrix: pd.DataFrame,
    real_prices: pd.Series,
    regimes: pd.Series,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """Compute per-cluster, per-regime forecast accuracy stats."""
    rows = []
    for regime in ["low_vol", "high_vol"]:
        regime_dates = regimes[regimes == regime].index
        regime_dates = [d for d in regime_dates if d in forecast_matrix.columns]
        if not regime_dates:
            continue

        real_sub = real_prices.reindex(regime_dates)

        for cid in sorted(cluster_labels.unique()):
            members = cluster_labels[cluster_labels == cid].index
            members_in_fm = [m for m in members if m in forecast_matrix.index]
            if not members_in_fm:
                continue

            cluster_forecasts = forecast_matrix.loc[members_in_fm, regime_dates]
            # Mean forecast of the cluster
            mean_forecast = cluster_forecasts.mean(axis=0)
            errors = mean_forecast - real_sub
            mae = float(errors.abs().mean())
            bias = float(errors.mean())
            rows.append(
                {
                    "regime": regime,
                    "cluster_id": cid,
                    "n_members": len(members_in_fm),
                    "n_dates": len(regime_dates),
                    "mae": round(mae, 4),
                    "bias": round(bias, 4),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyse_year(
    year: int,
    strategy_forecasts: dict[str, dict],
    real_prices: pd.Series,
    equilibrium: FuturesMarketEquilibrium,
) -> dict[str, pd.DataFrame]:
    """Run full cluster analysis for one year. Returns dict of DataFrames."""
    strategy_names = sorted(strategy_forecasts.keys())

    # 1. Forecast clustering
    fm = build_forecast_matrix(strategy_forecasts)
    forecast_clusters = cluster_by_correlation(fm, N_FORECAST_CLUSTERS)

    # 2. Profit clustering
    pm = build_profit_matrix(equilibrium, strategy_names)
    profit_clusters = cluster_by_correlation(pm, N_PROFIT_CLUSTERS)

    # 3. Weight matrix and cluster dominance
    wm = build_weight_matrix(equilibrium, strategy_names)
    forecast_dominance = compute_cluster_dominance(wm, forecast_clusters)
    profit_dominance = compute_cluster_dominance(wm, profit_clusters)

    # 4. Regime analysis
    regimes = identify_volatility_regimes(real_prices)
    regime_stats = compute_regime_forecast_stats(fm, real_prices, regimes, forecast_clusters)

    # 5. Build strategy cluster assignment table
    cluster_assignments = pd.DataFrame(
        {
            "strategy": strategy_names,
            "forecast_cluster": [forecast_clusters.get(s, -1) for s in strategy_names],
            "profit_cluster": [profit_clusters.get(s, -1) for s in strategy_names],
            "year": year,
        }
    )

    # 6. Cluster agreement: how much do forecast and profit clusters agree?
    agreement = _compute_cluster_agreement(forecast_clusters, profit_clusters)

    # 7. Dominance summary per cluster
    dominance_summary = _compute_dominance_summary(
        forecast_dominance,
        profit_dominance,
        forecast_clusters,
        profit_clusters,
        year,
    )

    return {
        "cluster_assignments": cluster_assignments,
        "forecast_dominance": forecast_dominance,
        "profit_dominance": profit_dominance,
        "regime_stats": regime_stats,
        "agreement": agreement,
        "dominance_summary": dominance_summary,
    }


def _compute_cluster_agreement(
    forecast_clusters: pd.Series,
    profit_clusters: pd.Series,
) -> pd.DataFrame:
    """Cross-tabulation of forecast vs profit cluster membership."""
    common = forecast_clusters.index.intersection(profit_clusters.index)
    fc = forecast_clusters.reindex(common)
    pc = profit_clusters.reindex(common)
    return pd.crosstab(fc, pc, rownames=["forecast_cluster"], colnames=["profit_cluster"])


def _compute_dominance_summary(
    forecast_dominance: pd.DataFrame,
    profit_dominance: pd.DataFrame,
    forecast_clusters: pd.Series,
    profit_clusters: pd.Series,
    year: int,
) -> pd.DataFrame:
    """Summary of when each cluster dominates (early vs late iterations)."""
    rows = []
    n_iters = forecast_dominance.shape[1]
    early_end = min(50, n_iters // 4)
    late_start = max(n_iters - 50, n_iters * 3 // 4)

    for cid in forecast_dominance.index:
        early_weight = float(forecast_dominance.loc[cid, :early_end].mean())
        late_weight = float(forecast_dominance.loc[cid, late_start:].mean())
        n_members = int((forecast_clusters == int(cid.split("_")[1])).sum())
        rows.append(
            {
                "year": year,
                "cluster_type": "forecast",
                "cluster_id": cid,
                "n_members": n_members,
                "early_avg_weight": round(early_weight, 4),
                "late_avg_weight": round(late_weight, 4),
                "weight_shift": round(late_weight - early_weight, 4),
            }
        )

    for cid in profit_dominance.index:
        early_weight = float(profit_dominance.loc[cid, :early_end].mean())
        late_weight = float(profit_dominance.loc[cid, late_start:].mean())
        n_members = int((profit_clusters == int(cid.split("_")[1])).sum())
        rows.append(
            {
                "year": year,
                "cluster_type": "profit",
                "cluster_id": cid,
                "n_members": n_members,
                "early_avg_weight": round(early_weight, 4),
                "late_avg_weight": round(late_weight, 4),
                "weight_shift": round(late_weight - early_weight, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    results: dict[int, dict[str, pd.DataFrame]],
) -> None:
    """Print a human-readable summary."""
    for year, data in sorted(results.items()):
        print(f"\n{'=' * 80}")
        print(f"  YEAR {year}: CLUSTER ANALYSIS")
        print(f"{'=' * 80}")

        # Cluster assignments
        ca = data["cluster_assignments"]
        print(f"\n  Strategy Cluster Assignments ({len(ca)} strategies):")
        for fc_id in sorted(ca["forecast_cluster"].unique()):
            members = ca[ca["forecast_cluster"] == fc_id]["strategy"].tolist()
            print(f"    Forecast cluster {fc_id} ({len(members)} members):")
            for m in sorted(members):
                pc = ca[ca["strategy"] == m]["profit_cluster"].iloc[0]
                print(f"      {m} (profit cluster {pc})")

        # Agreement
        print("\n  Forecast vs Profit Cluster Agreement:")
        print(data["agreement"].to_string())

        # Dominance summary
        ds = data["dominance_summary"]
        fds = ds[ds["cluster_type"] == "forecast"].sort_values("late_avg_weight", ascending=False)
        print("\n  Forecast Cluster Dominance (early -> late):")
        print(f"  {'Cluster':<15} {'Members':>8} {'Early wt':>10} {'Late wt':>10} {'Shift':>10}")
        for _, r in fds.iterrows():
            print(
                f"  {r['cluster_id']:<15} {r['n_members']:>8} "
                f"{r['early_avg_weight']:>10.4f} "
                f"{r['late_avg_weight']:>10.4f} "
                f"{r['weight_shift']:>+10.4f}"
            )

        # Regime stats
        rs = data["regime_stats"]
        if not rs.empty:
            print("\n  Regime-Dependent Forecast Accuracy:")
            print(
                f"  {'Regime':<10} {'Cluster':>8} "
                f"{'Members':>8} {'Dates':>6} {'MAE':>8} {'Bias':>8}"
            )
            for _, r in rs.iterrows():
                print(
                    f"  {r['regime']:<10} {r['cluster_id']:>8} "
                    f"{r['n_members']:>8} {r['n_dates']:>6} "
                    f"{r['mae']:>8.2f} {r['bias']:>+8.2f}"
                )

    # Cross-year comparison
    if len(results) == 2:
        print(f"\n{'=' * 80}")
        print("  CROSS-YEAR COMPARISON")
        print(f"{'=' * 80}")
        for year, data in sorted(results.items()):
            ca = data["cluster_assignments"]
            ds = data["dominance_summary"]
            fds = ds[ds["cluster_type"] == "forecast"]
            top_cluster = fds.sort_values("late_avg_weight", ascending=False).iloc[0]
            print(
                f"\n  {year}: dominant late cluster = "
                f"{top_cluster['cluster_id']} "
                f"(weight={top_cluster['late_avg_weight']:.4f}, "
                f"{top_cluster['n_members']} members)"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10d: regime and cluster analysis.")
    parser.add_argument(
        "--year",
        type=int,
        choices=[2024, 2025],
        default=None,
        help="Run only one year (default: both).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    PHASE10_DIR.mkdir(parents=True, exist_ok=True)
    years = [args.year] if args.year else [2024, 2025]

    results: dict[int, dict[str, pd.DataFrame]] = {}
    all_assignments: list[pd.DataFrame] = []

    for year in years:
        # Load forecasts
        fpath = PHASE8_DIR / f"forecasts_{year}.pkl"
        if not fpath.exists():
            logger.error("Missing %s", fpath)
            sys.exit(1)
        with open(fpath, "rb") as f:
            data = pickle.load(f)

        strategy_forecasts: dict[str, dict] = data["strategy_forecasts"]
        real_prices: pd.Series = data["real_prices"]

        # Load market result
        mpath = RESULTS_DIR / f"market_{year}.pkl"
        if not mpath.exists():
            logger.error("Missing %s", mpath)
            sys.exit(1)
        with open(mpath, "rb") as f:
            market_result = pickle.load(f)

        equilibrium = market_result.equilibrium

        logger.info(
            "Year %d: %d strategies, %d dates, %d iterations",
            year,
            len(strategy_forecasts),
            len(real_prices),
            len(equilibrium.iterations),
        )

        year_results = analyse_year(year, strategy_forecasts, real_prices, equilibrium)
        results[year] = year_results
        all_assignments.append(year_results["cluster_assignments"])

    # Save cluster assignments
    combined = pd.concat(all_assignments, ignore_index=True)
    out_path = PHASE10_DIR / "strategy_clusters.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Cluster assignments saved to %s", out_path)

    # Save dominance summaries
    all_dominance = pd.concat(
        [r["dominance_summary"] for r in results.values()],
        ignore_index=True,
    )
    dom_path = PHASE10_DIR / "cluster_dominance.csv"
    all_dominance.to_csv(dom_path, index=False)
    logger.info("Cluster dominance saved to %s", dom_path)

    # Save regime stats
    all_regime = pd.concat(
        [r["regime_stats"] for r in results.values()],
        ignore_index=True,
    )
    if not all_regime.empty:
        regime_path = PHASE10_DIR / "regime_forecast_accuracy.csv"
        all_regime.to_csv(regime_path, index=False)
        logger.info("Regime stats saved to %s", regime_path)

    print_summary(results)


if __name__ == "__main__":
    main()
