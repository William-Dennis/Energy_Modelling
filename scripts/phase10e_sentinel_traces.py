"""Phase 10e: Sentinel Case Studies -- iteration-level market traces.

Selects a small set of high-information windows and days from the 2024 and 2025
market simulations, then builds detailed iteration-by-iteration traces showing
how prices, weights, active strategies, and cluster dominance evolve.

Sentinel case types:
  1. **High-volatility non-convergence** (2024): a window where the real price
     is most volatile and the market fails to track it.
  2. **Zero-active convergence** (2025): the window around the iteration where
     all active strategies collapse to zero.
  3. **Early-accuracy-lost**: a specific date where iteration-0 MAE is lower
     than the final-iteration MAE.
  4. **Cluster-switching episode**: a window where the dominant cluster shifts.

Outputs
-------
- ``data/results/phase10/sentinel_traces/`` — per-case CSV traces
- ``data/results/phase10/sentinel_summaries.csv`` — one-row-per-case summary
- Plain-language narrative printed to stdout

Usage::

    uv run python scripts/phase10e_sentinel_traces.py
    uv run python scripts/phase10e_sentinel_traces.py --year 2024
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from energy_modelling.backtest.futures_market_engine import (  # noqa: E402
    FuturesMarketEquilibrium,
)
from energy_modelling.backtest.futures_market_runner import FuturesMarketResult  # noqa: E402

logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "data" / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
PHASE10_DIR = RESULTS_DIR / "phase10"
TRACE_DIR = PHASE10_DIR / "sentinel_traces"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SentinelCase:
    """Description of a sentinel case to trace."""

    case_id: str
    case_type: str
    year: int
    description: str
    # Iteration window to trace (start, end inclusive)
    iter_start: int
    iter_end: int
    # Optional: specific dates to focus on (None = all dates)
    focus_dates: list | None = None


@dataclass
class TraceRow:
    """One row of an iteration trace for a sentinel case."""

    case_id: str
    iteration: int
    convergence_delta: float
    mae: float
    rmse: float
    bias: float
    active_count: int
    weight_entropy: float
    top1_strategy: str
    top1_weight: float
    top3_strategies: str  # comma-separated
    top3_weights: str  # comma-separated
    dominant_cluster: str  # from forecast clusters if available
    cluster_weight: float


# ---------------------------------------------------------------------------
# Sentinel window selection
# ---------------------------------------------------------------------------


def find_high_volatility_window(
    real_prices: pd.Series,
    window_size: int = 20,
) -> tuple[int, int]:
    """Find the rolling window with the highest standard deviation.

    Returns (start_index, end_index) as integer positions into real_prices.
    """
    rolling_std = real_prices.rolling(window_size, min_periods=window_size).std()
    if rolling_std.dropna().empty:
        return 0, min(window_size, len(real_prices)) - 1
    peak_idx = rolling_std.idxmax()
    peak_pos = real_prices.index.get_loc(peak_idx)
    start_pos = max(0, peak_pos - window_size + 1)
    end_pos = peak_pos
    return int(start_pos), int(end_pos)


def find_active_collapse_window(
    equilibrium: FuturesMarketEquilibrium,
    margin: int = 25,
) -> tuple[int, int]:
    """Find the iteration window around the active-strategy collapse.

    Returns (iter_start, iter_end).
    """
    for it in equilibrium.iterations:
        if len(it.active_strategies) == 0:
            collapse_iter = it.iteration
            start = max(0, collapse_iter - margin)
            end = min(len(equilibrium.iterations) - 1, collapse_iter + margin)
            return start, end

    # No collapse found -- use last 50 iterations
    n = len(equilibrium.iterations)
    return max(0, n - 50), n - 1


def find_early_accuracy_dates(
    equilibrium: FuturesMarketEquilibrium,
    real_prices: pd.Series,
    n_dates: int = 5,
) -> list:
    """Find dates where iteration-0 has lower absolute error than final iteration.

    Returns up to ``n_dates`` dates sorted by the largest accuracy degradation.
    """
    if len(equilibrium.iterations) < 2:
        return []

    iter0 = equilibrium.iterations[0]
    final = equilibrium.iterations[-1]

    iter0_prices = iter0.market_prices
    final_prices = final.market_prices

    common_idx = iter0_prices.index.intersection(final_prices.index).intersection(real_prices.index)
    if common_idx.empty:
        return []

    iter0_ae = (iter0_prices.reindex(common_idx) - real_prices.reindex(common_idx)).abs()
    final_ae = (final_prices.reindex(common_idx) - real_prices.reindex(common_idx)).abs()

    degradation = final_ae - iter0_ae  # positive = final is worse
    worse_dates = degradation[degradation > 0].sort_values(ascending=False)

    return list(worse_dates.index[:n_dates])


def find_cluster_switching_window(
    equilibrium: FuturesMarketEquilibrium,
    strategy_clusters: dict[str, int] | None = None,
    window_size: int = 30,
) -> tuple[int, int]:
    """Find the iteration window with the most cluster dominance shifts.

    If ``strategy_clusters`` is None, assigns clusters based on weight rank.
    Returns (iter_start, iter_end).
    """
    iterations = equilibrium.iterations
    if len(iterations) < 2:
        return 0, 0

    # Compute the top-1 strategy at each iteration
    top1_series = []
    for it in iterations:
        if it.strategy_weights:
            top_name = max(it.strategy_weights, key=it.strategy_weights.get)
            if strategy_clusters and top_name in strategy_clusters:
                top1_series.append(strategy_clusters[top_name])
            else:
                top1_series.append(top_name)
        else:
            top1_series.append(None)

    # Count leadership changes in each window
    n = len(top1_series)
    best_start = 0
    best_changes = 0

    for start in range(max(1, n - window_size)):
        end = min(start + window_size, n)
        changes = sum(
            1
            for i in range(start + 1, end)
            if top1_series[i] != top1_series[i - 1] and top1_series[i] is not None
        )
        if changes > best_changes:
            best_changes = changes
            best_start = start

    return best_start, min(best_start + window_size - 1, n - 1)


# ---------------------------------------------------------------------------
# Trace building
# ---------------------------------------------------------------------------


def build_iteration_trace(
    case: SentinelCase,
    equilibrium: FuturesMarketEquilibrium,
    real_prices: pd.Series,
    strategy_clusters: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build a detailed per-iteration trace for a sentinel case.

    Parameters
    ----------
    case:
        The sentinel case definition.
    equilibrium:
        The market equilibrium result.
    real_prices:
        Ground-truth prices.
    strategy_clusters:
        Optional mapping from strategy name to cluster ID.

    Returns
    -------
    DataFrame with one row per iteration in the window.
    """
    rows: list[dict] = []
    iters = equilibrium.iterations

    for idx in range(case.iter_start, min(case.iter_end + 1, len(iters))):
        it = iters[idx]
        prev = iters[idx - 1] if idx > 0 else None

        # Price metrics
        market = it.market_prices
        if case.focus_dates:
            focus_idx = pd.Index(case.focus_dates).intersection(market.index)
            if not focus_idx.empty:
                market = market.reindex(focus_idx)
                aligned_real = real_prices.reindex(focus_idx)
            else:
                aligned_real = real_prices.reindex(market.index)
        else:
            aligned_real = real_prices.reindex(market.index)

        residuals = market - aligned_real

        # Convergence delta
        if prev is not None:
            prev_market = prev.market_prices.reindex(market.index)
            delta = float((market - prev_market).abs().max())
        else:
            delta = float("inf")

        mae = float(residuals.abs().mean())
        rmse = float(np.sqrt((residuals**2).mean()))
        bias = float(residuals.mean())

        # Active count + entropy
        weights = it.strategy_weights
        active_count = len(it.active_strategies)
        active_weights = [w for w in weights.values() if w > 0.0]
        entropy = -sum(w * math.log(w) for w in active_weights) if active_weights else 0.0

        # Top strategies
        sorted_strats = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top1_name = sorted_strats[0][0] if sorted_strats else ""
        top1_wt = sorted_strats[0][1] if sorted_strats else 0.0
        top3_names = [s[0] for s in sorted_strats[:3]]
        top3_wts = [s[1] for s in sorted_strats[:3]]

        # Dominant cluster
        if strategy_clusters:
            cluster_weights: dict[int, float] = {}
            for name, wt in weights.items():
                cid = strategy_clusters.get(name, -1)
                cluster_weights[cid] = cluster_weights.get(cid, 0.0) + wt
            if cluster_weights:
                dom_cluster = max(cluster_weights, key=cluster_weights.get)
                dom_weight = cluster_weights[dom_cluster]
            else:
                dom_cluster = -1
                dom_weight = 0.0
        else:
            dom_cluster = -1
            dom_weight = 0.0

        rows.append(
            {
                "case_id": case.case_id,
                "iteration": it.iteration,
                "convergence_delta": round(delta, 6),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "bias": round(bias, 4),
                "active_count": active_count,
                "weight_entropy": round(entropy, 4),
                "top1_strategy": top1_name,
                "top1_weight": round(top1_wt, 6),
                "top3_strategies": "; ".join(top3_names),
                "top3_weights": "; ".join(f"{w:.6f}" for w in top3_wts),
                "dominant_cluster": str(dom_cluster),
                "cluster_weight": round(dom_weight, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------


def generate_narrative(
    case: SentinelCase,
    trace: pd.DataFrame,
) -> str:
    """Generate a plain-language causal explanation for a sentinel case.

    Parameters
    ----------
    case:
        The sentinel case definition.
    trace:
        The iteration trace DataFrame.

    Returns
    -------
    Multi-line narrative string.
    """
    lines = [
        f"{'=' * 78}",
        f"  SENTINEL CASE: {case.case_id}",
        f"  Type: {case.case_type}",
        f"  Year: {case.year}",
        f"{'=' * 78}",
        "",
        f"  Description: {case.description}",
        "",
        f"  Iteration window: {case.iter_start} to {case.iter_end}",
        f"  Trace rows: {len(trace)}",
        "",
    ]

    if trace.empty:
        lines.append("  [No trace data available for this window]")
        return "\n".join(lines)

    # Key metrics at start and end of window
    first = trace.iloc[0]
    last = trace.iloc[-1]

    lines.extend(
        [
            "  Metrics at window start (iter {}):".format(int(first["iteration"])),
            f"    MAE:           {first['mae']:.4f} EUR/MWh",
            f"    Active count:  {int(first['active_count'])}",
            f"    Top-1 weight:  {first['top1_weight']:.4f}  ({first['top1_strategy']})",
            f"    Entropy:       {first['weight_entropy']:.4f}",
            "",
            "  Metrics at window end (iter {}):".format(int(last["iteration"])),
            f"    MAE:           {last['mae']:.4f} EUR/MWh",
            f"    Active count:  {int(last['active_count'])}",
            f"    Top-1 weight:  {last['top1_weight']:.4f}  ({last['top1_strategy']})",
            f"    Entropy:       {last['weight_entropy']:.4f}",
            "",
        ]
    )

    # MAE trajectory
    mae_start = first["mae"]
    mae_end = last["mae"]
    mae_best = trace["mae"].min()
    mae_best_iter = trace.loc[trace["mae"].idxmin(), "iteration"]

    if mae_end > mae_start * 1.05:
        mae_trend = "DEGRADED"
    elif mae_end < mae_start * 0.95:
        mae_trend = "IMPROVED"
    else:
        mae_trend = "STABLE"

    lines.extend(
        [
            f"  MAE trajectory: {mae_trend}",
            f"    Start: {mae_start:.4f} -> Best: {mae_best:.4f} (iter {int(mae_best_iter)})"
            f" -> End: {mae_end:.4f}",
            "",
        ]
    )

    # Active count trajectory
    ac_start = int(first["active_count"])
    ac_end = int(last["active_count"])
    ac_min = int(trace["active_count"].min())

    if ac_end == 0:
        ac_trend = "TOTAL COLLAPSE (all strategies eliminated)"
    elif ac_end < ac_start * 0.5:
        ac_trend = "SIGNIFICANT DECLINE"
    elif ac_end < ac_start:
        ac_trend = "MODERATE DECLINE"
    else:
        ac_trend = "STABLE"

    lines.extend(
        [
            f"  Active strategy trajectory: {ac_trend}",
            f"    Start: {ac_start} -> Min: {ac_min} -> End: {ac_end}",
            "",
        ]
    )

    # Convergence delta trajectory
    deltas = trace["convergence_delta"].replace(float("inf"), np.nan).dropna()
    if not deltas.empty:
        delta_start = deltas.iloc[0] if len(deltas) > 0 else float("nan")
        delta_end = deltas.iloc[-1] if len(deltas) > 0 else float("nan")
        lines.extend(
            [
                "  Convergence delta:",
                f"    Start: {delta_start:.6f} -> End: {delta_end:.6f}",
                "",
            ]
        )

    # Leadership changes
    top1_series = trace["top1_strategy"].tolist()
    leadership_changes = sum(
        1 for i in range(1, len(top1_series)) if top1_series[i] != top1_series[i - 1]
    )
    unique_leaders = len(set(top1_series))

    lines.extend(
        [
            f"  Leadership: {leadership_changes} changes across {unique_leaders} unique leaders",
            "",
        ]
    )

    # Causal narrative
    lines.append("  CAUSAL NARRATIVE:")
    lines.append("  " + "-" * 40)

    if case.case_type == "high_volatility_non_convergence":
        lines.extend(_narrate_high_vol(trace, case))
    elif case.case_type == "zero_active_convergence":
        lines.extend(_narrate_zero_active(trace, case))
    elif case.case_type == "early_accuracy_lost":
        lines.extend(_narrate_early_accuracy(trace, case))
    elif case.case_type == "cluster_switching":
        lines.extend(_narrate_cluster_switching(trace, case))
    else:
        lines.append(f"  (No specific narrative template for type: {case.case_type})")

    lines.append("")
    return "\n".join(lines)


def _narrate_high_vol(trace: pd.DataFrame, case: SentinelCase) -> list[str]:
    """Narrative for high-volatility non-convergence cases."""
    lines = []
    deltas = trace["convergence_delta"].replace(float("inf"), np.nan).dropna()
    is_oscillating = False
    if len(deltas) >= 4:
        diffs = deltas.diff().dropna()
        sign_changes = (diffs.shift(1) * diffs < 0).sum()
        is_oscillating = sign_changes > len(diffs) * 0.3

    lines.append("  During this high-volatility window, the market price repeatedly")
    if is_oscillating:
        lines.append("  overshoots and undershoots the real price. The convergence delta")
        lines.append("  oscillates rather than decaying, indicating the profit-weighted")
        lines.append("  average is being pulled in alternating directions by competing")
        lines.append("  strategy clusters. The ML regression cluster (which typically")
        lines.append("  captures >90% of weight) issues forecasts that are strongly")
        lines.append("  correlated with each other, amplifying directional swings.")
    else:
        lines.append("  adjusts slowly. The EMA dampening (alpha=0.1) prevents large")
        lines.append("  per-step corrections, but the market cannot track fast-moving")
        lines.append("  real prices across volatile days. The result is persistent")
        lines.append("  non-convergence with a slowly drifting delta.")

    lines.append("")
    lines.append("  Mechanism: EMA dampening + correlated ML forecasts + positive-profit")
    lines.append("  truncation (Phase 10c: ML strategies are the primary driver of 2024")
    lines.append("  oscillation).")
    return lines


def _narrate_zero_active(trace: pd.DataFrame, case: SentinelCase) -> list[str]:
    """Narrative for zero-active convergence (absorbing collapse)."""
    lines = []

    # Find the collapse point within the trace
    collapse_iter = None
    for _, row in trace.iterrows():
        if row["active_count"] == 0:
            collapse_iter = int(row["iteration"])
            break

    if collapse_iter is not None:
        lines.append(f"  At iteration {collapse_iter}, the last remaining strategies")
    else:
        lines.append("  Across this window, the remaining strategies gradually")

    lines.extend(
        [
            "  lose all accumulated profit and are eliminated by the positive-",
            "  profit truncation rule (w_i = max(Pi_i, 0) / sum(max(Pi_j, 0))).",
            "  Once every strategy has non-positive total profit, all weights",
            "  become zero and the market price freezes at its last value.",
            "",
            "  This is an absorbing state: no strategy can regain positive",
            "  profit because all trading positions q_i = sign(f_i - P_market)",
            "  produce zero net profit when the market price is frozen.",
            "",
            "  Mechanism: positive-profit truncation creates a one-way ratchet.",
            "  Strategies that underperform even briefly lose weight permanently.",
            "  In 2025, the real price trajectory is such that eventually all",
            "  strategies accumulate net-negative total profit (Phase 10c).",
        ]
    )
    return lines


def _narrate_early_accuracy(trace: pd.DataFrame, case: SentinelCase) -> list[str]:
    """Narrative for dates where early iterations were more accurate."""
    lines = []
    first = trace.iloc[0]
    last = trace.iloc[-1]

    lines.extend(
        [
            "  The initial market price (from last settlement or naive forecast)",
            f"  achieves MAE = {first['mae']:.4f}, but the iterative repricing",
            f"  degrades accuracy to MAE = {last['mae']:.4f} by the final iteration.",
            "",
            "  This counter-intuitive result occurs because the profit-weighted",
            "  selection preferentially amplifies strategies that were profitable",
            "  on average, even if their forecasts for specific days are poor.",
            "  On these dates, the naive starting price happened to be closer",
            "  to the realised price than the consensus of profitable strategies.",
            "",
            "  Mechanism: the market optimises for aggregate profitability, not",
            "  per-date accuracy. Strategies that are profitable overall can",
            "  still have poor forecasts on individual dates, and the weighting",
            "  scheme has no mechanism to down-weight them day-by-day.",
        ]
    )
    return lines


def _narrate_cluster_switching(trace: pd.DataFrame, case: SentinelCase) -> list[str]:
    """Narrative for cluster-switching episodes."""
    lines = []

    top1_series = trace["top1_strategy"].tolist()
    iters = trace["iteration"].tolist()

    # Find transition points
    transitions = []
    for i in range(1, len(top1_series)):
        if top1_series[i] != top1_series[i - 1]:
            transitions.append((iters[i], top1_series[i - 1], top1_series[i]))

    lines.append(f"  This window contains {len(transitions)} leadership transition(s):")
    for it_num, old, new in transitions[:5]:
        lines.append(f"    Iter {it_num}: {old} -> {new}")

    if len(transitions) > 5:
        lines.append(f"    ... and {len(transitions) - 5} more")

    lines.extend(
        [
            "",
            "  Frequent leadership changes indicate that no single strategy or",
            "  cluster has a stable profit advantage. The positive-profit",
            "  truncation rule amplifies small profit differences into large",
            "  weight shifts, causing the top position to flip between strategies",
            "  that are close competitors. This contributes to price oscillation",
            "  because each leadership change alters the weighted forecast average.",
        ]
    )
    return lines


# ---------------------------------------------------------------------------
# Case summary
# ---------------------------------------------------------------------------


def build_case_summary(
    case: SentinelCase,
    trace: pd.DataFrame,
) -> dict:
    """Build a one-row summary dict for the sentinel case."""
    if trace.empty:
        return {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "year": case.year,
            "description": case.description,
            "iter_start": case.iter_start,
            "iter_end": case.iter_end,
            "n_trace_rows": 0,
        }

    first = trace.iloc[0]
    last = trace.iloc[-1]

    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "year": case.year,
        "description": case.description,
        "iter_start": case.iter_start,
        "iter_end": case.iter_end,
        "n_trace_rows": len(trace),
        "start_mae": round(float(first["mae"]), 4),
        "end_mae": round(float(last["mae"]), 4),
        "best_mae": round(float(trace["mae"].min()), 4),
        "start_active": int(first["active_count"]),
        "end_active": int(last["active_count"]),
        "start_top1": first["top1_strategy"],
        "end_top1": last["top1_strategy"],
        "leadership_changes": sum(
            1
            for i in range(1, len(trace))
            if trace.iloc[i]["top1_strategy"] != trace.iloc[i - 1]["top1_strategy"]
        ),
    }


# ---------------------------------------------------------------------------
# Load cluster assignments from Phase 10d
# ---------------------------------------------------------------------------


def load_cluster_assignments(year: int) -> dict[str, int] | None:
    """Load strategy -> forecast cluster mapping from Phase 10d results."""
    path = PHASE10_DIR / "strategy_clusters.csv"
    if not path.exists():
        logger.warning("Cluster assignments not found: %s", path)
        return None

    df = pd.read_csv(path)
    year_df = df[df["year"] == year]
    if year_df.empty:
        return None

    return dict(zip(year_df["strategy"], year_df["forecast_cluster"], strict=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def select_sentinel_cases(
    year: int,
    real_prices: pd.Series,
    equilibrium: FuturesMarketEquilibrium,
    strategy_clusters: dict[str, int] | None = None,
) -> list[SentinelCase]:
    """Select sentinel cases for one year."""
    cases: list[SentinelCase] = []

    n_iters = len(equilibrium.iterations)

    if year == 2024:
        # Case 1: High-volatility non-convergence window
        start_pos, end_pos = find_high_volatility_window(real_prices)
        focus_dates = list(real_prices.index[start_pos : end_pos + 1])
        # Trace iterations 0 to 100 (the first 100 iterations where most change happens)
        cases.append(
            SentinelCase(
                case_id=f"hvnc_{year}",
                case_type="high_volatility_non_convergence",
                year=year,
                description=(
                    f"High-volatility window ({len(focus_dates)} days) "
                    f"with persistent non-convergence."
                ),
                iter_start=0,
                iter_end=min(99, n_iters - 1),
                focus_dates=focus_dates,
            )
        )

        # Case 2: Cluster switching in 2024
        cs_start, cs_end = find_cluster_switching_window(
            equilibrium, strategy_clusters, window_size=30
        )
        cases.append(
            SentinelCase(
                case_id=f"clsw_{year}",
                case_type="cluster_switching",
                year=year,
                description=(
                    f"Iteration window ({cs_start}-{cs_end}) with most leadership transitions."
                ),
                iter_start=cs_start,
                iter_end=cs_end,
            )
        )

    if year == 2025:
        # Case 3: Zero-active convergence
        ac_start, ac_end = find_active_collapse_window(equilibrium, margin=25)
        cases.append(
            SentinelCase(
                case_id=f"zact_{year}",
                case_type="zero_active_convergence",
                year=year,
                description=(
                    f"Window around active-strategy collapse to zero (iters {ac_start}-{ac_end})."
                ),
                iter_start=ac_start,
                iter_end=ac_end,
            )
        )

    # Case 4: Early accuracy lost (both years)
    ea_dates = find_early_accuracy_dates(equilibrium, real_prices, n_dates=5)
    if ea_dates:
        cases.append(
            SentinelCase(
                case_id=f"ealost_{year}",
                case_type="early_accuracy_lost",
                year=year,
                description=(
                    f"{len(ea_dates)} dates where iteration-0 was more accurate "
                    f"than the final iteration."
                ),
                iter_start=0,
                iter_end=min(99, n_iters - 1),
                focus_dates=ea_dates,
            )
        )

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 10e: sentinel case studies -- iteration-level traces."
    )
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

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    years = [args.year] if args.year else [2024, 2025]

    all_summaries: list[dict] = []

    for year in years:
        # Load market result
        mpath = RESULTS_DIR / f"market_{year}.pkl"
        if not mpath.exists():
            logger.error("Missing %s", mpath)
            continue
        with open(mpath, "rb") as f:
            market_result: FuturesMarketResult = pickle.load(f)

        equilibrium = market_result.equilibrium

        # Load real prices from Phase 8 forecasts
        fpath = PHASE8_DIR / f"forecasts_{year}.pkl"
        if not fpath.exists():
            logger.error("Missing %s", fpath)
            continue
        with open(fpath, "rb") as f:
            phase8_data: dict = pickle.load(f)

        real_prices: pd.Series = phase8_data["real_prices"]

        # Load cluster assignments from Phase 10d
        strategy_clusters = load_cluster_assignments(year)

        logger.info(
            "Year %d: %d iterations, %d dates, clusters=%s",
            year,
            len(equilibrium.iterations),
            len(real_prices),
            "loaded" if strategy_clusters else "unavailable",
        )

        # Select sentinel cases
        cases = select_sentinel_cases(year, real_prices, equilibrium, strategy_clusters)
        logger.info("Year %d: selected %d sentinel cases", year, len(cases))

        for case in cases:
            logger.info("Processing case: %s (%s)", case.case_id, case.case_type)

            # Build trace
            trace = build_iteration_trace(case, equilibrium, real_prices, strategy_clusters)

            # Save trace CSV
            trace_path = TRACE_DIR / f"{case.case_id}.csv"
            trace.to_csv(trace_path, index=False)
            logger.info("  Saved trace: %s (%d rows)", trace_path, len(trace))

            # Generate and print narrative
            narrative = generate_narrative(case, trace)
            print(narrative)

            # Build summary
            summary = build_case_summary(case, trace)
            all_summaries.append(summary)

    # Save combined summaries
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = PHASE10_DIR / "sentinel_summaries.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved sentinel summaries: %s", summary_path)

    print(f"\n{'=' * 78}")
    print(f"  SENTINEL CASE STUDIES COMPLETE: {len(all_summaries)} cases analysed")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
