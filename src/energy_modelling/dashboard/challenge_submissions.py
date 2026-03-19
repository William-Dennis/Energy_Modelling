"""Streamlit dashboard for comparing challenge submission strategies.

Run with::

    streamlit run src/energy_modelling/dashboard/challenge_submissions.py
"""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from energy_modelling.challenge.runner import ChallengeBacktestResult, run_challenge_backtest
from energy_modelling.challenge.scoring import leaderboard_score, market_leaderboard_score
from energy_modelling.challenge.types import ChallengeStrategy
from energy_modelling.challenge.data import build_feature_glossary, write_challenge_data
from energy_modelling.challenge.market_runner import MarketEvaluationResult, run_market_evaluation

_DATASET_DEFAULT = Path("data/challenge/daily_public.csv")
_HIDDEN_DATASET_DEFAULT = Path("data/challenge/daily_hidden_test_full.csv")
_SOURCE_DATASET_DEFAULT = Path("kaggle_upload/dataset_de_lu.csv")
_SUBMISSION_SKIP_MODULES = frozenset({"__init__", "common"})


def _class_display_name(cls: type) -> str:
    name = cls.__name__
    name = re.sub(r"Strategy$", "", name)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name.strip()


def _class_description(cls: type) -> str:
    class_doc = cls.__doc__ or ""
    first_para = class_doc.strip().split("\n\n")[0]
    return " ".join(first_para.split())


def _discover_submission_strategies() -> tuple[
    dict[str, Callable[[], ChallengeStrategy]],
    dict[str, str],
]:
    import submission as submission_pkg

    submission_dir = Path(submission_pkg.__file__).parent
    factories: dict[str, Callable[[], ChallengeStrategy]] = {}
    descriptions: dict[str, str] = {}

    for py_file in sorted(submission_dir.glob("*.py")):
        module_stem = py_file.stem
        if module_stem in _SUBMISSION_SKIP_MODULES:
            continue

        module_name = f"submission.{module_stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not ChallengeStrategy
                and issubclass(obj, ChallengeStrategy)
                and not inspect.isabstract(obj)
                and obj.__module__ == module_name
            ):
                display = _class_display_name(obj)
                descriptions[display] = _class_description(obj)
                factories[display] = obj

    return factories, descriptions


_STRATEGY_FACTORIES, _STRATEGY_DESCRIPTIONS = _discover_submission_strategies()


def _resolve_path(path_like: Path | str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _ensure_challenge_datasets(public_path: Path, hidden_path: Path) -> tuple[Path, Path | None]:
    if public_path.exists() and hidden_path.exists():
        return public_path, hidden_path

    source_dataset = _resolve_path(_SOURCE_DATASET_DEFAULT)
    if not source_dataset.exists():
        return public_path, hidden_path if hidden_path.exists() else None

    output_dir = public_path.parent
    write_challenge_data(source_dataset, output_dir, include_hidden_test=True)
    return public_path, hidden_path if hidden_path.exists() else None


def _load_daily_dataset(dataset_path: Path | str) -> pd.DataFrame:
    daily = pd.read_csv(dataset_path, parse_dates=["delivery_date"])
    daily["delivery_date"] = daily["delivery_date"].dt.date
    return daily


def _combine_public_and_hidden(
    public_daily: pd.DataFrame, hidden_daily: pd.DataFrame | None
) -> pd.DataFrame:
    if hidden_daily is None:
        return public_daily.copy()
    combined = pd.concat([public_daily, hidden_daily], ignore_index=True)
    combined["delivery_date"] = pd.to_datetime(combined["delivery_date"]).dt.date
    return combined.sort_values("delivery_date").reset_index(drop=True)


def _evaluate_strategy(
    strategy_factory: Callable[[], ChallengeStrategy],
    daily: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
) -> ChallengeBacktestResult:
    strategy = strategy_factory()
    return run_challenge_backtest(strategy, daily, training_end, evaluation_start, evaluation_end)


def _leaderboard_frame(results: dict[str, ChallengeBacktestResult]) -> pd.DataFrame:
    rows = []
    for strategy_name, result in results.items():
        metrics = result.metrics
        score = leaderboard_score(metrics)
        rows.append(
            {
                "Strategy": strategy_name,
                "Total PnL": metrics["total_pnl"],
                "Sharpe": metrics["sharpe_ratio"],
                "Max Drawdown": metrics["max_drawdown"],
                "Trades": metrics["trade_count"],
                "Win Rate": metrics["win_rate"],
                "Leaderboard Score": score,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["Total PnL", "Sharpe", "Max Drawdown"], ascending=[False, False, True]
    )


def _monthly_pnl(result: ChallengeBacktestResult) -> pd.DataFrame:
    pnl = result.daily_pnl.copy()
    idx = pd.DatetimeIndex(pnl.index)
    pnl.index = idx
    monthly = pnl.groupby([idx.year, idx.month]).sum().reset_index()
    monthly.columns = ["year", "month", "pnl"]
    return monthly


def _evaluate_periods(
    selected: list[str],
    public_daily: pd.DataFrame,
    hidden_daily: pd.DataFrame | None,
) -> tuple[dict[str, ChallengeBacktestResult], dict[str, ChallengeBacktestResult]]:
    validation_results: dict[str, ChallengeBacktestResult] = {}
    hidden_results: dict[str, ChallengeBacktestResult] = {}
    combined_daily = _combine_public_and_hidden(public_daily, hidden_daily)

    for strategy_name in selected:
        factory = _STRATEGY_FACTORIES[strategy_name]
        validation_results[strategy_name] = _evaluate_strategy(
            factory,
            public_daily,
            date(2023, 12, 31),
            date(2024, 1, 1),
            date(2024, 12, 31),
        )
        if hidden_daily is not None:
            hidden_results[strategy_name] = _evaluate_strategy(
                factory,
                combined_daily,
                date(2024, 12, 31),
                date(2025, 1, 1),
                date(2025, 12, 31),
            )

    return validation_results, hidden_results


def _period_summary_frame(
    validation_results: dict[str, ChallengeBacktestResult],
    hidden_results: dict[str, ChallengeBacktestResult],
) -> pd.DataFrame:
    rows = []
    strategy_names = sorted(set(validation_results) | set(hidden_results))
    for strategy_name in strategy_names:
        validation = validation_results.get(strategy_name)
        hidden = hidden_results.get(strategy_name)
        rows.append(
            {
                "Strategy": strategy_name,
                "2024 PnL": validation.metrics["total_pnl"] if validation else float("nan"),
                "2024 Sharpe": validation.metrics["sharpe_ratio"] if validation else float("nan"),
                "2024 Max DD": validation.metrics["max_drawdown"] if validation else float("nan"),
                "2025 PnL": hidden.metrics["total_pnl"] if hidden else float("nan"),
                "2025 Sharpe": hidden.metrics["sharpe_ratio"] if hidden else float("nan"),
                "2025 Max DD": hidden.metrics["max_drawdown"] if hidden else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["2024 PnL", "2025 PnL"], ascending=[False, False])


def _comparison_curve_frame(
    results_by_period: dict[str, dict[str, ChallengeBacktestResult]],
    series_name: str,
) -> pd.DataFrame:
    rows = []
    for period, strategy_results in results_by_period.items():
        for strategy_name, result in strategy_results.items():
            series = getattr(result, series_name)
            for delivery_date, value in series.items():
                rows.append(
                    {
                        "Date": delivery_date,
                        "Strategy": strategy_name,
                        "Period": period,
                        "Value": value,
                    }
                )
    return pd.DataFrame(rows)


def _format_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["Leaderboard Score"]).assign(
        **{
            "Total PnL": lambda df: df["Total PnL"].map(lambda value: f"EUR {value:,.2f}"),
            "Sharpe": lambda df: df["Sharpe"].map(lambda value: f"{value:.2f}"),
            "Max Drawdown": lambda df: df["Max Drawdown"].map(lambda value: f"EUR {value:,.2f}"),
            "Trades": lambda df: df["Trades"].map(lambda value: f"{value:.0f}"),
            "Win Rate": lambda df: df["Win Rate"].map(lambda value: f"{value:.1%}"),
        }
    )


# ---------------------------------------------------------------------------
# Synthetic Futures Market helpers
# ---------------------------------------------------------------------------


def _run_market_for_period(
    selected: list[str],
    daily: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
) -> MarketEvaluationResult | None:
    """Run the synthetic futures market for a single evaluation period."""
    factories = {name: _STRATEGY_FACTORIES[name] for name in selected}
    try:
        return run_market_evaluation(
            strategy_factories=factories,
            daily_data=daily,
            training_end=training_end,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
        )
    except Exception:  # noqa: BLE001
        return None


def _market_leaderboard_frame(market_result: MarketEvaluationResult) -> pd.DataFrame:
    """Build a leaderboard from market-adjusted results."""
    rows = []
    for name, result in market_result.market_results.items():
        m = result.metrics
        orig = market_result.original_results[name].metrics
        score = market_leaderboard_score(m)
        rows.append(
            {
                "Strategy": name,
                "Market PnL": m["total_pnl"],
                "Original PnL": orig["total_pnl"],
                "Sharpe": m["sharpe_ratio"],
                "Max Drawdown": m["max_drawdown"],
                "Trades": m["trade_count"],
                "Win Rate": m["win_rate"],
                "Leaderboard Score": score,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["Market PnL", "Sharpe", "Max Drawdown"], ascending=[False, False, True]
    )


def _format_market_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["Leaderboard Score"]).assign(
        **{
            "Market PnL": lambda df: df["Market PnL"].map(lambda v: f"EUR {v:,.2f}"),
            "Original PnL": lambda df: df["Original PnL"].map(lambda v: f"EUR {v:,.2f}"),
            "Sharpe": lambda df: df["Sharpe"].map(lambda v: f"{v:.2f}"),
            "Max Drawdown": lambda df: df["Max Drawdown"].map(lambda v: f"EUR {v:,.2f}"),
            "Trades": lambda df: df["Trades"].map(lambda v: f"{v:.0f}"),
            "Win Rate": lambda df: df["Win Rate"].map(lambda v: f"{v:.1%}"),
        }
    )


def _render_market_section(
    period_name: str,
    market_result: MarketEvaluationResult | None,
) -> None:
    """Render the synthetic market view for one evaluation period."""
    if market_result is None:
        st.info(f"Market simulation not available for {period_name}.")
        return

    eq = market_result.equilibrium

    # Convergence info
    conv_col1, conv_col2, conv_col3 = st.columns(3)
    conv_col1.metric("Converged", "Yes" if eq.converged else "No")
    conv_col2.metric("Iterations", str(len(eq.iterations)))
    conv_col3.metric("Final Delta", f"EUR {eq.convergence_delta:.4f}")

    # Market leaderboard
    st.subheader(f"Market-Adjusted Leaderboard - {period_name}")
    lb = _market_leaderboard_frame(market_result)
    if not lb.empty:
        st.dataframe(_format_market_leaderboard(lb), use_container_width=True)

    # Rank change comparison
    st.subheader("Rank Change: Original vs Market")
    orig_order = lb.sort_values("Original PnL", ascending=False)["Strategy"].reset_index(drop=True)
    market_order = lb["Strategy"].reset_index(drop=True)  # already sorted by market PnL
    rank_rows = []
    for strategy in lb["Strategy"]:
        orig_rank = int(orig_order[orig_order == strategy].index[0]) + 1
        mkt_rank = int(market_order[market_order == strategy].index[0]) + 1
        rank_rows.append(
            {
                "Strategy": strategy,
                "Original Rank": orig_rank,
                "Market Rank": mkt_rank,
                "Change": orig_rank - mkt_rank,
            }
        )
    st.dataframe(pd.DataFrame(rank_rows), use_container_width=True)

    # Convergence plot
    if len(eq.iterations) > 1:
        st.subheader("Convergence")
        deltas = []
        for i, it in enumerate(eq.iterations):
            if i == 0:
                prev = market_result.original_results
                # compute max delta from initial
                init_prices = it.market_prices  # approximation for display
                deltas.append({"Iteration": i, "Max Delta": 0.0})
            else:
                prev_prices = eq.iterations[i - 1].market_prices
                d = float((it.market_prices - prev_prices).abs().max())
                deltas.append({"Iteration": i, "Max Delta": d})
        delta_df = pd.DataFrame(deltas)
        fig_conv = px.line(
            delta_df,
            x="Iteration",
            y="Max Delta",
            title="Market Price Convergence (max absolute delta per iteration)",
        )
        fig_conv.add_hline(y=0.01, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_conv, use_container_width=True)

    # Weight evolution
    st.subheader("Strategy Weights Across Iterations")
    weight_rows = []
    for it in eq.iterations:
        for name, w in it.strategy_weights.items():
            weight_rows.append({"Iteration": it.iteration, "Strategy": name, "Weight": w})
    if weight_rows:
        weight_df = pd.DataFrame(weight_rows)
        fig_w = px.area(
            weight_df,
            x="Iteration",
            y="Weight",
            color="Strategy",
            title="Strategy Weight Evolution",
        )
        st.plotly_chart(fig_w, use_container_width=True)

    # Market price vs settlement
    st.subheader("Market Price vs Settlement Price")
    mp = eq.final_market_prices.reset_index()
    mp.columns = ["Date", "Market Price"]
    # Get settlement from original results (any strategy will do)
    first_orig = next(iter(market_result.original_results.values()))
    settlement_data = first_orig.daily_pnl.copy()  # placeholder for getting dates
    # Reconstruct settlement from original PnL and predictions
    price_rows = []
    for d in eq.final_market_prices.index:
        price_rows.append({"Date": d, "Market Price": float(eq.final_market_prices.loc[d])})
    price_df = pd.DataFrame(price_rows)
    fig_price = px.line(
        price_df,
        x="Date",
        y="Market Price",
        title="Converged Market Price",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # Cumulative market PnL
    st.subheader("Cumulative PnL (Market-Adjusted)")
    cum_rows = []
    for name, result in market_result.market_results.items():
        for d, val in result.cumulative_pnl.items():
            cum_rows.append({"Date": d, "Strategy": name, "Cumulative PnL": val})
    if cum_rows:
        cum_df = pd.DataFrame(cum_rows)
        fig_cum = px.line(
            cum_df,
            x="Date",
            y="Cumulative PnL",
            color="Strategy",
            title=f"Cumulative Market-Adjusted PnL - {period_name}",
        )
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_cum, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Challenge Submissions Dashboard",
        page_icon="🏁",
        layout="wide",
    )
    st.title("DE-LU Futures Challenge - Submission Dashboard")
    st.markdown(
        "Compare all example and submitted strategies in `submission/` on the public daily challenge dataset."
    )

    st.sidebar.header("Configuration")
    dataset_path = st.sidebar.text_input(
        "2024 public dataset path",
        value=str(_DATASET_DEFAULT),
        help="Path to the public daily challenge CSV.",
    )
    hidden_dataset_path = st.sidebar.text_input(
        "2025 hidden dataset path",
        value=str(_HIDDEN_DATASET_DEFAULT),
        help="Path to the organizer-only hidden 2025 daily CSV.",
    )
    selected = st.sidebar.multiselect(
        "Strategies",
        options=list(_STRATEGY_FACTORIES.keys()),
        default=list(_STRATEGY_FACTORIES.keys()),
    )
    run_btn = st.sidebar.button("Run Comparison", type="primary")

    st.sidebar.divider()
    st.sidebar.header("Organizer Tools")
    enable_market = st.sidebar.checkbox(
        "Enable Synthetic Market",
        value=False,
        help="Run the synthetic futures market model to see market-adjusted rankings. "
        "This is an organizer-only tool — students do not see this.",
    )

    if not selected:
        st.info("Select at least one strategy to compare.")
        return

    strategy_help = "\n".join(
        f"- {name}: {_STRATEGY_DESCRIPTIONS.get(name, '')}" for name in selected
    )
    st.sidebar.caption(strategy_help)

    if not run_btn:
        st.info("Choose strategies and click **Run Comparison**.")
        return

    public_path = _resolve_path(dataset_path)
    hidden_path = _resolve_path(hidden_dataset_path)
    public_path, hidden_path_or_none = _ensure_challenge_datasets(public_path, hidden_path)

    if not public_path.exists():
        st.error(f"Public dataset not found: {public_path}")
        return

    public_daily = _load_daily_dataset(public_path)
    hidden_daily = _load_daily_dataset(hidden_path_or_none) if hidden_path_or_none else None
    glossary = build_feature_glossary(public_daily)

    with st.spinner("Evaluating submissions..."):
        try:
            validation_results, hidden_results = _evaluate_periods(
                selected, public_daily, hidden_daily
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Evaluation failed: {exc}")
            return

    if not validation_results:
        st.error("No strategies completed successfully.")
        return

    leaderboard_2024 = _leaderboard_frame(validation_results)
    leaderboard_2025 = _leaderboard_frame(hidden_results) if hidden_results else pd.DataFrame()
    period_summary = _period_summary_frame(validation_results, hidden_results)

    # --- Synthetic market evaluation (organizer only) ---
    market_2024: MarketEvaluationResult | None = None
    market_2025: MarketEvaluationResult | None = None
    if enable_market and len(selected) >= 2:
        combined_daily = _combine_public_and_hidden(public_daily, hidden_daily)
        with st.spinner("Running synthetic futures market..."):
            market_2024 = _run_market_for_period(
                selected,
                public_daily,
                training_end=date(2023, 12, 31),
                evaluation_start=date(2024, 1, 1),
                evaluation_end=date(2024, 12, 31),
            )
            if hidden_daily is not None:
                market_2025 = _run_market_for_period(
                    selected,
                    combined_daily,
                    training_end=date(2024, 12, 31),
                    evaluation_start=date(2025, 1, 1),
                    evaluation_end=date(2025, 12, 31),
                )

    st.header("Period Summary")
    st.dataframe(
        period_summary.assign(
            **{
                "2024 PnL": lambda frame: frame["2024 PnL"].map(lambda value: f"EUR {value:,.2f}"),
                "2024 Sharpe": lambda frame: frame["2024 Sharpe"].map(lambda value: f"{value:.2f}"),
                "2024 Max DD": lambda frame: frame["2024 Max DD"].map(
                    lambda value: f"EUR {value:,.2f}"
                ),
                "2025 PnL": lambda frame: frame["2025 PnL"].map(
                    lambda value: "-" if pd.isna(value) else f"EUR {value:,.2f}"
                ),
                "2025 Sharpe": lambda frame: frame["2025 Sharpe"].map(
                    lambda value: "-" if pd.isna(value) else f"{value:.2f}"
                ),
                "2025 Max DD": lambda frame: frame["2025 Max DD"].map(
                    lambda value: "-" if pd.isna(value) else f"EUR {value:,.2f}"
                ),
            }
        ),
        use_container_width=True,
    )

    st.header("Feature Timing")
    timing_col1, timing_col2 = st.columns([1, 2])
    with timing_col1:
        counts = glossary.groupby("timing_group").size().reset_index(name="columns")
        st.dataframe(counts, use_container_width=True)
    with timing_col2:
        timing_tab1, timing_tab2 = st.tabs(["Lagged Realised", "Same-Day Forecast"])
        with timing_tab1:
            realised_cols = glossary[glossary["timing_group"] == "lagged_realised"]
            st.dataframe(realised_cols[["column", "description"]], use_container_width=True)
        with timing_tab2:
            forecast_cols = glossary[glossary["timing_group"] == "same_day_forecast"]
            st.dataframe(forecast_cols[["column", "description"]], use_container_width=True)

    tab_names = ["2024 Validation", "2025 Hidden Test"]
    if enable_market:
        tab_names += ["Market: 2024", "Market: 2025"]
    period_tabs = st.tabs(tab_names)
    with period_tabs[0]:
        st.subheader("2024 Leaderboard")
        st.dataframe(_format_leaderboard(leaderboard_2024), use_container_width=True)
        st.caption("Top 2024 strategies after the forecast-timing fix are often forecast-driven.")
    with period_tabs[1]:
        if leaderboard_2025.empty:
            st.info(
                "Hidden 2025 data is not available, so only the 2024 public evaluation is shown."
            )
        else:
            st.subheader("2025 Hidden-Test Leaderboard")
            st.dataframe(_format_leaderboard(leaderboard_2025), use_container_width=True)
    if enable_market:
        with period_tabs[2]:
            if len(selected) < 2:
                st.info("Market simulation requires at least 2 strategies.")
            else:
                _render_market_section("2024", market_2024)
        with period_tabs[3]:
            if len(selected) < 2:
                st.info("Market simulation requires at least 2 strategies.")
            elif hidden_daily is None:
                st.info("Hidden 2025 data is not available for market simulation.")
            else:
                _render_market_section("2025", market_2025)

    st.header("Leaderboard Snapshot")
    snapshot_rows = (
        leaderboard_2024[["Strategy", "Total PnL", "Sharpe", "Max Drawdown"]].head(5).copy()
    )
    snapshot_rows.columns = ["Strategy", "2024 PnL", "2024 Sharpe", "2024 Max DD"]
    if not leaderboard_2025.empty:
        top_2025 = leaderboard_2025[["Strategy", "Total PnL", "Sharpe", "Max Drawdown"]].copy()
        top_2025.columns = ["Strategy", "2025 PnL", "2025 Sharpe", "2025 Max DD"]
        snapshot_rows = snapshot_rows.merge(top_2025, on="Strategy", how="left")
    st.dataframe(snapshot_rows, use_container_width=True)

    st.header("Cumulative PnL Comparison")
    cumulative_df = _comparison_curve_frame(
        {"2024": validation_results, "2025": hidden_results},
        "cumulative_pnl",
    )
    fig_cum = px.line(
        cumulative_df,
        x="Date",
        y="Value",
        color="Strategy",
        line_dash="Period",
        title="Cumulative PnL by Strategy and Period",
        labels={"Value": "Cumulative PnL"},
    )
    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_cum, use_container_width=True)

    st.header("Drawdown Comparison")
    drawdown_payload: dict[str, dict[str, ChallengeBacktestResult]] = {"2024": {}, "2025": {}}
    for strategy_name, result in validation_results.items():
        drawdown_result = ChallengeBacktestResult(
            predictions=result.predictions,
            daily_pnl=result.daily_pnl,
            cumulative_pnl=result.cumulative_pnl.cummax() - result.cumulative_pnl,
            trade_count=result.trade_count,
            days_evaluated=result.days_evaluated,
            metrics=result.metrics,
        )
        drawdown_payload["2024"][strategy_name] = drawdown_result
    for strategy_name, result in hidden_results.items():
        drawdown_result = ChallengeBacktestResult(
            predictions=result.predictions,
            daily_pnl=result.daily_pnl,
            cumulative_pnl=result.cumulative_pnl.cummax() - result.cumulative_pnl,
            trade_count=result.trade_count,
            days_evaluated=result.days_evaluated,
            metrics=result.metrics,
        )
        drawdown_payload["2025"][strategy_name] = drawdown_result
    drawdown_df = _comparison_curve_frame(drawdown_payload, "cumulative_pnl")
    fig_dd = px.line(
        drawdown_df,
        x="Date",
        y="Value",
        color="Strategy",
        line_dash="Period",
        title="Drawdown by Strategy and Period",
        labels={"Value": "Drawdown"},
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.header("Strategy Detail")
    focus_strategy = st.selectbox("Inspect strategy", options=list(validation_results.keys()))
    detail_tabs = st.tabs(["2024 Detail", "2025 Detail"])

    def _render_detail(period_name: str, result: ChallengeBacktestResult | None) -> None:
        if result is None:
            st.info(f"No {period_name} result is available for this strategy.")
            return

        focus_metrics = result.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total PnL", f"EUR {focus_metrics['total_pnl']:,.0f}")
        c2.metric("Sharpe", f"{focus_metrics['sharpe_ratio']:.2f}")
        c3.metric("Max Drawdown", f"EUR {focus_metrics['max_drawdown']:,.0f}")
        c4.metric("Trades", f"{focus_metrics['trade_count']:.0f}")
        c5.metric("Win Rate", f"{focus_metrics['win_rate']:.1%}")

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.subheader(f"Daily PnL Distribution - {period_name}")
            fig_hist = px.histogram(
                result.daily_pnl,
                nbins=40,
                title=f"Daily PnL Distribution - {focus_strategy} ({period_name})",
                labels={"value": "Daily PnL", "count": "Frequency"},
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_hist, use_container_width=True)

        with detail_col2:
            st.subheader(f"Monthly PnL - {period_name}")
            monthly = _monthly_pnl(result)
            pivot = monthly.pivot(index="year", columns="month", values="pnl").fillna(0.0)
            pivot.columns = [
                [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ][month - 1]
                for month in pivot.columns
            ]
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=[str(year) for year in pivot.index],
                    colorscale="RdYlGn",
                    zmid=0,
                    text=[[f"EUR {value:,.0f}" for value in row] for row in pivot.values],
                    texttemplate="%{text}",
                    hovertemplate="Month: %{x}<br>Year: %{y}<br>PnL: %{text}<extra></extra>",
                )
            )
            fig_heat.update_layout(
                title=f"Monthly PnL - {focus_strategy} ({period_name})",
                yaxis_autorange="reversed",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader(f"Latest Predictions - {period_name}")
        latest = pd.DataFrame(
            {
                "prediction": result.predictions,
                "daily_pnl": result.daily_pnl,
                "cumulative_pnl": result.cumulative_pnl,
            }
        ).tail(20)
        st.dataframe(latest, use_container_width=True)

    with detail_tabs[0]:
        _render_detail("2024", validation_results.get(focus_strategy))
    with detail_tabs[1]:
        _render_detail("2025", hidden_results.get(focus_strategy))


if __name__ == "__main__":
    main()
