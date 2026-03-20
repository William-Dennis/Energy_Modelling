"""Tab -- Benchmark Comparison: strategy robustness across entry-price scenarios."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.backtest.benchmarks import ALL_BENCHMARKS, get_benchmark
from energy_modelling.backtest.io import RESULTS_DIR, load_backtest_results, save_backtest_results
from energy_modelling.backtest.runner import BacktestResult, run_backtest
from energy_modelling.dashboard._backtest import (
    STRATEGY_FACTORIES,
    _resolve_path,
    load_daily,
)

_DATASET_DEFAULT = Path("data/backtest/daily_public.csv")

_METRIC_LABELS: dict[str, str] = {
    "total_pnl": "Total PnL (EUR)",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown (EUR)",
    "win_rate": "Win Rate",
}


def _load_benchmark_results() -> dict[str, dict[str, BacktestResult]]:
    """Load all saved benchmark results. Returns {benchmark_id: {strategy: result}}."""
    results: dict[str, dict[str, BacktestResult]] = {}
    for bench_id in ALL_BENCHMARKS:
        path = RESULTS_DIR / f"benchmark_{bench_id}.pkl"
        loaded = load_backtest_results(path)
        if loaded is not None:
            results[bench_id] = loaded
    return results


def _build_comparison_matrix(
    bench_results: dict[str, dict[str, BacktestResult]],
    metric: str = "total_pnl",
) -> pd.DataFrame:
    """Build a strategy x benchmark matrix for a given metric."""
    rows: list[dict[str, object]] = []
    for bench_id, strat_results in bench_results.items():
        for strat_name, result in strat_results.items():
            rows.append(
                {
                    "Strategy": strat_name,
                    "Benchmark": bench_id,
                    "Value": result.metrics.get(metric, 0.0),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot(index="Strategy", columns="Benchmark", values="Value").fillna(0)


def _render_heatmap(matrix: pd.DataFrame, metric_label: str) -> None:
    """Render a Plotly heatmap of strategy x benchmark."""
    if matrix.empty:
        st.info("No benchmark data to display.")
        return
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="RdYlGn",
            text=matrix.round(2).astype(str).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar_title=metric_label,
        )
    )
    fig.update_layout(
        title=f"Strategy × Benchmark: {metric_label}",
        xaxis_title="Benchmark",
        yaxis_title="Strategy",
        height=max(300, len(matrix) * 35 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_comparison_table(matrix: pd.DataFrame) -> None:
    """Show the raw comparison table with EUR units in column headers."""
    if matrix.empty:
        return
    renamed = matrix.rename(columns={col: f"{col} (EUR)" for col in matrix.columns})
    st.dataframe(renamed, use_container_width=True)


def _run_benchmarks(
    selected_benchmarks: list[str],
    bench_results: dict[str, dict[str, BacktestResult]],
) -> None:
    """Recompute benchmarks for the selected IDs and save results."""
    pub_path = _resolve_path(_DATASET_DEFAULT)
    if not pub_path.exists():
        st.error(f"Dataset not found: {pub_path}")
        return

    daily = load_daily(pub_path)
    with st.spinner("Running benchmarks..."):
        for bench_id in selected_benchmarks:
            entry_prices = get_benchmark(bench_id, daily)
            strat_results: dict[str, BacktestResult] = {}
            for name, factory in STRATEGY_FACTORIES.items():
                strategy = factory()
                result = run_backtest(
                    strategy=strategy,
                    daily_data=daily,
                    training_end=date(2023, 12, 31),
                    evaluation_start=date(2024, 1, 1),
                    evaluation_end=date(2024, 12, 31),
                    entry_prices=entry_prices if bench_id != "baseline" else None,
                )
                strat_results[name] = result
            bench_results[bench_id] = strat_results
            save_backtest_results(strat_results, RESULTS_DIR / f"benchmark_{bench_id}.pkl")
    st.success(f"Computed {len(selected_benchmarks)} benchmarks.")


def render() -> None:
    """Render the benchmark comparison tab."""
    st.markdown(
        "Compare strategy performance across different **entry price benchmarks**. "
        "Each benchmark tests strategy robustness by varying the entry price "
        "(noisy, biased, or oracle)."
    )

    bench_results = _load_benchmark_results()

    if not bench_results:
        st.info(
            "No benchmark results found. Run `recompute-all` or use the **Recompute** "
            "button below to generate benchmark results."
        )

    available = list(ALL_BENCHMARKS.keys())
    selected_benchmarks = st.multiselect(
        "Entry Price Benchmarks",
        options=available,
        default=list(bench_results.keys()) if bench_results else ["baseline"],
        key="bench_select",
    )

    recompute = st.button("Recompute Benchmarks", type="primary", key="bench_recompute")
    if recompute and selected_benchmarks:
        _run_benchmarks(selected_benchmarks, bench_results)

    if not bench_results:
        return

    filtered = {k: v for k, v in bench_results.items() if k in selected_benchmarks}
    if not filtered:
        st.info("Select benchmarks above to view.")
        return

    metric = st.selectbox(
        "Metric",
        options=list(_METRIC_LABELS.keys()),
        format_func=lambda x: _METRIC_LABELS[x],
        key="bench_metric",
    )

    matrix = _build_comparison_matrix(filtered, metric)

    st.subheader("Comparison Table")
    if metric == "total_pnl":
        _render_comparison_table(matrix)
    else:
        st.dataframe(matrix.round(3), use_container_width=True)

    st.subheader("Heatmap")
    _render_heatmap(matrix, _METRIC_LABELS.get(metric, metric))
