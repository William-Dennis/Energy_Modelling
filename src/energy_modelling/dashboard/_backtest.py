"""Tab 2 -- Single-Strategy Backtest.

Runs one strategy at a time via the ``MarketEnvironment`` / ``BacktestRunner``
pipeline (EEX-style settlement), and renders cumulative PnL, drawdown,
daily PnL distribution, monthly heatmap, rolling Sharpe, and a metrics table.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from energy_modelling.dashboard import (
    class_display_name as _class_display_name,
    monthly_pnl_heatmap,
    render_metric_cards,
)
from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.strategy.analysis import compute_metrics, rolling_sharpe
from energy_modelling.strategy.base import Strategy
from energy_modelling.strategy.runner import BacktestRunner

_DATASET_DEFAULT = Path("kaggle_upload/dataset_de_lu.csv")
_STRATEGY_SKIP_MODULES = frozenset({"__init__", "base", "runner", "analysis", "template"})


# ---------------------------------------------------------------------------
# Strategy discovery
# ---------------------------------------------------------------------------


def _class_description(cls: type) -> str:
    module_doc = inspect.getmodule(cls).__doc__ or ""
    m = re.search(r"Description\s*\n[-]+\s*\n(.*?)(?:\n\s*\n|\Z)", module_doc, re.DOTALL)
    if m:
        return " ".join(m.group(1).split())
    class_doc = cls.__doc__ or ""
    return " ".join(class_doc.strip().split("\n\n")[0].split())


def _discover_strategies() -> tuple[
    dict[str, Callable[[MarketEnvironment], Strategy]],
    dict[str, str],
]:
    import energy_modelling.strategy as _pkg

    strategy_dir = Path(_pkg.__file__).parent
    factories: dict[str, Callable[[MarketEnvironment], Strategy]] = {}
    descriptions: dict[str, str] = {}

    for py_file in sorted(strategy_dir.glob("*.py")):
        stem = py_file.stem
        if stem in _STRATEGY_SKIP_MODULES:
            continue
        module_name = f"energy_modelling.strategy.{stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj is not Strategy
                and issubclass(obj, Strategy)
                and not inspect.isabstract(obj)
                and obj.__module__ == module_name
            ):
                display = _class_display_name(obj)
                descriptions[display] = _class_description(obj)

                def _make_factory(cls: type[Strategy]) -> Callable[[MarketEnvironment], Strategy]:
                    sig = inspect.signature(cls.__init__)
                    params = list(sig.parameters.keys())
                    has_market = "market" in params
                    has_settlement = "settlement_prices" in params

                    def factory(market: MarketEnvironment) -> Strategy:
                        if has_settlement:
                            return cls(market.settlement_prices)  # type: ignore[call-arg]
                        if has_market:
                            return cls(market)  # type: ignore[call-arg]
                        return cls()  # type: ignore[call-arg]

                    return factory

                factories[display] = _make_factory(obj)

    return factories, descriptions


_STRATEGY_FACTORIES, _STRATEGY_DESCRIPTIONS = _discover_strategies()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the single-strategy backtest tab."""

    # --- Controls (rendered inside the tab) ---------------------------------
    col_path, col_strat = st.columns([2, 1])
    with col_path:
        dataset_path = st.text_input(
            "Dataset path",
            value=str(_DATASET_DEFAULT),
            help="Path to dataset_de_lu.csv",
            key="bt_dataset",
        )
    with col_strat:
        strategy_name = st.selectbox(
            "Strategy",
            options=list(_STRATEGY_FACTORIES.keys()),
            key="bt_strategy",
        )

    col_s, col_e, col_btn = st.columns([1, 1, 1])
    with col_s:
        start_date = st.date_input("Start date", value=date(2024, 1, 1), key="bt_start")
    with col_e:
        end_date = st.date_input("End date", value=date(2025, 12, 31), key="bt_end")
    with col_btn:
        st.markdown("")  # spacer
        run_btn = st.button("Run Backtest", type="primary", key="bt_run")

    if strategy_name in _STRATEGY_DESCRIPTIONS:
        st.caption(_STRATEGY_DESCRIPTIONS[strategy_name])

    if not run_btn:
        st.info("Configure parameters above and click **Run Backtest**.")
        return

    path = Path(dataset_path)
    if not path.exists():
        st.error(f"Dataset not found: {path}")
        return

    # --- Execute ------------------------------------------------------------
    with st.spinner("Running backtest..."):
        market = MarketEnvironment(dataset_path=path, start_date=start_date, end_date=end_date)
        strategy = _STRATEGY_FACTORIES[strategy_name](market)
        runner = BacktestRunner(market, strategy)
        result = runner.run()
        metrics = compute_metrics(result)

    if not result.settlements:
        st.warning("No trades were executed in the selected date range.")
        return

    # --- Metrics cards ------------------------------------------------------
    st.header("Performance Summary")
    render_metric_cards(metrics)

    # --- Cumulative PnL -----------------------------------------------------
    st.header("Cumulative PnL")
    cum_df = pd.DataFrame(
        {"Date": result.cumulative_pnl.index, "PnL": result.cumulative_pnl.values}
    )
    fig = px.line(cum_df, x="Date", y="PnL", title="Cumulative Profit & Loss")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # --- Drawdown -----------------------------------------------------------
    st.header("Drawdown")
    dd = result.cumulative_pnl.cummax() - result.cumulative_pnl
    dd_df = pd.DataFrame({"Date": dd.index, "Drawdown": dd.values})
    fig = px.area(dd_df, x="Date", y="Drawdown", title="Drawdown from Peak")
    fig.update_traces(fillcolor="rgba(255,0,0,0.2)", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # --- Daily PnL histogram ------------------------------------------------
    st.header("Daily PnL Distribution")
    fig = px.histogram(
        result.daily_pnl,
        nbins=50,
        title="Distribution of Daily PnL",
        labels={"value": "Daily PnL (EUR)", "count": "Frequency"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # --- Monthly heatmap ----------------------------------------------------
    st.header("Monthly PnL")
    st.plotly_chart(
        monthly_pnl_heatmap(result.daily_pnl, title="Monthly PnL Heatmap"), use_container_width=True
    )

    # --- Rolling Sharpe -----------------------------------------------------
    st.header("Rolling Sharpe Ratio (30-day)")
    rs = rolling_sharpe(result, window=30)
    if len(rs.dropna()) > 0:
        rs_df = pd.DataFrame({"Date": rs.index, "Rolling Sharpe": rs.values})
        fig = px.line(rs_df.dropna(), x="Date", y="Rolling Sharpe", title="30-Day Rolling Sharpe")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # --- Raw metrics table --------------------------------------------------
    st.header("All Metrics")
    st.dataframe(
        pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()]),
        use_container_width=True,
        hide_index=True,
    )
