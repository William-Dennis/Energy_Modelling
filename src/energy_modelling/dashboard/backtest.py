"""Streamlit dashboard for backtest analysis.

Provides interactive visualisation of strategy backtest results
including cumulative PnL curves, drawdown charts, monthly heatmaps,
and key performance metrics.

Run with::

    streamlit run src/energy_modelling/dashboard/backtest.py
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
import plotly.graph_objects as go
import streamlit as st

from energy_modelling.dashboard import class_display_name as _class_display_name
from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.strategy.analysis import compute_metrics, monthly_pnl, rolling_sharpe
from energy_modelling.strategy.base import Strategy
from energy_modelling.strategy.runner import BacktestRunner

_DATASET_DEFAULT = Path("kaggle_upload/dataset_de_lu.csv")

# Files in the strategy package that are not user strategies.
_STRATEGY_SKIP_MODULES = frozenset({"__init__", "base", "runner", "analysis", "template"})


def _class_description(cls: type) -> str:
    """Extract the Description section from a strategy module docstring.

    Returns the paragraph of text immediately following a ``Description``
    heading (``Description\\n-----------``) in the module's docstring, or
    falls back to the class docstring's first paragraph.
    """
    module_doc = inspect.getmodule(cls).__doc__ or ""
    # Look for the Description section in the module docstring.
    m = re.search(
        r"Description\s*\n[-]+\s*\n(.*?)(?:\n\s*\n|\Z)",
        module_doc,
        re.DOTALL,
    )
    if m:
        # Collapse internal newlines and strip leading/trailing whitespace.
        return " ".join(m.group(1).split())
    # Fall back to the class docstring first paragraph.
    class_doc = cls.__doc__ or ""
    first_para = class_doc.strip().split("\n\n")[0]
    return " ".join(first_para.split())


def _discover_strategies() -> tuple[
    dict[str, Callable[[MarketEnvironment], Strategy]],
    dict[str, str],
]:
    """Auto-discover all concrete Strategy subclasses in the strategy package.

    Scans every ``.py`` file in the ``energy_modelling.strategy`` package,
    imports it, and collects concrete (non-abstract) subclasses of
    :class:`~energy_modelling.strategy.base.Strategy`.

    Files listed in ``_STRATEGY_SKIP_MODULES`` (infrastructure modules) and
    ``template.py`` are excluded so scaffold files never appear in the
    dashboard.

    Returns
    -------
    tuple of:
    - dict mapping display name → factory callable (MarketEnvironment → Strategy)
    - dict mapping display name → human-readable description string
    """
    import energy_modelling.strategy as _strategy_pkg

    strategy_dir = Path(_strategy_pkg.__file__).parent
    factories: dict[str, Callable[[MarketEnvironment], Strategy]] = {}
    descriptions: dict[str, str] = {}

    for py_file in sorted(strategy_dir.glob("*.py")):
        module_stem = py_file.stem
        if module_stem in _STRATEGY_SKIP_MODULES:
            continue

        module_name = f"energy_modelling.strategy.{module_stem}"
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

                # Build a factory that captures the class in the closure.
                def _make_factory(cls: type[Strategy]) -> Callable[[MarketEnvironment], Strategy]:
                    # Inspect the constructor to decide how to call it.
                    sig = inspect.signature(cls.__init__)
                    params = list(sig.parameters.keys())  # includes 'self'
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


# Build registries once at module load time so Streamlit doesn't re-scan on
# every widget interaction.
_STRATEGY_FACTORIES, _STRATEGY_DESCRIPTIONS = _discover_strategies()


def main() -> None:
    """Entry point for the backtest dashboard."""
    st.set_page_config(
        page_title="Backtest – DE-LU Power Futures",
        page_icon="⚡",
        layout="wide",
    )
    st.title("German Base Day Power Future – Backtest Dashboard")
    st.markdown(
        "Evaluate trading strategies on the **DE-LU day-ahead auction** "
        "using the EEX-style financially settled base-day power future."
    )

    # --- Sidebar controls ---------------------------------------------------
    st.sidebar.header("Configuration")

    dataset_path = st.sidebar.text_input(
        "Dataset path",
        value=str(_DATASET_DEFAULT),
        help="Path to dataset_de_lu.csv",
    )

    strategy_name = st.sidebar.selectbox(
        "Strategy",
        options=list(_STRATEGY_FACTORIES.keys()),
    )

    col_start, col_end = st.sidebar.columns(2)
    start_date = col_start.date_input(
        "Start date",
        value=date(2024, 1, 1),
        help="First delivery day (inclusive)",
    )
    end_date = col_end.date_input(
        "End date",
        value=date(2025, 12, 31),
        help="Last delivery day (inclusive)",
    )

    run_btn = st.sidebar.button("Run Backtest", type="primary")

    # Show strategy description beneath the selector
    if strategy_name in _STRATEGY_DESCRIPTIONS:
        st.sidebar.caption(_STRATEGY_DESCRIPTIONS[strategy_name])

    if not run_btn:
        st.info("Configure parameters in the sidebar and click **Run Backtest**.")
        return

    # --- Run backtest -------------------------------------------------------
    path = Path(dataset_path)
    if not path.exists():
        st.error(f"Dataset not found: {path}")
        return

    with st.spinner("Running backtest…"):
        market = MarketEnvironment(
            dataset_path=path,
            start_date=start_date,
            end_date=end_date,
        )
        strategy = _STRATEGY_FACTORIES[strategy_name](market)
        runner = BacktestRunner(market, strategy)
        result = runner.run()
        metrics = compute_metrics(result)

    if len(result.settlements) == 0:
        st.warning("No trades were executed in the selected date range.")
        return

    # --- Key metrics cards --------------------------------------------------
    st.header("Performance Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total PnL", f"€{metrics['total_pnl']:,.0f}")
    c2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    c3.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    c4.metric("Max Drawdown", f"€{metrics['max_drawdown']:,.0f}")
    c5.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    c6.metric("Trading Days", f"{metrics['num_trading_days']:.0f}")

    # Secondary metrics
    c7, c8, c9, c10 = st.columns(4)
    c7.metric("Avg Win", f"€{metrics['avg_win']:,.1f}")
    c8.metric("Avg Loss", f"€{metrics['avg_loss']:,.1f}")
    c9.metric("Best Day", f"€{metrics['best_day']:,.1f}")
    c10.metric("Worst Day", f"€{metrics['worst_day']:,.1f}")

    # --- Cumulative PnL curve -----------------------------------------------
    st.header("Cumulative PnL")
    cum_df = pd.DataFrame(
        {
            "Date": result.cumulative_pnl.index,
            "Cumulative PnL (EUR)": result.cumulative_pnl.values,
        }
    )
    fig_cum = px.line(
        cum_df,
        x="Date",
        y="Cumulative PnL (EUR)",
        title="Cumulative Profit & Loss",
    )
    fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_cum, use_container_width=True)

    # --- Drawdown chart -----------------------------------------------------
    st.header("Drawdown")
    running_max = result.cumulative_pnl.cummax()
    drawdown = running_max - result.cumulative_pnl
    dd_df = pd.DataFrame(
        {
            "Date": drawdown.index,
            "Drawdown (EUR)": drawdown.values,
        }
    )
    fig_dd = px.area(
        dd_df,
        x="Date",
        y="Drawdown (EUR)",
        title="Drawdown from Peak",
    )
    fig_dd.update_traces(fillcolor="rgba(255, 0, 0, 0.2)", line_color="red")
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Daily PnL distribution ---------------------------------------------
    st.header("Daily PnL Distribution")
    fig_hist = px.histogram(
        result.daily_pnl,
        nbins=50,
        title="Distribution of Daily PnL",
        labels={"value": "Daily PnL (EUR)", "count": "Frequency"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Monthly PnL heatmap ------------------------------------------------
    st.header("Monthly PnL")
    mpnl = monthly_pnl(result)
    if not mpnl.empty:
        pivot = mpnl.pivot(index="year", columns="month", values="pnl").fillna(0)
        pivot.columns = [
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][
                m - 1
            ]
            for m in pivot.columns
        ]
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index],
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"€{v:,.0f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                hovertemplate="Month: %{x}<br>Year: %{y}<br>PnL: %{text}<extra></extra>",
            )
        )
        fig_heat.update_layout(title="Monthly PnL Heatmap", yaxis_autorange="reversed")
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Rolling Sharpe -----------------------------------------------------
    st.header("Rolling Sharpe Ratio (30-day)")
    rs = rolling_sharpe(result, window=30)
    if len(rs.dropna()) > 0:
        rs_df = pd.DataFrame(
            {
                "Date": rs.index,
                "Rolling Sharpe": rs.values,
            }
        )
        fig_rs = px.line(
            rs_df.dropna(),
            x="Date",
            y="Rolling Sharpe",
            title="30-Day Rolling Sharpe Ratio",
        )
        fig_rs.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_rs, use_container_width=True)

    # --- Raw metrics table --------------------------------------------------
    st.header("All Metrics")
    metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
