"""Tab 3 -- Challenge Leaderboard (Yesterday-Settlement pricing).

Evaluates all ``ChallengeStrategy`` submissions on the public 2024 and
hidden 2025 datasets using yesterday's settlement price as the entry
price.  This is the baseline view that students interact with.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from energy_modelling.dashboard import (
    class_display_name as _class_display_name,
    monthly_pnl_heatmap,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from energy_modelling.challenge.data import build_feature_glossary, write_challenge_data
from energy_modelling.challenge.runner import ChallengeBacktestResult, run_challenge_backtest
from energy_modelling.challenge.scoring import leaderboard_score
from energy_modelling.challenge.types import ChallengeStrategy

_DATASET_DEFAULT = Path("data/challenge/daily_public.csv")
_HIDDEN_DATASET_DEFAULT = Path("data/challenge/daily_hidden_test_full.csv")
_SOURCE_DATASET_DEFAULT = Path("kaggle_upload/dataset_de_lu.csv")
_SUBMISSION_SKIP_MODULES = frozenset({"__init__", "common", "perfect_foresight"})


# ---------------------------------------------------------------------------
# Strategy discovery
# ---------------------------------------------------------------------------


def _class_description(cls: type) -> str:
    doc = cls.__doc__ or ""
    return " ".join(doc.strip().split("\n\n")[0].split())


def _discover_submission_strategies() -> tuple[
    dict[str, Callable[[], ChallengeStrategy]],
    dict[str, str],
]:
    import strategies as _pkg

    sub_dir = Path(_pkg.__file__).parent
    factories: dict[str, Callable[[], ChallengeStrategy]] = {}
    descriptions: dict[str, str] = {}

    for py_file in sorted(sub_dir.glob("*.py")):
        stem = py_file.stem
        if stem in _SUBMISSION_SKIP_MODULES:
            continue
        module_name = f"strategies.{stem}"
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


STRATEGY_FACTORIES, STRATEGY_DESCRIPTIONS = _discover_submission_strategies()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _resolve_path(p: Path | str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _ensure_datasets(public: Path, hidden: Path) -> tuple[Path, Path | None]:
    if public.exists() and hidden.exists():
        return public, hidden
    source = _resolve_path(_SOURCE_DATASET_DEFAULT)
    if not source.exists():
        return public, hidden if hidden.exists() else None
    write_challenge_data(source, public.parent, include_hidden_test=True)
    return public, hidden if hidden.exists() else None


def load_daily(path: Path | str) -> pd.DataFrame:
    daily = pd.read_csv(path, parse_dates=["delivery_date"])
    daily["delivery_date"] = daily["delivery_date"].dt.date
    return daily


def combine_public_hidden(
    public: pd.DataFrame,
    hidden: pd.DataFrame | None,
) -> pd.DataFrame:
    if hidden is None:
        return public.copy()
    combined = pd.concat([public, hidden], ignore_index=True)
    combined["delivery_date"] = pd.to_datetime(combined["delivery_date"]).dt.date
    return combined.sort_values("delivery_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_strategy(
    factory: Callable[[], ChallengeStrategy],
    daily: pd.DataFrame,
    training_end: date,
    eval_start: date,
    eval_end: date,
) -> ChallengeBacktestResult:
    return run_challenge_backtest(factory(), daily, training_end, eval_start, eval_end)


def evaluate_all(
    selected: list[str],
    public: pd.DataFrame,
    hidden: pd.DataFrame | None,
) -> tuple[dict[str, ChallengeBacktestResult], dict[str, ChallengeBacktestResult]]:
    combined = combine_public_hidden(public, hidden)
    val: dict[str, ChallengeBacktestResult] = {}
    hid: dict[str, ChallengeBacktestResult] = {}

    for name in selected:
        factory = STRATEGY_FACTORIES[name]
        val[name] = evaluate_strategy(
            factory, public, date(2023, 12, 31), date(2024, 1, 1), date(2024, 12, 31)
        )
        if hidden is not None:
            hid[name] = evaluate_strategy(
                factory, combined, date(2024, 12, 31), date(2025, 1, 1), date(2025, 12, 31)
            )

    return val, hid


def leaderboard_frame(results: dict[str, ChallengeBacktestResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        m = r.metrics
        rows.append(
            {
                "Strategy": name,
                "Total PnL": m["total_pnl"],
                "Sharpe": m["sharpe_ratio"],
                "Max Drawdown": m["max_drawdown"],
                "Trades": m["trade_count"],
                "Win Rate": m["win_rate"],
                "Score": leaderboard_score(m),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["Total PnL", "Sharpe", "Max Drawdown"], ascending=[False, False, True]
    )


def _fmt_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["Score"]).assign(
        **{
            "Total PnL": lambda df: df["Total PnL"].map(lambda v: f"EUR {v:,.2f}"),
            "Sharpe": lambda df: df["Sharpe"].map(lambda v: f"{v:.2f}"),
            "Max Drawdown": lambda df: df["Max Drawdown"].map(lambda v: f"EUR {v:,.2f}"),
            "Trades": lambda df: df["Trades"].map(lambda v: f"{v:.0f}"),
            "Win Rate": lambda df: df["Win Rate"].map(lambda v: f"{v:.1%}"),
        }
    )


def _comparison_frame(
    results_by_period: dict[str, dict[str, ChallengeBacktestResult]],
    attr: str,
) -> pd.DataFrame:
    rows = []
    for period, strats in results_by_period.items():
        for name, r in strats.items():
            for d, v in getattr(r, attr).items():
                rows.append({"Date": d, "Strategy": name, "Period": period, "Value": v})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the challenge-leaderboard tab."""

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        pub_path = st.text_input("2024 public dataset", value=str(_DATASET_DEFAULT), key="ch_pub")
    with col2:
        hid_path = st.text_input(
            "2025 hidden dataset", value=str(_HIDDEN_DATASET_DEFAULT), key="ch_hid"
        )

    selected = st.multiselect(
        "Strategies",
        options=list(STRATEGY_FACTORIES),
        default=list(STRATEGY_FACTORIES),
        key="ch_strats",
    )
    run_btn = st.button("Run Comparison", type="primary", key="ch_run")

    if not selected:
        st.info("Select at least one strategy.")
        return
    if not run_btn:
        st.info("Choose strategies and click **Run Comparison**.")
        return

    pub = _resolve_path(pub_path)
    hid = _resolve_path(hid_path)
    pub, hid_or_none = _ensure_datasets(pub, hid)

    if not pub.exists():
        st.error(f"Public dataset not found: {pub}")
        return

    public_daily = load_daily(pub)
    hidden_daily = load_daily(hid_or_none) if hid_or_none else None
    glossary = build_feature_glossary(public_daily)

    with st.spinner("Evaluating submissions on yesterday-settlement pricing..."):
        try:
            val_results, hid_results = evaluate_all(selected, public_daily, hidden_daily)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Evaluation failed: {exc}")
            return

    if not val_results:
        st.error("No strategies completed successfully.")
        return

    # Store results in session state so the Market & Accuracy tabs can reuse them
    st.session_state["challenge_val_results"] = val_results
    st.session_state["challenge_hid_results"] = hid_results
    st.session_state["challenge_selected"] = selected
    st.session_state["challenge_public_daily"] = public_daily
    st.session_state["challenge_hidden_daily"] = hidden_daily

    lb_2024 = leaderboard_frame(val_results)
    lb_2025 = leaderboard_frame(hid_results) if hid_results else pd.DataFrame()

    # --- Period summary ---------------------------------------------------------
    st.header("Period Summary")
    _render_period_summary(val_results, hid_results)

    # --- Feature timing ---------------------------------------------------------
    st.header("Feature Timing")
    _render_feature_timing(glossary)

    # --- Leaderboards -----------------------------------------------------------
    tabs = st.tabs(["2024 Validation", "2025 Hidden Test"])
    with tabs[0]:
        st.subheader("2024 Leaderboard (Yesterday-Settlement Price)")
        st.dataframe(_fmt_leaderboard(lb_2024), use_container_width=True)
    with tabs[1]:
        if lb_2025.empty:
            st.info("Hidden 2025 data not available.")
        else:
            st.subheader("2025 Leaderboard (Yesterday-Settlement Price)")
            st.dataframe(_fmt_leaderboard(lb_2025), use_container_width=True)

    # --- Cumulative PnL ---------------------------------------------------------
    st.header("Cumulative PnL Comparison")
    cum_df = _comparison_frame({"2024": val_results, "2025": hid_results}, "cumulative_pnl")
    if not cum_df.empty:
        fig = px.line(
            cum_df,
            x="Date",
            y="Value",
            color="Strategy",
            line_dash="Period",
            title="Cumulative PnL",
            labels={"Value": "Cumulative PnL"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

    # --- Drawdown ---------------------------------------------------------------
    st.header("Drawdown Comparison")
    dd_payload: dict[str, dict[str, ChallengeBacktestResult]] = {"2024": {}, "2025": {}}
    for period, results in [("2024", val_results), ("2025", hid_results)]:
        for name, r in results.items():
            dd_payload[period][name] = ChallengeBacktestResult(
                predictions=r.predictions,
                daily_pnl=r.daily_pnl,
                cumulative_pnl=r.cumulative_pnl.cummax() - r.cumulative_pnl,
                trade_count=r.trade_count,
                days_evaluated=r.days_evaluated,
                metrics=r.metrics,
            )
    dd_df = _comparison_frame(dd_payload, "cumulative_pnl")
    if not dd_df.empty:
        fig = px.line(
            dd_df,
            x="Date",
            y="Value",
            color="Strategy",
            line_dash="Period",
            title="Drawdown",
            labels={"Value": "Drawdown"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Strategy detail --------------------------------------------------------
    st.header("Strategy Detail")
    focus = st.selectbox("Inspect strategy", options=list(val_results.keys()), key="ch_focus")
    dtabs = st.tabs(["2024 Detail", "2025 Detail"])
    with dtabs[0]:
        _render_detail(focus, "2024", val_results.get(focus))
    with dtabs[1]:
        _render_detail(focus, "2025", hid_results.get(focus))


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_period_summary(
    val: dict[str, ChallengeBacktestResult],
    hid: dict[str, ChallengeBacktestResult],
) -> None:
    rows = []
    for name in sorted(set(val) | set(hid)):
        v, h = val.get(name), hid.get(name)
        rows.append(
            {
                "Strategy": name,
                "2024 PnL": v.metrics["total_pnl"] if v else float("nan"),
                "2024 Sharpe": v.metrics["sharpe_ratio"] if v else float("nan"),
                "2024 Max DD": v.metrics["max_drawdown"] if v else float("nan"),
                "2025 PnL": h.metrics["total_pnl"] if h else float("nan"),
                "2025 Sharpe": h.metrics["sharpe_ratio"] if h else float("nan"),
                "2025 Max DD": h.metrics["max_drawdown"] if h else float("nan"),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["2024 PnL", "2025 PnL"], ascending=[False, False])

    def _safe(v: float, fmt: str) -> str:
        return "-" if pd.isna(v) else fmt.format(v)

    st.dataframe(
        frame.assign(
            **{
                "2024 PnL": lambda df: df["2024 PnL"].map(lambda v: f"EUR {v:,.2f}"),
                "2024 Sharpe": lambda df: df["2024 Sharpe"].map(lambda v: f"{v:.2f}"),
                "2024 Max DD": lambda df: df["2024 Max DD"].map(lambda v: f"EUR {v:,.2f}"),
                "2025 PnL": lambda df: df["2025 PnL"].map(lambda v: _safe(v, "EUR {0:,.2f}")),
                "2025 Sharpe": lambda df: df["2025 Sharpe"].map(lambda v: _safe(v, "{0:.2f}")),
                "2025 Max DD": lambda df: df["2025 Max DD"].map(lambda v: _safe(v, "EUR {0:,.2f}")),
            }
        ),
        use_container_width=True,
    )


def _render_feature_timing(glossary: pd.DataFrame) -> None:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(
            glossary.groupby("timing_group").size().reset_index(name="columns"),
            use_container_width=True,
        )
    with c2:
        t1, t2 = st.tabs(["Lagged Realised", "Same-Day Forecast"])
        with t1:
            st.dataframe(
                glossary[glossary["timing_group"] == "lagged_realised"][["column", "description"]],
                use_container_width=True,
            )
        with t2:
            st.dataframe(
                glossary[glossary["timing_group"] == "same_day_forecast"][
                    ["column", "description"]
                ],
                use_container_width=True,
            )


def _render_detail(
    strategy_name: str,
    period: str,
    result: ChallengeBacktestResult | None,
) -> None:
    if result is None:
        st.info(f"No {period} result available.")
        return

    m = result.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total PnL", f"EUR {m['total_pnl']:,.0f}")
    c2.metric("Sharpe", f"{m['sharpe_ratio']:.2f}")
    c3.metric("Max Drawdown", f"EUR {m['max_drawdown']:,.0f}")
    c4.metric("Trades", f"{m['trade_count']:.0f}")
    c5.metric("Win Rate", f"{m['win_rate']:.1%}")

    cl, cr = st.columns(2)
    with cl:
        fig = px.histogram(
            result.daily_pnl,
            nbins=40,
            title=f"Daily PnL - {strategy_name} ({period})",
            labels={"value": "Daily PnL", "count": "Frequency"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        st.plotly_chart(
            monthly_pnl_heatmap(
                result.daily_pnl, title=f"Monthly PnL - {strategy_name} ({period})"
            ),
            use_container_width=True,
        )

    st.subheader(f"Latest Predictions - {period}")
    st.dataframe(
        pd.DataFrame(
            {
                "prediction": result.predictions,
                "daily_pnl": result.daily_pnl,
                "cumulative_pnl": result.cumulative_pnl,
            }
        ).tail(20),
        use_container_width=True,
    )
