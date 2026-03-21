"""Tab 3 -- Backtest Leaderboard (Yesterday-Settlement pricing)."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from energy_modelling.dashboard import class_display_name as _class_display_name

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_ROOT):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from energy_modelling.backtest.data import build_feature_glossary, write_backtest_data  # noqa: E402
from energy_modelling.backtest.runner import BacktestResult, run_backtest  # noqa: E402
from energy_modelling.backtest.scoring import leaderboard_score  # noqa: E402
from energy_modelling.backtest.types import BacktestStrategy  # noqa: E402

_DATASET_DEFAULT = Path("data/backtest/daily_public.csv")
_HIDDEN_DATASET_DEFAULT = Path("data/backtest/daily_hidden_test_full.csv")
_SOURCE_DATASET_DEFAULT = Path("kaggle_upload/dataset_de_lu.csv")
_SUBMISSION_SKIP_MODULES = frozenset({"__init__", "common", "perfect_foresight"})


def _class_description(cls: type) -> str:
    doc = cls.__doc__ or ""
    return " ".join(doc.strip().split("\n\n")[0].split())


def _discover_submission_strategies() -> tuple[
    dict[str, Callable[[], BacktestStrategy]],
    dict[str, str],
]:
    import strategies as _pkg

    sub_dir = Path(_pkg.__file__).parent
    factories: dict[str, Callable[[], BacktestStrategy]] = {}
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
                obj is not BacktestStrategy
                and issubclass(obj, BacktestStrategy)
                and not inspect.isabstract(obj)
                and obj.__module__ == module_name
            ):
                display = _class_display_name(obj)
                descriptions[display] = _class_description(obj)
                factories[display] = obj

    return factories, descriptions


STRATEGY_FACTORIES, STRATEGY_DESCRIPTIONS = _discover_submission_strategies()


def _resolve_path(p: Path | str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _ensure_datasets(public: Path, hidden: Path) -> tuple[Path, Path | None]:
    if public.exists() and hidden.exists():
        return public, hidden
    source = _resolve_path(_SOURCE_DATASET_DEFAULT)
    if not source.exists():
        return public, hidden if hidden.exists() else None
    write_backtest_data(source, public.parent, include_hidden_test=True)
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


def evaluate_strategy(
    factory: Callable[[], BacktestStrategy],
    daily: pd.DataFrame,
    training_end: date,
    eval_start: date,
    eval_end: date,
) -> BacktestResult:
    return run_backtest(factory(), daily, training_end, eval_start, eval_end)


def evaluate_all(
    selected: list[str],
    public: pd.DataFrame,
    hidden: pd.DataFrame | None,
) -> tuple[dict[str, BacktestResult], dict[str, BacktestResult]]:
    combined = combine_public_hidden(public, hidden)
    val: dict[str, BacktestResult] = {}
    hid: dict[str, BacktestResult] = {}

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


def leaderboard_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
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


def _render_controls() -> tuple[str, str, list[str], bool]:
    """Render sidebar controls and return user selections."""
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
    return pub_path, hid_path, selected, run_btn


def _render_results(
    val_results: dict[str, BacktestResult],
    hid_results: dict[str, BacktestResult],
    public_daily: pd.DataFrame | None,
) -> None:
    """Render all result sections."""
    from energy_modelling.dashboard._backtest_render import (
        _render_cumulative_pnl,
        _render_drawdown,
        _render_feature_timing,
        _render_leaderboards,
        _render_period_summary,
        _render_strategy_detail,
    )

    glossary = build_feature_glossary(public_daily) if public_daily is not None else pd.DataFrame()
    lb_2024 = leaderboard_frame(val_results)
    lb_2025 = leaderboard_frame(hid_results) if hid_results else pd.DataFrame()

    st.header("Period Summary")
    _render_period_summary(val_results, hid_results)
    if not glossary.empty:
        st.header("Feature Timing")
        _render_feature_timing(glossary)
    _render_leaderboards(lb_2024, lb_2025)
    _render_cumulative_pnl(val_results, hid_results)
    _render_drawdown(val_results, hid_results)
    _render_strategy_detail(val_results, hid_results)


def render() -> None:
    """Render the backtest-leaderboard tab."""
    pub_path, hid_path, selected, run_btn = _render_controls()
    _try_load_cached(run_btn, pub_path, hid_path)

    if not selected:
        st.info("Select at least one strategy.")
        return

    val_results = st.session_state.get("backtest_val_results")
    if not run_btn and val_results is None:
        st.info("Choose strategies and click **Run Comparison**, or pre-compute results via CLI.")
        return

    if run_btn and _run_evaluation(selected, pub_path, hid_path) is None:
        return

    val_results = st.session_state.get("backtest_val_results", {})
    hid_results = st.session_state.get("backtest_hid_results", {})
    public_daily = st.session_state.get("backtest_public_daily")
    if val_results:
        _render_results(val_results, hid_results, public_daily)


def _try_load_cached(run_btn: bool, pub_path: str, hid_path: str) -> None:
    """Load cached results from disk if available."""
    if run_btn or "backtest_val_results" in st.session_state:
        return
    from energy_modelling.backtest.io import RESULTS_DIR, load_backtest_results

    cached_val = load_backtest_results(RESULTS_DIR / "backtest_val_2024.pkl")
    cached_hid = load_backtest_results(RESULTS_DIR / "backtest_hid_2025.pkl")
    if cached_val is None:
        return
    pub = _resolve_path(pub_path)
    hid = _resolve_path(hid_path)
    pub, hid_or_none = _ensure_datasets(pub, hid)
    public_daily = load_daily(pub) if pub.exists() else None
    hidden_daily = load_daily(hid_or_none) if hid_or_none and hid_or_none.exists() else None

    st.session_state["backtest_val_results"] = cached_val
    st.session_state["backtest_hid_results"] = cached_hid if cached_hid else {}
    st.session_state["backtest_selected"] = list(cached_val.keys())
    if public_daily is not None:
        st.session_state["backtest_public_daily"] = public_daily
    if hidden_daily is not None:
        st.session_state["backtest_hidden_daily"] = hidden_daily
    st.success("Loaded pre-computed results from disk.")


def _run_evaluation(
    selected: list[str], pub_path: str, hid_path: str
) -> dict[str, BacktestResult] | None:
    """Run backtest evaluation and store results in session state."""
    pub = _resolve_path(pub_path)
    hid = _resolve_path(hid_path)
    pub, hid_or_none = _ensure_datasets(pub, hid)

    if not pub.exists():
        st.error(f"Public dataset not found: {pub}")
        return None

    public_daily = load_daily(pub)
    hidden_daily = load_daily(hid_or_none) if hid_or_none else None

    with st.spinner("Evaluating submissions on yesterday-settlement pricing..."):
        try:
            val_results, hid_results = evaluate_all(selected, public_daily, hidden_daily)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Evaluation failed: {exc}")
            return None

    if not val_results:
        st.error("No strategies completed successfully.")
        return None

    st.session_state["backtest_val_results"] = val_results
    st.session_state["backtest_hid_results"] = hid_results
    st.session_state["backtest_selected"] = selected
    st.session_state["backtest_public_daily"] = public_daily
    st.session_state["backtest_hidden_daily"] = hidden_daily

    from energy_modelling.backtest.io import RESULTS_DIR, save_backtest_results

    save_backtest_results(val_results, RESULTS_DIR / "backtest_val_2024.pkl")
    if hid_results:
        save_backtest_results(hid_results, RESULTS_DIR / "backtest_hid_2025.pkl")
    return val_results
