"""Tests for the SQLite forecast cache module."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from energy_modelling.backtest.forecast_cache import (
    _connect,
    _ensure_schema,
    _forecast_table,
    clear_cache,
    get_metadata,
    is_cached,
    load_all_backtest_results,
    load_all_forecasts,
    load_backtest_result,
    load_forecasts,
    remove_strategy,
    store_forecasts,
)
from energy_modelling.backtest.runner import BacktestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result() -> BacktestResult:
    """Create a minimal BacktestResult for testing."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    return BacktestResult(
        predictions=pd.Series([1, -1, 1], index=dates, name="prediction", dtype="Int64"),
        daily_pnl=pd.Series([10.0, -5.0, 8.0], index=dates, name="pnl"),
        cumulative_pnl=pd.Series([10.0, 5.0, 13.0], index=dates, name="pnl"),
        trade_count=3,
        days_evaluated=3,
        metrics={"total_pnl": 13.0, "sharpe": 0.5},
    )


def _make_forecasts() -> dict[date, float]:
    """Create sample forecasts for testing."""
    return {
        date(2024, 1, 1): 55.0,
        date(2024, 1, 2): 48.5,
        date(2024, 1, 3): 62.1,
    }


@pytest.fixture
def db_conn(tmp_path: Path) -> sqlite3.Connection:
    """Provide a temporary database connection."""
    conn = _connect(tmp_path / "test_cache.db")
    yield conn
    conn.close()


@pytest.fixture
def strategies_dir(tmp_path: Path) -> Path:
    """Create a fake strategies directory with a dummy file."""
    sd = tmp_path / "strategies"
    sd.mkdir()
    (sd / "__init__.py").write_text("# init")
    (sd / "my_strat.py").write_text("class MyStrat: pass")
    (sd / "ml_base.py").write_text("class Base: pass")
    (sd / "ensemble_base.py").write_text("class EnsBase: pass")
    return sd


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Create a fake CSV dataset."""
    p = tmp_path / "data.csv"
    p.write_text("date,price\n2024-01-01,50.0\n")
    return p


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchema:
    def test_tables_created(self, db_conn: sqlite3.Connection) -> None:
        tables = {
            r[0]
            for r in db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "forecasts_2024" in tables
        assert "forecasts_2025" in tables
        assert "backtest_results_2024" in tables
        assert "backtest_results_2025" in tables
        assert "metadata" in tables

    def test_schema_idempotent(self, db_conn: sqlite3.Connection) -> None:
        _ensure_schema(db_conn)  # second call should not raise
        _ensure_schema(db_conn)

    def test_forecast_table_names(self) -> None:
        assert _forecast_table(2024) == "forecasts_2024"
        assert _forecast_table(2025) == "forecasts_2025"

    def test_forecast_table_invalid_year(self) -> None:
        with pytest.raises(ValueError, match="Unsupported year"):
            _forecast_table(2023)


# ---------------------------------------------------------------------------
# Store and load tests
# ---------------------------------------------------------------------------


class TestStoreAndLoad:
    def test_store_and_load_forecasts(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        forecasts = _make_forecasts()
        result = _make_result()

        store_forecasts(
            "Test Strategy",
            2024,
            forecasts,
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )

        loaded = load_forecasts("Test Strategy", 2024, db_conn)
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[date(2024, 1, 1)] == pytest.approx(55.0)
        assert loaded[date(2024, 1, 3)] == pytest.approx(62.1)

    def test_store_and_load_backtest_result(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        forecasts = _make_forecasts()
        result = _make_result()

        store_forecasts(
            "Test Strategy",
            2024,
            forecasts,
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )

        loaded = load_backtest_result("Test Strategy", 2024, db_conn)
        assert loaded is not None
        assert loaded.trade_count == 3
        assert loaded.metrics["total_pnl"] == pytest.approx(13.0)

    def test_load_missing_strategy_returns_none(self, db_conn: sqlite3.Connection) -> None:
        assert load_forecasts("Nonexistent", 2024, db_conn) is None
        assert load_backtest_result("Nonexistent", 2024, db_conn) is None

    def test_store_overwrites_existing(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        result = _make_result()

        store_forecasts(
            "Test",
            2024,
            {date(2024, 1, 1): 50.0},
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )
        store_forecasts(
            "Test",
            2024,
            {date(2024, 1, 1): 99.0, date(2024, 1, 2): 100.0},
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )

        loaded = load_forecasts("Test", 2024, db_conn)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[date(2024, 1, 1)] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# Bulk load tests
# ---------------------------------------------------------------------------


class TestBulkLoad:
    def test_load_all_forecasts(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        result = _make_result()
        store_forecasts(
            "A",
            2024,
            {date(2024, 1, 1): 10.0},
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )
        store_forecasts(
            "B",
            2024,
            {date(2024, 1, 1): 20.0},
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )

        all_fc = load_all_forecasts(2024, db_conn)
        assert len(all_fc) == 2
        assert "A" in all_fc
        assert "B" in all_fc
        assert all_fc["A"][date(2024, 1, 1)] == pytest.approx(10.0)

    def test_load_all_backtest_results(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        result = _make_result()
        store_forecasts(
            "A",
            2024,
            _make_forecasts(),
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )
        store_forecasts(
            "B",
            2024,
            _make_forecasts(),
            result,
            strategies_dir,
            csv_path,
            db_conn,
        )

        all_res = load_all_backtest_results(2024, db_conn)
        assert len(all_res) == 2
        assert all_res["A"].trade_count == 3


# ---------------------------------------------------------------------------
# Cache validity tests
# ---------------------------------------------------------------------------


class TestCacheValidity:
    def test_is_cached_true(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        store_forecasts(
            "Test",
            2024,
            _make_forecasts(),
            _make_result(),
            strategies_dir,
            csv_path,
            db_conn,
        )
        assert is_cached("Test", 2024, strategies_dir, csv_path, db_conn) is True

    def test_is_cached_false_when_missing(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        assert is_cached("Nonexistent", 2024, strategies_dir, csv_path, db_conn) is False

    def test_cache_invalidated_by_source_change(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        store_forecasts(
            "Test",
            2024,
            _make_forecasts(),
            _make_result(),
            strategies_dir,
            csv_path,
            db_conn,
        )
        assert is_cached("Test", 2024, strategies_dir, csv_path, db_conn) is True

        # Modify a source file
        (strategies_dir / "my_strat.py").write_text("class MyStrat: changed = True")
        assert is_cached("Test", 2024, strategies_dir, csv_path, db_conn) is False

    def test_cache_invalidated_by_data_change(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        store_forecasts(
            "Test",
            2024,
            _make_forecasts(),
            _make_result(),
            strategies_dir,
            csv_path,
            db_conn,
        )
        assert is_cached("Test", 2024, strategies_dir, csv_path, db_conn) is True

        # Modify the CSV
        csv_path.write_text("date,price\n2024-01-01,99.9\n2024-01-02,55.0\n")
        assert is_cached("Test", 2024, strategies_dir, csv_path, db_conn) is False


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_stored(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        store_forecasts(
            "Test",
            2024,
            _make_forecasts(),
            _make_result(),
            strategies_dir,
            csv_path,
            db_conn,
        )

        meta = get_metadata(db_conn)
        assert len(meta) == 1
        assert meta[0]["strategy_name"] == "Test"
        assert meta[0]["year"] == 2024
        assert meta[0]["row_count"] == 3
        assert "python_version" in meta[0]
        assert "created_at" in meta[0]
        assert len(meta[0]["source_hash"]) == 64  # SHA-256

    def test_metadata_multiple_strategies(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        result = _make_result()
        store_forecasts("A", 2024, _make_forecasts(), result, strategies_dir, csv_path, db_conn)
        store_forecasts("B", 2025, _make_forecasts(), result, strategies_dir, csv_path, db_conn)

        meta = get_metadata(db_conn)
        assert len(meta) == 2
        names = {m["strategy_name"] for m in meta}
        assert names == {"A", "B"}


# ---------------------------------------------------------------------------
# Cleanup tests
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_clear_cache(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        store_forecasts(
            "Test",
            2024,
            _make_forecasts(),
            _make_result(),
            strategies_dir,
            csv_path,
            db_conn,
        )
        assert load_forecasts("Test", 2024, db_conn) is not None

        clear_cache(db_conn)
        assert load_forecasts("Test", 2024, db_conn) is None
        assert get_metadata(db_conn) == []

    def test_remove_strategy(
        self,
        db_conn: sqlite3.Connection,
        strategies_dir: Path,
        csv_path: Path,
    ) -> None:
        result = _make_result()
        store_forecasts("A", 2024, _make_forecasts(), result, strategies_dir, csv_path, db_conn)
        store_forecasts("B", 2024, _make_forecasts(), result, strategies_dir, csv_path, db_conn)

        remove_strategy("A", db_conn)

        assert load_forecasts("A", 2024, db_conn) is None
        assert load_forecasts("B", 2024, db_conn) is not None


# ---------------------------------------------------------------------------
# Integration with futures_market_runner cached_forecasts parameter
# ---------------------------------------------------------------------------


class TestRunnerCacheIntegration:
    """Test that run_futures_market_evaluation accepts cached data."""

    def test_cached_forecasts_parameter_accepted(self) -> None:
        """Verify the function signature accepts cached_forecasts and cached_results."""
        import inspect

        from energy_modelling.backtest.futures_market_runner import (
            run_futures_market_evaluation,
        )

        sig = inspect.signature(run_futures_market_evaluation)
        assert "cached_forecasts" in sig.parameters
        assert "cached_results" in sig.parameters
