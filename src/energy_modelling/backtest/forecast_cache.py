"""SQLite-based forecast cache for strategy predictions.

Stores per-strategy forecasts in a database so that the expensive
``fit()`` + ``forecast()`` cycle only happens once per strategy version.
Subsequent runs of ``recompute-all`` can skip model training entirely
and jump straight to the market engine.

Schema
------
- **forecasts_2024**: strategy_name TEXT, delivery_date TEXT, forecast REAL
- **forecasts_2025**: strategy_name TEXT, delivery_date TEXT, forecast REAL
- **backtest_results_2024**: strategy_name TEXT, predictions BLOB, pnl BLOB, metrics BLOB
- **backtest_results_2025**: strategy_name TEXT, predictions BLOB, pnl BLOB, metrics BLOB
- **metadata**: strategy_name TEXT PK, source_hash TEXT, data_hash TEXT,
  created_at TEXT, python_version TEXT, row_count INT
"""

from __future__ import annotations

import hashlib
import pickle
import platform
import sqlite3
from datetime import date, datetime
from pathlib import Path

from energy_modelling.backtest.runner import BacktestResult

_DB_DIR = Path("data/results")
_DB_NAME = "forecast_cache.db"


def _db_path() -> Path:
    """Return the path to the forecast cache database."""
    return _DB_DIR / _DB_NAME


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (or create) the cache database."""
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS forecasts_2024 (
            strategy_name TEXT NOT NULL,
            delivery_date TEXT NOT NULL,
            forecast REAL NOT NULL,
            PRIMARY KEY (strategy_name, delivery_date)
        );
        CREATE TABLE IF NOT EXISTS forecasts_2025 (
            strategy_name TEXT NOT NULL,
            delivery_date TEXT NOT NULL,
            forecast REAL NOT NULL,
            PRIMARY KEY (strategy_name, delivery_date)
        );
        CREATE TABLE IF NOT EXISTS backtest_results_2024 (
            strategy_name TEXT PRIMARY KEY,
            result_blob BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS backtest_results_2025 (
            strategy_name TEXT PRIMARY KEY,
            result_blob BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metadata (
            strategy_name TEXT NOT NULL,
            year INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            data_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            python_version TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (strategy_name, year)
        );
        """
    )


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def _hash_strategy_source(strategy_name: str, strategies_dir: Path) -> str:
    """SHA-256 of the strategy's source file.

    For strategies that depend on base classes (ml_base, ensemble_base),
    we include those too so that changes to shared code invalidate the cache.
    """
    h = hashlib.sha256()
    # Always include the base modules
    for base in ("ml_base.py", "ensemble_base.py"):
        base_path = strategies_dir / base
        if base_path.exists():
            h.update(base_path.read_bytes())
    # Include the strategy's own file (use _class_to_file mapping)
    for py in sorted(strategies_dir.glob("*.py")):
        if py.name.startswith("__"):
            continue
        # We read strategy source content and include the strategy name in hash
        # to detect renames.  For simplicity, hash ALL strategy files
        # (cheap — ~74 small files).
        h.update(py.read_bytes())
    return h.hexdigest()


def _hash_data(csv_path: Path) -> str:
    """SHA-256 of the dataset CSV."""
    if csv_path.exists():
        h = hashlib.sha256()
        h.update(csv_path.read_bytes())
        return h.hexdigest()
    return "MISSING"


# ---------------------------------------------------------------------------
# Cache read/write
# ---------------------------------------------------------------------------


def _forecast_table(year: int) -> str:
    """Return the forecast table name for a given year."""
    if year == 2024:
        return "forecasts_2024"
    if year == 2025:
        return "forecasts_2025"
    msg = f"Unsupported year {year}; only 2024 and 2025 are cached."
    raise ValueError(msg)


def _backtest_table(year: int) -> str:
    """Return the backtest results table name for a given year."""
    if year == 2024:
        return "backtest_results_2024"
    if year == 2025:
        return "backtest_results_2025"
    msg = f"Unsupported year {year}; only 2024 and 2025 are cached."
    raise ValueError(msg)


def is_cached(
    strategy_name: str,
    year: int,
    strategies_dir: Path,
    csv_path: Path,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Check whether a strategy's forecasts are cached and up-to-date."""
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        row = conn.execute(
            "SELECT source_hash, data_hash FROM metadata WHERE strategy_name=? AND year=?",
            (strategy_name, year),
        ).fetchone()
        if row is None:
            return False
        src_hash = _hash_strategy_source(strategy_name, strategies_dir)
        dat_hash = _hash_data(csv_path)
        return row[0] == src_hash and row[1] == dat_hash
    finally:
        if own_conn:
            conn.close()


def store_forecasts(
    strategy_name: str,
    year: int,
    forecasts: dict,
    backtest_result: BacktestResult,
    strategies_dir: Path,
    csv_path: Path,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Write a strategy's forecasts and backtest result into the cache."""
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        table = _forecast_table(year)
        bt_table = _backtest_table(year)

        # Delete old data for this strategy
        conn.execute(f"DELETE FROM {table} WHERE strategy_name=?", (strategy_name,))  # noqa: S608
        conn.execute(
            f"DELETE FROM {bt_table} WHERE strategy_name=?",  # noqa: S608
            (strategy_name,),
        )

        # Insert forecasts
        rows = [(strategy_name, str(d), float(f)) for d, f in forecasts.items()]
        conn.executemany(
            f"INSERT INTO {table} (strategy_name, delivery_date, forecast) VALUES (?, ?, ?)",
            rows,
        )

        # Insert backtest result (pickled)
        result_blob = pickle.dumps(backtest_result)
        conn.execute(
            f"INSERT INTO {bt_table} (strategy_name, result_blob) VALUES (?, ?)",
            (strategy_name, result_blob),
        )

        # Update metadata
        conn.execute(
            "DELETE FROM metadata WHERE strategy_name=? AND year=?",
            (strategy_name, year),
        )
        conn.execute(
            "INSERT INTO metadata (strategy_name, year, source_hash, data_hash, "
            "created_at, python_version, row_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                strategy_name,
                year,
                _hash_strategy_source(strategy_name, strategies_dir),
                _hash_data(csv_path),
                datetime.now().isoformat(),
                platform.python_version(),
                len(forecasts),
            ),
        )
        conn.commit()
    finally:
        if own_conn:
            conn.close()


def load_forecasts(
    strategy_name: str,
    year: int,
    conn: sqlite3.Connection | None = None,
) -> dict[date, float] | None:
    """Load cached forecasts for a strategy.

    Returns ``{date: float}`` or ``None`` if not cached.
    """
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        table = _forecast_table(year)
        rows = conn.execute(
            f"SELECT delivery_date, forecast FROM {table} WHERE strategy_name=?",  # noqa: S608
            (strategy_name,),
        ).fetchall()
        if not rows:
            return None
        return {date.fromisoformat(r[0]): r[1] for r in rows}
    finally:
        if own_conn:
            conn.close()


def load_backtest_result(
    strategy_name: str,
    year: int,
    conn: sqlite3.Connection | None = None,
) -> BacktestResult | None:
    """Load cached backtest result for a strategy.

    Returns ``BacktestResult`` or ``None`` if not cached.
    """
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        bt_table = _backtest_table(year)
        row = conn.execute(
            f"SELECT result_blob FROM {bt_table} WHERE strategy_name=?",  # noqa: S608
            (strategy_name,),
        ).fetchone()
        if row is None:
            return None
        return pickle.loads(row[0])  # noqa: S301
    finally:
        if own_conn:
            conn.close()


def load_all_forecasts(
    year: int,
    conn: sqlite3.Connection | None = None,
) -> dict[str, dict[date, float]]:
    """Load all cached forecasts for a year.

    Returns ``{strategy_name: {date: float}}``.
    """
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        table = _forecast_table(year)
        rows = conn.execute(
            f"SELECT strategy_name, delivery_date, forecast FROM {table}",  # noqa: S608
        ).fetchall()
        result: dict[str, dict[date, float]] = {}
        for name, d, f in rows:
            if name not in result:
                result[name] = {}
            result[name][date.fromisoformat(d)] = f
        return result
    finally:
        if own_conn:
            conn.close()


def load_all_backtest_results(
    year: int,
    conn: sqlite3.Connection | None = None,
) -> dict[str, BacktestResult]:
    """Load all cached backtest results for a year.

    Returns ``{strategy_name: BacktestResult}``.
    """
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        bt_table = _backtest_table(year)
        rows = conn.execute(
            f"SELECT strategy_name, result_blob FROM {bt_table}",  # noqa: S608
        ).fetchall()
        return {name: pickle.loads(blob) for name, blob in rows}  # noqa: S301
    finally:
        if own_conn:
            conn.close()


def get_metadata(
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """Return all metadata rows as a list of dicts."""
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        rows = conn.execute(
            "SELECT strategy_name, year, source_hash, data_hash, "
            "created_at, python_version, row_count FROM metadata"
        ).fetchall()
        return [
            {
                "strategy_name": r[0],
                "year": r[1],
                "source_hash": r[2],
                "data_hash": r[3],
                "created_at": r[4],
                "python_version": r[5],
                "row_count": r[6],
            }
            for r in rows
        ]
    finally:
        if own_conn:
            conn.close()


def clear_cache(conn: sqlite3.Connection | None = None) -> None:
    """Remove all cached data."""
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        conn.execute("DELETE FROM forecasts_2024")
        conn.execute("DELETE FROM forecasts_2025")
        conn.execute("DELETE FROM backtest_results_2024")
        conn.execute("DELETE FROM backtest_results_2025")
        conn.execute("DELETE FROM metadata")
        conn.commit()
    finally:
        if own_conn:
            conn.close()


def remove_strategy(
    strategy_name: str,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Remove a single strategy from the cache."""
    own_conn = conn is None
    if own_conn:
        conn = _connect()
    try:
        for table in (
            "forecasts_2024",
            "forecasts_2025",
            "backtest_results_2024",
            "backtest_results_2025",
        ):
            conn.execute(
                f"DELETE FROM {table} WHERE strategy_name=?",  # noqa: S608
                (strategy_name,),
            )
        conn.execute("DELETE FROM metadata WHERE strategy_name=?", (strategy_name,))
        conn.commit()
    finally:
        if own_conn:
            conn.close()
