"""Shared utilities for ENTSO-E data collection modules."""

from __future__ import annotations

import pandas as pd


def year_range(year: int, timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) timestamps for a calendar year in the given timezone.

    ``end`` is set to Jan 1 of the *next* year so that the ENTSO-E API returns
    data up to and including Dec 31 23:00.
    """
    start = pd.Timestamp(f"{year}-01-01", tz=timezone)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=timezone)
    return start, end


def normalise_name(name: str) -> str:
    """Convert a raw column name string to snake_case."""
    return (
        name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "").replace("/", "_")
    )
