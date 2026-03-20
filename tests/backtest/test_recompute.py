"""Tests for the recompute module."""

from __future__ import annotations

from datetime import date

from energy_modelling.backtest.recompute import _parse_date, main, recompute_all


def test_parse_date_none():
    assert _parse_date(None) is None


def test_parse_date_valid():
    assert _parse_date("2024-06-15") == date(2024, 6, 15)


def test_main_is_callable():
    assert callable(main)


def test_recompute_all_is_callable():
    assert callable(recompute_all)
