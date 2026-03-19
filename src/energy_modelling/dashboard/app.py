"""Consolidated Streamlit dashboard for the DE-LU energy market platform.

Launch with::

    uv run streamlit run src/energy_modelling/dashboard/app.py

Tabs
----
1. **EDA** -- Exploratory data analysis (hourly Parquet dataset).
2. **Backtest** -- Multi-strategy leaderboard (yesterday-settlement pricing).
3. **Futures Market** -- Synthetic futures market model (organizer tool).
4. **Futures Market Simulation** -- Converged market price vs real settlement.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="DE-LU Energy Market Platform",
    page_icon="⚡",
    layout="wide",
)

st.title("DE-LU Day-Ahead Electricity Market Platform")
st.caption(
    "ENTSO-E + Open-Meteo + Yahoo Finance data for the Germany-Luxembourg "
    "bidding zone, 2019-2025.  All timestamps UTC."
)

tab_eda, tab_ch, tab_mkt, tab_acc = st.tabs(
    [
        "EDA",
        "Backtest",
        "Futures Market",
        "Futures Market Simulation",
    ]
)

with tab_eda:
    from energy_modelling.dashboard._eda import render as _render_eda

    _render_eda()

with tab_ch:
    from energy_modelling.dashboard._backtest import render as _render_ch

    _render_ch()

with tab_mkt:
    from energy_modelling.dashboard._futures_market import render as _render_mkt

    _render_mkt()

with tab_acc:
    from energy_modelling.dashboard._accuracy import render as _render_acc

    _render_acc()
