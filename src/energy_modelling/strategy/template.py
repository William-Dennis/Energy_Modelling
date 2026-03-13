"""Template for a new trading strategy -- rename this file and fill in the blanks.

Description
-----------
One sentence describing what this strategy bets on.

Decision rule
-------------
* If <condition>  → signal **long**  (+1)
* If <condition>  → signal **short** (-1)
* Otherwise       → skip (return *None*)

PnL formula (set by the runner, not the strategy)
--------------------------------------------------
PnL = (today's DA settlement − yesterday's DA settlement) × direction × 1 MW × 24 h

Getting started
---------------
1. Copy this file to a new name, e.g. ``my_strategy.py``.
2. Rename the class and fill in ``act()``.
3. Save the file -- the dashboard picks it up automatically on next reload.
"""

from __future__ import annotations

from energy_modelling.market_simulation.market import MarketEnvironment
from energy_modelling.market_simulation.types import DayState, Signal
from energy_modelling.strategy.base import Strategy


class TemplateStrategy(Strategy):
    """One-line description of what this strategy does.

    Parameters
    ----------
    market:
        The market environment, provided automatically by the dashboard
        factory.  Use ``market.settlement_prices`` if your strategy
        needs future price data (e.g. perfect-foresight variants).
        Leave the constructor parameter-free for signal-only strategies.
    """

    # ------------------------------------------------------------------ #
    # Optional: store anything you need across calls here.                #
    # ------------------------------------------------------------------ #
    def __init__(self, market: MarketEnvironment | None = None) -> None:  # noqa: ARG002
        pass  # remove if you don't need market data at init time

    def act(self, state: DayState) -> Signal | None:
        """Decide on a trade direction for the given delivery day.

        Available inputs via ``state``
        --------------------------------
        state.delivery_date          -- the date being traded (datetime.date)
        state.last_settlement_price  -- prior day's average DA price (EUR/MWh)
        state.features               -- single-row DataFrame of daily features:
                                        lagged realised features plus any
                                        same-day day-ahead forecast features
        state.neighbor_prices        -- dict of prior-day DA prices for
                                        neighbouring zones (FR, NL, AT, PL,
                                        CZ, DK1)

        Returns
        -------
        Signal(delivery_date, direction=+1)   -- go long
        Signal(delivery_date, direction=-1)   -- go short
        None                                  -- skip this day
        """
        # -------------------------------------------------------------- #
        # Replace the logic below with your own.                          #
        # Example: go long if the prior day's load was above 55 GW.       #
        # -------------------------------------------------------------- #
        features = state.features
        if features.empty:
            return None

        load = features.get("load_actual_mw_mean")
        if load is None:
            return None

        prior_load_mw = float(load.iloc[0])
        if prior_load_mw > 55_000:
            direction = 1  # long
        else:
            direction = -1  # short

        return Signal(delivery_date=state.delivery_date, direction=direction)

    def reset(self) -> None:
        """Reset any state accumulated between backtest runs (optional)."""
        pass  # clear counters, rolling windows, ML model state, etc.
