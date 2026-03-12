"""Market environment for day-ahead power futures simulation.

Provides a :class:`MarketEnvironment` that iterates over delivery days,
constructs observable :class:`DayState` objects (enforcing information
cutoffs), and settles trades against realised day-ahead prices.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from energy_modelling.market_simulation.contract import compute_pnl, compute_settlement_price
from energy_modelling.market_simulation.data import (
    build_daily_features,
    compute_daily_settlement,
    load_dataset,
)
from energy_modelling.market_simulation.types import DayState, Settlement, Trade

# Neighbour price columns to extract for DayState.
_NEIGHBOR_PRICE_MAP: dict[str, str] = {
    "price_fr_eur_mwh": "FR",
    "price_nl_eur_mwh": "NL",
    "price_at_eur_mwh": "AT",
    "price_pl_eur_mwh": "PL",
    "price_cz_eur_mwh": "CZ",
    "price_dk_1_eur_mwh": "DK1",
}


class MarketEnvironment:
    """Simulated market for German Base Day Power Futures.

    Iterates over delivery days within the specified date range.  For
    each day, builds a :class:`DayState` containing only information
    available before the day-ahead auction close, then settles trades
    against the realised average hourly price.

    Parameters
    ----------
    dataset_path:
        Path to ``dataset_de_lu.csv``.
    start_date:
        First delivery day (inclusive).  If *None*, uses the second
        day in the dataset (the first day with a prior settlement).
    end_date:
        Last delivery day (inclusive).  If *None*, uses the last
        complete day in the dataset.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> None:
        self._df = load_dataset(dataset_path)
        self._settlement = compute_daily_settlement(self._df)
        self._features = build_daily_features(self._df)

        all_dates = sorted(self._settlement.index)

        # Default range: start at day 2 (first day with a prior settlement),
        # end at the last complete day.
        if start_date is None:
            start_date = all_dates[1] if len(all_dates) > 1 else all_dates[0]
        if end_date is None:
            end_date = all_dates[-1]

        self._start = start_date
        self._end = end_date
        self._all_dates = all_dates

    @property
    def delivery_dates(self) -> list[date]:
        """Return the list of delivery dates in the simulation range."""
        return [d for d in self._all_dates if self._start <= d <= self._end]

    @property
    def settlement_prices(self) -> pd.Series:
        """Daily settlement prices (mean of 24 hourly DA prices) for all dates.

        Returns the full series covering the entire dataset, not just the
        simulation range, so strategies such as
        :class:`~energy_modelling.strategy.perfect_foresight.PerfectForesightStrategy`
        that require look-ahead data can be constructed at the call site
        with the appropriate level of access control.
        """
        return self._settlement.copy()

    def get_state(self, delivery_date: date) -> DayState:
        """Build the observable state for a given delivery day.

        Parameters
        ----------
        delivery_date:
            The delivery day for which to construct the state.

        Returns
        -------
        DayState
            Market state containing only information available before
            the auction close for *delivery_date*.

        Raises
        ------
        ValueError
            If *delivery_date* is outside the simulation range.
        """
        if delivery_date < self._start or delivery_date > self._end:
            msg = (
                f"delivery_date {delivery_date} is outside the simulation "
                f"range [{self._start}, {self._end}]"
            )
            raise ValueError(msg)

        # Find the prior day's settlement price
        prior_dates = [d for d in self._all_dates if d < delivery_date]
        if not prior_dates:
            msg = f"No prior settlement available for {delivery_date}"
            raise ValueError(msg)
        prior_date = prior_dates[-1]
        last_settlement = float(self._settlement[prior_date])

        # Features: the row for this delivery_date (already lagged by 1 day)
        if delivery_date in self._features.index:
            features = self._features.loc[[delivery_date]]
        else:
            # If features not available, return empty DataFrame with correct columns
            features = pd.DataFrame(columns=self._features.columns, index=[delivery_date])

        # Extract neighbour prices from the prior day's hourly data
        prior_mask = self._df.index.date == prior_date
        prior_data = self._df.loc[prior_mask]
        neighbor_prices: dict[str, float] = {}
        for col, zone in _NEIGHBOR_PRICE_MAP.items():
            if col in prior_data.columns:
                val = prior_data[col].mean()
                if pd.notna(val):
                    neighbor_prices[zone] = float(val)

        return DayState(
            delivery_date=delivery_date,
            last_settlement_price=last_settlement,
            features=features,
            neighbor_prices=neighbor_prices,
        )

    def settle(self, trade: Trade) -> Settlement:
        """Settle a trade against the realised day-ahead prices.

        Parameters
        ----------
        trade:
            A trade to settle.

        Returns
        -------
        Settlement
            The settlement result including PnL.

        Raises
        ------
        ValueError
            If the trade's delivery date has no price data.
        """
        if trade.delivery_date not in self._settlement.index:
            msg = f"No price data for delivery date {trade.delivery_date}"
            raise ValueError(msg)

        # Get hourly prices for the delivery day
        day_mask = self._df.index.date == trade.delivery_date
        hourly = self._df.loc[day_mask, "price_eur_mwh"]
        settlement_price = compute_settlement_price(hourly)
        pnl = compute_pnl(trade, settlement_price)

        return Settlement(
            trade=trade,
            settlement_price=settlement_price,
            pnl=pnl,
        )
