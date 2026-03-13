"""Challenge backtest runner for public validation and private scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from energy_modelling.challenge.scoring import compute_challenge_metrics
from energy_modelling.challenge.types import ChallengeState, ChallengeStrategy

_STATE_EXCLUDE_COLUMNS = {
    "delivery_date",
    "split",
    "settlement_price",
    "price_change_eur_mwh",
    "target_direction",
    "pnl_long_eur",
    "pnl_short_eur",
}


@dataclass(frozen=True)
class ChallengeBacktestResult:
    """Results from a challenge evaluation run."""

    predictions: pd.Series
    daily_pnl: pd.Series
    cumulative_pnl: pd.Series
    trade_count: int
    days_evaluated: int
    metrics: dict[str, float]


def _normalise_daily_data(daily_data: pd.DataFrame) -> pd.DataFrame:
    if "delivery_date" in daily_data.columns:
        normalised = daily_data.copy()
        normalised["delivery_date"] = pd.to_datetime(normalised["delivery_date"]).dt.date
        normalised = normalised.set_index("delivery_date", drop=False)
    else:
        normalised = daily_data.copy()
        normalised.index = pd.Index(pd.to_datetime(normalised.index).date, name="delivery_date")
        normalised["delivery_date"] = normalised.index

    return normalised.sort_index()


def _feature_row(row: pd.Series) -> pd.Series:
    return row.drop(labels=list(_STATE_EXCLUDE_COLUMNS), errors="ignore").copy()


def _validate_prediction(prediction: int | None, delivery_date: date) -> None:
    if prediction not in (-1, 1, None):
        msg = (
            f"Strategy returned invalid prediction {prediction!r} for {delivery_date}. "
            "Expected +1, -1, or None."
        )
        raise ValueError(msg)


def run_challenge_backtest(
    strategy: ChallengeStrategy,
    daily_data: pd.DataFrame,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
) -> ChallengeBacktestResult:
    """Fit a strategy, then evaluate it one day at a time."""

    data = _normalise_daily_data(daily_data)
    train_mask = data.index <= training_end
    eval_mask = (data.index >= evaluation_start) & (data.index <= evaluation_end)

    train_data = data.loc[train_mask].copy()
    evaluation_data = data.loc[eval_mask].copy()
    if evaluation_data.empty:
        msg = "No evaluation rows found for the requested date range."
        raise ValueError(msg)

    strategy.fit(train_data)
    strategy.reset()

    predictions: list[int | None] = []
    pnl_values: list[float] = []
    eval_dates: list[date] = []

    for delivery_date, row in evaluation_data.iterrows():
        state = ChallengeState(
            delivery_date=delivery_date,
            last_settlement_price=float(row["last_settlement_price"]),
            features=_feature_row(row),
            history=data.loc[data.index < delivery_date].copy(),
        )
        prediction = strategy.act(state)
        _validate_prediction(prediction, delivery_date)

        price_change = float(row["settlement_price"] - row["last_settlement_price"])
        pnl = 0.0 if prediction is None else price_change * float(prediction) * 24.0

        predictions.append(prediction)
        pnl_values.append(pnl)
        eval_dates.append(delivery_date)

    prediction_series = pd.Series(predictions, index=eval_dates, name="prediction", dtype="Int64")
    daily_pnl = pd.Series(pnl_values, index=eval_dates, name="pnl", dtype=float)
    cumulative_pnl = daily_pnl.cumsum()
    trade_count = int(prediction_series.notna().sum())
    metrics = compute_challenge_metrics(daily_pnl, trade_count)

    return ChallengeBacktestResult(
        predictions=prediction_series,
        daily_pnl=daily_pnl,
        cumulative_pnl=cumulative_pnl,
        trade_count=trade_count,
        days_evaluated=len(daily_pnl),
        metrics=metrics,
    )
