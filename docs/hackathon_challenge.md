# DE-LU Futures Hackathon Challenge

This challenge asks students to design a trading strategy for the **DE-LU day-ahead power futures market** using only daily data available at decision time.

## Objective

Build a `StudentStrategy` class that decides each day whether to:

- go `long` (`+1`)
- go `short` (`-1`)
- or `skip` (`None`)

The strategy is evaluated on out-of-sample profit and loss.

## Data Access Rules

- Students receive only the **daily public dataset**.
- Public training period: **2019-2023**
- Public validation period: **2024**
- Final hidden test period: **2025**
- At decision time for day `t`, the strategy may use:
  - the current row's lagged realised features for day `t`
  - the current row's day-ahead forecast features for day `t`
  - all historical rows before day `t`
  - realised settlements up to `t-1`
- The strategy may not use the true settlement for day `t` before acting.

## Scoring

Leaderboard ranking is:

1. highest `total_pnl`
2. highest `sharpe_ratio`
3. lowest `max_drawdown`

PnL is computed as:

```text
PnL = (settlement_price - last_settlement_price) * direction * 24
```

Quantity is fixed at `1 MW`, so students only choose direction.

## Student Submission Contract

Students edit `submission/student_strategy.py` and implement:

```python
class StudentStrategy:
    def fit(self, train_data):
        ...

    def act(self, state):
        return 1, -1, or None
```

The baseline implementation is the naive-copy strategy and always returns `1`.

## Public Data Build

Generate the daily public dataset with:

```bash
uv run build-challenge-data
```

This writes:

- `data/challenge/daily_public.csv`
- `data/challenge/daily_public_glossary.csv`

For organizer-only private scoring, add `--include-hidden-test`.

## Important Packaging Note

Do not distribute the full hourly repository dataset or the existing dashboards to students, because the repo contains 2025 data and open exploration tools.

For the actual hackathon release, distribute only:

- `data/challenge/daily_public.csv`
- `data/challenge/daily_public_glossary.csv`
- `submission/student_strategy.py`
- `notebooks/hackathon_baseline.ipynb`
- this challenge brief
