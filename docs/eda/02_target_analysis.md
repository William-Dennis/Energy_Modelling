# EDA Part 2: Target Variable Analysis

This document analyses the target variable (`target_direction`, `price_change_eur_mwh`, and associated PnL columns) in depth. Understanding the statistical properties of what you are trying to predict is a prerequisite for designing effective strategies.

---

## 2.1 The Target: Price Direction

Each day the strategy must decide whether the settlement price will **rise (+1)** or **fall (−1)** relative to the prior day's settlement. Skipping (`None`) is also allowed. The three columns that describe the outcome are:

| Column | Type | Description |
|--------|------|-------------|
| `price_change_eur_mwh` | Continuous | Settlement − last_settlement (EUR/MWh) |
| `target_direction` | Binary | Sign of price_change (+1 or −1) |
| `pnl_long_eur` | Continuous | PnL for a 1 MW long position = price_change × 24 |
| `pnl_short_eur` | Continuous | PnL for a 1 MW short position = −price_change × 24 |

---

## 2.2 Direction Distribution (Training Set, 2019–2023)

| Direction | Count | Share |
|-----------|-------|-------|
| **−1** (price fell) | 972 | **53.2%** |
| **+1** (price rose) | 854 | **46.8%** |

The market has a **slight downward bias** over the training period. This is consistent with an overall energy-crisis peak in 2022 followed by unwinding in 2023. A naive strategy that always goes short would be correct 53.2% of the time on training data — but as we will show, the Sharpe and PnL of the always-short strategy are modest, because the large up-moves (2021–2022 crisis) can severely damage any mechanical short bias.

---

## 2.3 Price Change Distribution (Training Set)

```
count  1826
mean   −0.01  EUR/MWh
std    36.17  EUR/MWh
min  −209.41  EUR/MWh   (largest single-day fall)
25%   −10.84  EUR/MWh
50%    −0.99  EUR/MWh
75%    10.05  EUR/MWh
max   258.17  EUR/MWh   (largest single-day rise)
```

Key observations:
- The **median** is close to zero (−0.99 EUR/MWh), confirming the near-random nature of day-to-day moves.
- The distribution is **roughly symmetric** but with fat tails — extreme moves of ±100 EUR/MWh are not rare.
- **Asymmetry in extremes:** The largest daily fall (−209 EUR/MWh) is 20% larger in absolute terms than the largest rise (258 EUR/MWh). This asymmetry is due to 2022 energy-crisis volatility.

In PnL terms, a single large wrong-way day can erase weeks of small gains. This makes **risk management** (Sharpe ratio and max drawdown) as important as hit rate.

---

## 2.4 Year-by-Year Price Volatility

| Year | Mean Change | Std Dev | Min | Max |
|------|-------------|---------|-----|-----|
| 2019 | +0.01 | 11.63 | −73.42 | +63.17 |
| 2020 | +0.04 | 11.48 | −34.09 | +53.62 |
| 2021 | −0.09 | 33.91 | −139.84 | +213.71 |
| 2022 | −0.04 | 65.06 | −209.41 | +258.17 |
| 2023 | +0.03 | 30.17 | −108.25 | +118.75 |

**2021–2022 is a distinct regime.** Volatility in 2022 was more than **5× higher** than in 2019–2020. Any model trained only on 2019–2020 data would be severely underprepared for the energy crisis. Conversely, a model over-fit to 2022 dynamics (extreme gas and carbon price moves) may perform poorly in calmer conditions.

This has important implications for feature engineering:
- Consider **normalising** price-level features (e.g. price deviation from a rolling mean) rather than using raw price levels.
- Apply **year-aware cross-validation** to avoid leaking 2022 patterns into earlier-year folds.

---

## 2.5 Settlement Price Levels

Over all splits (train + validation):

```
mean   95.65 EUR/MWh
std    92.43 EUR/MWh
min   −50.13 EUR/MWh   (negative prices occurred)
25%    38.40 EUR/MWh
50%    66.86 EUR/MWh
75%   113.41 EUR/MWh
max   705.36 EUR/MWh   (peak energy crisis)
```

**Average settlement by year (training):**

| Year | Avg Settlement (EUR/MWh) |
|------|--------------------------|
| 2019 | 37.67 |
| 2020 | 30.47 |
| 2021 | 96.86 |
| 2022 | 235.44 |
| 2023 | 95.18 |

The five-year average masks a dramatic 8× range. Students should be careful with any features that are correlated with the absolute price level (e.g. `last_settlement_price`) without de-trending.

---

## 2.6 Seasonal Patterns in Direction (Monthly)

The share of up-days by month over the training period:

| Month | Up Days | Down Days | % Up |
|-------|---------|-----------|------|
| Jan | 73 | 82 | 47.1% |
| Feb | 66 | 75 | 46.8% |
| Mar | 76 | 79 | 49.0% |
| Apr | 68 | 82 | 45.3% |
| May | 70 | 85 | 45.2% |
| Jun | 73 | 77 | 48.7% |
| Jul | 73 | 82 | 47.1% |
| Aug | 75 | 80 | 48.4% |
| Sep | 65 | 85 | 43.3% |
| Oct | 71 | 84 | 45.8% |
| **Nov** | **81** | **69** | **54.0%** |
| Dec | 63 | 92 | 40.6% |

- **November** is the only month with a meaningfully upward bias (54%) — likely driven by rising heating demand and gas storage drawdown.
- **September and December** show the strongest downward bias (~43% and ~41% up respectively).
- Most months cluster between 45–49%, confirming the near-random nature of the monthly signal in isolation.

---

## 2.7 Day-of-Week Effect

The most striking pattern in the target variable is a very strong **day-of-week (DOW) effect**:

| Day | Up | Down | % Up |
|-----|----|------|------|
| **Monday** | 235 | 25 | **90.4%** |
| Tuesday | 160 | 101 | 61.3% |
| Wednesday | 131 | 130 | 50.2% |
| Thursday | 123 | 138 | 47.1% |
| Friday | 101 | 160 | 38.7% |
| **Saturday** | 34 | 227 | **13.0%** |
| **Sunday** | 70 | 191 | **26.8%** |

This is arguably the single most powerful signal in the entire dataset. Monday settlement prices rise 90.4% of the time; Saturday prices fall 87.0% of the time.

**Economic explanation:** Day-ahead settlement prices for weekend delivery (Saturday and Sunday) reflect low industrial and commercial demand, depressing prices. Prices typically recover sharply for Monday delivery because workday demand resumes. The Monday contract price rises relative to the Sunday contract price the vast majority of the time.

**Quick verification code:**

```python
train['dow'] = train['delivery_date'].dt.day_of_week
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_stats = (
    train.groupby('dow')['target_direction']
    .value_counts(normalize=True)
    .unstack()
    .rename(columns={1: 'pct_up', -1: 'pct_down'})
)
dow_stats.index = day_names
print(dow_stats.round(3))
```

---

## 2.8 Quarterly Pattern

| Quarter | Up Days | % Up | Sum PnL (always long) |
|---------|---------|------|-----------------------|
| Q1 | 224/451 | 47.7% | +6,945 EUR |
| Q2 | 211/455 | 46.4% | +3,578 EUR |
| Q3 | 213/460 | 46.3% | +289 EUR |
| Q4 | 215/460 | 46.7% | −11,291 EUR |

Quarterly variation is modest and the "always long" strategy produces near-zero PnL across all quarters. The day-of-week effect dominates any seasonal quarterly pattern.

---

## 2.9 Baseline Strategy Benchmarks

To contextualise any new signal, compare against these baselines:

| Strategy | Training PnL | Training Sharpe |
|----------|-------------|-----------------|
| Always Long | −478 EUR | −0.006 |
| Always Short | +478 EUR | +0.006 |
| Random (50/50) | ~0 EUR | ~0 |
| **DOW Strategy\*** | **+471,218 EUR** | **+6.59** |

\*DOW strategy: long on Mon/Tue, short on Fri/Sat/Sun, skip Wed/Thu.

The near-zero PnL of always-long and always-short confirms the overall directional balance. The DOW strategy dramatically outperforms because it exploits the structural weekend/weekday price pattern.

---

## 2.10 What "Good" EDA Looks Like for the Target

A thorough target analysis should:

1. ✅ Report the direction class balance (53/47 split)
2. ✅ Show the price change distribution (mean ≈ 0, std ≈ 36 EUR/MWh, fat tails)
3. ✅ Identify the year-on-year volatility regime shift (2021–2022)
4. ✅ Decompose direction by month, day-of-week, and year
5. ✅ Compute the PnL and Sharpe of simple baselines
6. ✅ Note that the median daily PnL per trade is close to zero — higher hit rate alone is insufficient; managing the size of right/wrong moves matters

---

**Next:** [Part 3 — Feature Analysis](03_feature_analysis.md)
