# EDA Part 5: Scoring Rubric

This document defines the ideal EDA submission for students working on the DE-LU Day-Ahead Power Futures challenge. It serves as the reference answer against which student work is graded.

---

## 5.1 Purpose of the EDA

The EDA is a structured investigation of the training dataset (`daily_public.csv`, 2019–2023) that aims to:

1. Characterise the target variable and understand its statistical properties
2. Identify features with genuine predictive signal
3. Test hypotheses about economic relationships in energy markets
4. Translate findings into concrete, implementable trading ideas
5. Validate each idea against the out-of-sample validation set (2024)

A well-executed EDA is the foundation of any defensible strategy submission.

---

## 5.2 Scoring Dimensions

Student EDA submissions are scored across five dimensions. The total is 100 points.

---

### Dimension 1: Data Understanding (20 points)

**What we are looking for:**

| Criteria | Max Points | Key Indicators |
|----------|------------|----------------|
| Correctly identifies all column groups and timing constraints | 5 | Labels `timing_group` for all columns; flags that label columns are unavailable at inference time |
| Correctly characterises the dataset size, splits, and date ranges | 3 | Reports 1,826 train rows (2019–2023), 366 val rows (2024) |
| Identifies and handles missing values appropriately | 4 | Lists columns with missing values; proposes a valid imputation strategy |
| Notes structural breaks in the data | 5 | Identifies nuclear phase-out in 2023; identifies 2021–2022 energy crisis regime |
| Demonstrates valid code for loading and inspecting the dataset | 3 | Runnable Python that reads the CSV and reports basic statistics |

**Common mistakes (point deductions):**
- Using label columns as input features (−5 points)
- Ignoring the timing constraint (lagged vs. same-day data) (−3 points)
- Reporting statistics on the full dataset without splitting train/val (−2 points)

---

### Dimension 2: Target Variable Analysis (25 points)

**What we are looking for:**

| Criteria | Max Points | Key Indicators |
|----------|------------|----------------|
| Reports direction class balance (~53% down, ~47% up) | 3 | Correct figures; notes slight negative class imbalance |
| Describes price change distribution (mean, std, skew, fat tails) | 5 | Reports mean ≈ 0, std ≈ 36, max ≈ 258, min ≈ −209 EUR/MWh |
| Analyses seasonality by month | 4 | Reports November upward bias; December/September downward bias |
| Identifies and quantifies the **day-of-week effect** | 8 | Monday ~90% positive; Saturday ~13% positive — explicit table or chart |
| Analyses year-on-year volatility regime | 5 | Notes 5× volatility increase in 2022 vs. 2019–2020 |

**Ideal output:** A table or chart showing the percentage of positive days by day of week, clearly demonstrating the Monday/Saturday asymmetry.

**Common mistakes:**
- Missing the DOW effect entirely (−6 points)
- Conflating price *level* with price *change* (−3 points)
- Failing to note the 2021–2022 volatility regime (−3 points)

---

### Dimension 3: Feature Analysis and Signal Identification (30 points)

**What we are looking for:**

| Criteria | Max Points | Key Indicators |
|----------|------------|----------------|
| Computes feature-target correlations for all or most features | 6 | Presents a correlation table or bar chart sorted by magnitude |
| Identifies load forecast as the strongest individual feature (+0.235) | 4 | Correct sign and approximate magnitude reported |
| Identifies wind forecast as a key inverse signal (−0.19 to −0.21) | 5 | Notes that higher wind forecast → lower prices |
| Derives the net demand signal and shows it outperforms individual features | 5 | Reports correlation ~+0.30 for `load − wind_onshore − wind_offshore` |
| Analyses the DE-FR spread as a mean-reversion signal | 3 | Reports correct sign (negative spread → positive bias) |
| Notes that gas/carbon *level* has near-zero signal; proposes trend instead | 3 | Distinguishes level vs. trend; tests gas_trend_3d or similar |
| Notes low signal from solar, momentum, and nuclear (post-2023) | 4 | Explicit statement that these are weak or low-correlation signals with supporting evidence |

**Ideal output:** A ranked table of all features by correlation magnitude with direction, plus a section on derived features (net demand, price vs. MA, spreads).

**Common mistakes:**
- Treating lagged realised wind as a direct short signal (confuses lagged with forecast wind) (−3 points)
- Using solar forecast as a strong signal without evidence (−2 points)
- Overfitting correlation analysis to 2022 data only (−3 points)

---

### Dimension 4: Strategy Ideas and Validation (15 points)

**What we are looking for:**

| Criteria | Max Points | Key Indicators |
|----------|------------|----------------|
| Proposes at least two testable signal-based strategies | 4 | e.g. DOW strategy, wind contrarian, net-demand threshold |
| Backtests at least one strategy on the training set and reports PnL + Sharpe | 5 | Uses `pnl_long_eur` and `pnl_short_eur` columns correctly; does not look ahead |
| Validates at least one strategy on the 2024 validation set | 4 | Reports out-of-sample results separately from training results |
| Notes the danger of overfitting (using all data for both fit and evaluation) | 2 | Explicit statement about train/val split discipline |

**Ideal output:** A table comparing simple strategies on both training and validation sets, with PnL and Sharpe ratio for each.

**Common mistakes:**
- Evaluating strategy on the training set only (−4 points)
- Using the validation set to *select* the strategy and then reporting validation performance as "out-of-sample" (−3 points)
- Reporting accuracy (hit rate) without Sharpe ratio or PnL (−2 points)

---

### Dimension 5: Communication and Depth (10 points)

**What we are looking for:**

| Criteria | Max Points | Key Indicators |
|----------|------------|----------------|
| Clear narrative structure (overview → target → features → signals → conclusions) | 3 | Reader can follow the logic from data loading through to actionable ideas |
| Economic intuition for observed patterns | 4 | Explains *why* the DOW effect exists; why wind suppresses prices; why gas trend matters |
| Reproducible code | 3 | All claims supported by runnable code that produces the reported figures |

**Common mistakes:**
- Presenting charts without explanatory text (−2 points)
- No code or non-runnable code (−2 points)
- Patterns stated without economic justification (−1 point each)

---

## 5.3 Score Bands

| Score | Grade | Interpretation |
|-------|-------|----------------|
| 90–100 | Distinction | Identified all key signals including DOW and net demand; validated strategies out-of-sample; clear economic justification throughout |
| 75–89 | Merit | Identified most signals; reasonable backtest; minor gaps in regime analysis or out-of-sample validation |
| 60–74 | Pass | Identified at least 2 significant signals (e.g. wind + load); produced a working backtest; some gaps in analysis |
| 45–59 | Marginal Fail | Only identified 1 key signal; backtest is in-sample only or contains look-ahead; limited code |
| <45 | Fail | Missed the DOW effect; no backtest; used label columns as features; minimal analysis |

---

## 5.4 Ideal EDA Structure

A full-mark EDA should contain these sections in order:

```
1. Dataset Overview
   1.1 Shape, splits, date ranges
   1.2 Column groups and timing constraints
   1.3 Missing data handling

2. Target Variable Analysis
   2.1 Direction class balance
   2.2 Price change distribution (histogram, stats)
   2.3 Day-of-week analysis (table + chart)
   2.4 Monthly seasonality
   2.5 Year-on-year volatility regime

3. Feature Analysis
   3.1 Feature-target correlation table (all features)
   3.2 Generation feature deep-dive (nuclear phase-out)
   3.3 Forecast feature analysis (load, wind, solar)
   3.4 Price spread analysis (DE-FR, price vs MA)
   3.5 Commodity price signals (gas trend, carbon trend)
   3.6 Cross-border flow signals

4. Signal Extraction
   4.1 Net demand signal (derived, ~+0.30 correlation)
   4.2 DOW strategy backtest (train + val)
   4.3 Wind contrarian backtest (train + val)
   4.4 Signal combination approach
   4.5 Low-signal features (what doesn't work and why)

5. Conclusions
   5.1 Top 5 actionable signals ranked by strength
   5.2 Proposed strategy direction
   5.3 Remaining research questions
```

---

## 5.5 Reference Answer: Key Facts

The following facts should appear in any passing EDA:

| Fact | Value | Where Found |
|------|-------|-------------|
| Training rows | 1,826 | `df[df['split']=='train'].shape` |
| Validation rows | 366 | `df[df['split']=='validation'].shape` |
| Direction balance (train) | 53.2% down / 46.8% up | `target_direction.value_counts()` |
| Monday positive rate | 90.4% | DOW analysis |
| Saturday positive rate | 13.0% | DOW analysis |
| Load forecast correlation | +0.235 | Feature correlations |
| Wind offshore forecast correlation | −0.212 | Feature correlations |
| Net demand correlation | ~+0.30 | Derived feature |
| 2022 price std | 65.06 EUR/MWh | Year-by-year analysis |
| Nuclear zero rows | 259 (all 2023) | Missing/structural analysis |
| DOW strategy train Sharpe | ~6.6 | Strategy backtest |
| DOW strategy 2024 PnL | +93,213 EUR | Out-of-sample validation |

---

## 5.6 Bonus Points

Up to 10 bonus points are available for exceptional work:

| Achievement | Points |
|-------------|--------|
| Implements a combined signal model (DOW + wind + load) that clearly beats both baselines | +3 |
| Implements look-ahead-free cross-validation correctly | +2 |
| Tests a simple ML model (logistic regression, random forest) with proper train/val split | +3 |
| Analyses the impact of the 2022 energy crisis on signal stability | +2 |

Maximum total score (including bonus) is capped at 100.

---

**End of EDA reference documents.**

← Back to [Part 4 — Signal Extraction](04_signal_extraction.md) | [Challenge Brief](../hackathon_backtest.md)
