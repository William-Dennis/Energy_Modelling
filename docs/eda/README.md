# EDA Reference: DE-LU Day-Ahead Power Futures Challenge

This directory contains the reference Exploratory Data Analysis (EDA) for the DE-LU Day-Ahead Power Futures hackathon challenge. These documents are intended for challenge organisers to:

1. **Set the standard** for what a high-quality student EDA looks like
2. **Score student submissions** using the rubric in Part 5
3. **Validate student hypotheses** against the quantified signal strengths here

---

## Document Index

| Document | Description |
|----------|-------------|
| [01 — Data Overview](01_data_overview.md) | Dataset structure, column groups, timing constraints, missing values |
| [02 — Target Analysis](02_target_analysis.md) | Price change distribution, direction balance, DOW effect, seasonality |
| [03 — Feature Analysis](03_feature_analysis.md) | Correlation analysis, generation/weather/load/price/flow deep-dives |
| [04 — Signal Extraction](04_signal_extraction.md) | Tradeable signals, backtest results, strategy ideas, ML guidance |
| [05 — EDA Scoring Rubric](05_eda_scoring_rubric.md) | Scoring dimensions (100 pts), grade bands, ideal EDA structure |

---

## Key Findings at a Glance

| Finding | Details |
|---------|---------|
| **Strongest signal** | Day-of-week effect (Monday 90% positive, Saturday 13% positive) |
| **Best single feature** | Load forecast (+0.235 correlation with direction) |
| **Best derived feature** | Net demand = `load − wind_onshore − wind_offshore` (~+0.30 correlation) |
| **Low-signal features** | Solar forecast, momentum, raw gas level, nuclear generation |
| **Regime break** | Nuclear phase-out (2023), energy crisis (2021–2022) |
| **DOW strategy Sharpe** | 6.59 (train) / 8.23 (validation 2024) |

---

## Related Documents

- [Challenge Brief](../hackathon_backtest.md) — Task definition, scoring rules, submission contract
- [Kaggle Dataset Description](../kaggle_description.md) — Full hourly data dictionary
- [Strategy Development Guide](../../src/energy_modelling/strategy/STRATEGIES.md) — How to implement strategies
- [Baseline Notebook](../../notebooks/hackathon_baseline.ipynb) — Student starting point
