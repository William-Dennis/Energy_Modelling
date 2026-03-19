# Phase 2: Deepen EDA Analysis

## Status: NOT STARTED

## Objective

Add richer EDA sections to `_eda.py` that go beyond surface-level descriptive stats.
Focus on analyses that reveal patterns useful for trading strategy development.
Each new section should naturally lead to a testable hypothesis (feeding Phase 3).

## Prerequisites

- Phase 0 complete (consolidated codebase)
- Phase 1 complete (audit findings and prioritized gap list)

## Checklist

### New EDA sections to add (priority order from Phase 1 audit)

*(Section list to be finalized after Phase 1 audit. Candidate sections below.)*

- [ ] Price Change Distribution — histogram of daily `settlement - last_settlement`, skewness, kurtosis, fat tails
- [ ] Autocorrelation Analysis — ACF/PACF of price changes, test for mean reversion vs momentum at various lags
- [ ] Forecast Error Analysis — scatter/histogram of DA forecast vs actual for load/wind/solar, systematic bias detection
- [ ] Lagged Feature Importance — correlation of D-1 features with D price direction, mutual information scores
- [ ] Regime Detection — rolling volatility, HMM or threshold-based high/low price regimes, regime-conditional returns
- [ ] Volatility Clustering — GARCH-like analysis, rolling std, time-of-year effects on volatility
- [ ] Seasonal Decomposition — STL decomposition of price, identify trend/seasonal/residual components
- [ ] Cross-Border Flow Impact — how do net imports from FR/NL correlate with price direction changes?

### Implementation approach
- [ ] Each section is a private function `_render_section_name()` called from `render()`
- [ ] Follow existing pattern: expander + charts + stat cards
- [ ] All new sections added AFTER existing 12 (non-breaking)
- [ ] Write unit tests for any pure computation functions extracted

### Testing
- [ ] Extract pure computation functions (no Streamlit dependency) for testability
- [ ] Write tests for each computation function
- [ ] Verify dashboard renders without errors

## Architecture Notes

- New sections follow the existing pattern in `_eda.py`
- Pure computation logic extracted into testable functions
- Streamlit rendering kept separate from data analysis
- All analyses operate on the hourly parquet dataset (`data/processed/dataset_de_lu.parquet`)

## Sections Added

*(To be filled in during execution)*

| Section | Lines | Tests | Hypothesis Generated |
|---------|-------|-------|---------------------|
| | | | |
