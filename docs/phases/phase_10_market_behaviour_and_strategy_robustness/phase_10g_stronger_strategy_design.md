# Phase 10g: Stronger Strategy Design

## Status: COMPLETE

## Objective

Translate the findings from Phase 10a-10f into concrete design principles and a
prioritised backlog of stronger market-robust strategies.

## Design Rules (Derived from Phase 10 Findings)

### Rule 1: Prioritise Forecast Orthogonality Over Raw Accuracy

**Source**: Phase 10d (cluster analysis), Phase 10f (redundancy scores)

The ML regression cluster (11 strategies) produces near-identical forecasts
(pairwise correlation >0.99) yet captures >90% of market weight. Adding another
highly correlated ML strategy provides zero marginal information and may actively
destabilise the market (Phase 10f: 25 destabilising strategies in 2024).

**Guideline**: New strategies should have max forecast correlation < 0.85 with
all existing strategies, measured on the full evaluation window.

### Rule 2: Balance Long/Short Exposure

**Source**: Phase 10f (Always Short gained 56 ranks in 2025)

Strategies with persistent directional bias are vulnerable to market repricing.
In standalone backtests, long-biased strategies dominate (real prices trend up
from initial prices). But in the market, this bias is priced out, and balanced
strategies gain relative advantage.

**Guideline**: New strategies should aim for approximately 40-60% long/short
ratio across the evaluation window, rather than persistent directional bias.

### Rule 3: Design for Regime Awareness

**Source**: Phase 10d (regime analysis), Phase 10c (alpha sensitivity)

The 2024 and 2025 datasets have fundamentally different volatility structures.
Strategies that work well in low-volatility periods often fail in high-volatility
windows (and vice versa). The ML cluster is more robust to volatility, but
rule-based strategies show larger regime-dependent accuracy gaps.

**Guideline**: New strategies should have an explicit regime-detection component
or should be tested across both low-vol and high-vol subsets before inclusion.

### Rule 4: Avoid the Positive-Profit Trap

**Source**: Phase 10e (absorbing collapse narrative), Phase 10c (truncation as root cause)

The positive-profit truncation rule (w_i = max(Pi_i, 0) / sum) creates a
one-way ratchet. Strategies that underperform early in the iteration sequence
lose weight permanently and can never recover. This means a strategy needs to
be profitable not just on average, but specifically in the early iterations when
the market price is still close to the initial price.

**Guideline**: Test candidate strategies for profitability in the first 50
iterations, not just at convergence. A strategy that is profitable at convergence
but not early will never reach meaningful weight.

### Rule 5: Prefer Sparse High-Conviction Forecasts

**Source**: Phase 10e (early-accuracy-lost cases), Phase 10f (LOO analysis)

The market optimises aggregate profitability, not per-date accuracy. Strategies
that issue moderate forecasts on every date are drowned out by high-conviction
ML strategies. Strategies that issue strong forecasts only on dates where they
have genuine signal (and sit out otherwise) can achieve better risk-adjusted
market contribution.

**Guideline**: New strategies should consider issuing a "no forecast" / skip
signal on dates where confidence is low, rather than forcing a forecast on
every date.

## Signal Family Assessment

### Overcrowded Families (Reduce)

| Family | Current Count | Redundancy Level | Action |
|--------|--------------|-----------------|--------|
| ML Regression | 11 | Very High (>0.99 pairwise) | Keep top 2-3 (Stacked Ridge Meta, Ridge Net Demand, Lasso Top Features); prune rest |
| ML Classification | 6 | High (>0.95 pairwise) | Keep 1-2 (Logistic Direction is most distinct); prune rest |
| Ensemble/Meta | 12 | Moderate-High | Redesign around diversity-weighted ensembles; prune naive vote-based |

### Underrepresented Families (Expand)

| Family | Current Count | Market Contribution | Action |
|--------|--------------|-------------------|--------|
| Cross-border Spread | 6 | Positive (Multi Spread: +0.0268 LOO) | Add more cross-border/interconnector strategies |
| Supply-side | 5 | Mixed but unique | Add weather-derivative or generation-mix strategies |
| Regime-aware | 2 | Unknown (need testing) | New regime-conditional forecasters |
| Balanced / Market-neutral | 0 | High potential | New family targeting ~50% long/short ratio |

### Gap Analysis vs Strategy Registry

Comparing the 67-strategy registry against the design rules:

1. **No strategy explicitly targets balanced long/short exposure** --
   all strategies are "always forecast, always trade". There is no
   selective high-conviction strategy.

2. **No strategy incorporates market-feedback** -- none adjusts its
   forecast based on the current market price or its own weight.

3. **Cross-border signals are promising but sparse** -- only 6 strategies
   use cross-border data, yet Multi Spread has the strongest positive
   LOO contribution in 2025.

4. **Weather-only strategies are absent** -- temperature and radiation
   features are underused. Temperature Extreme exists but is a simple
   threshold; no strategy models the temperature-demand relationship
   non-linearly.

## Candidate Strategy Shortlist

### Tier 1: High Priority (implement first)

| # | Candidate | Rationale | Expected Contribution | Difficulty |
|---|-----------|-----------|----------------------|------------|
| 1 | **Diverse Ensemble v2** | Rebuild ensemble to weight sub-strategies by forecast diversity (not just accuracy); penalise correlated members | Reduce redundancy; improve market MAE | Medium |
| 2 | **Spread Momentum** | 3-day momentum on Multi Spread signal; exploits most orthogonal signal family | Unique cross-border information | Low |
| 3 | **Selective High-Conviction** | Only forecast on days where the strategy's signal exceeds 1.5 std from neutral; skip otherwise | Better signal-to-noise ratio | Low |

### Tier 2: Medium Priority

| # | Candidate | Rationale | Expected Contribution | Difficulty |
|---|-----------|-----------|----------------------|------------|
| 4 | **Temperature Curve** | Non-linear (quadratic or spline) temperature-demand model; captures heating and cooling demand asymmetries | Unique weather signal | Medium |
| 5 | **Regime-Conditional Ridge** | Ridge regression that switches feature sets based on volatility regime | Adaptive to regime shifts | Medium |
| 6 | **Market-Aware Contrarian** | If the market consensus is heavily one-directional (top-1 weight > 0.5), trade against it | Stabilising effect on market; reduces oscillation | Medium |

### Tier 3: Lower Priority (exploratory)

| # | Candidate | Rationale | Expected Contribution | Difficulty |
|---|-----------|-----------|----------------------|------------|
| 7 | **Intraday Pattern** | Exploit day-ahead vs intraday price patterns using load actual vs forecast | Orthogonal temporal signal | High |
| 8 | **Nuclear Event** | Binary strategy: forecast price spike when nuclear availability drops sharply | Rare but high-impact signal | Low |
| 9 | **Flow Imbalance** | Combined FR+NL flow imbalance signal; trade when net imports exceed historical 75th percentile | Cross-border flow dynamics | Low |
| 10 | **Pruned ML Ensemble** | Keep only top-3 uncorrelated ML strategies; ensemble with equal weight | Direct redundancy reduction | Low |

## Implementation Briefs (Top-3 Candidates)

### 1. Diverse Ensemble v2

Replace the current 12 ensemble strategies with a single diversity-aware
ensemble. The ensemble should:
- Accept forecasts from all 67 base strategies
- Compute a diversity penalty: for each pair of strategies, penalise weight
  when their forecasts are correlated > 0.9 on the training window
- Weight = accuracy_score * diversity_bonus
- Forecast = diversity-weighted average of base forecasts

Expected behaviour: reduces the effective weight of the ML regression cluster
from >90% to ~40-50%, allowing more diverse signals to contribute.

### 2. Spread Momentum

A simple rule-based strategy exploiting the Multi Spread signal:
- Compute 3-day exponential moving average of the average cross-border spread
  (DE-FR, DE-NL)
- If spread_ema is rising and above zero: long (foreign prices higher -> DE
  price has room to rise)
- If spread_ema is falling and below zero: short

This leverages the cross-border signal family, which Phase 10f identified as
the most positively contributing family to market quality.

### 3. Selective High-Conviction

A meta-strategy wrapper that applies to any base strategy:
- Compute the base strategy's forecast
- Compute the rolling z-score of the forecast distance from the current price
- Only trade when |z-score| > 1.5 (high conviction days)
- On other days, forecast = current market price (no trade)

This addresses the per-date accuracy problem identified in Phase 10e, where
the market degrades accuracy on specific dates by over-weighting strategies
that are profitable overall but poor on individual days.

## Acceptance Criteria for Next Build Phase

Any new strategy implemented in the next phase must satisfy:

1. **Standalone PnL**: positive total PnL on both 2024 and 2025 evaluation windows
2. **Market contribution**: non-negative LOO delta-MAE (removing the strategy
   should not improve market MAE)
3. **Orthogonality**: max forecast correlation < 0.90 with any existing strategy
4. **Test coverage**: at least 5 unit tests using synthetic data
5. **Balanced exposure**: long/short ratio between 35-65% across the evaluation
   window
6. **All existing tests pass**: 1000+ tests green, ruff clean, theorems verified

## Connection to Existing Tracker

The following items from `issues/issue_3_new_strategies.md` are now addressed:
- All 7 originally proposed strategies have been implemented (Phases B-E)
- The feature utilisation gap is largely closed (20 of 28 features now used)
- The net-long bias issue is documented but not yet resolved (Candidate #3
  and #6 address this directly)

New candidates 1-10 above should be added to the strategy backlog when a
Phase 11 is scoped.

## Checklist

- [x] Convert Phase 10 findings into explicit design rules
- [x] Review the expansion strategy registry for gaps
- [x] Define a shortlist of 10 candidate new strategies or revisions
- [x] Prioritise candidates using expected contribution, complexity, orthogonality
- [x] For the top-3 candidates, write a 1-paragraph implementation brief
- [x] Connect candidates to existing issue tracker
- [x] Specify acceptance criteria for the next build phase
