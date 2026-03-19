# Phase 3: Hypothesis Checkpoints

## Status: NOT STARTED

## Objective

Identify natural hypotheses that emerge from the EDA analysis (Phases 1-2) and
translate them into concrete, testable trading strategy ideas. Each hypothesis
should have a clear signal, expected edge, and implementation approach.

## Prerequisites

- Phase 2 complete (deepened EDA with pattern discoveries)

## Checklist

### 3a. Extract hypotheses from EDA
- [ ] List all patterns discovered in EDA that suggest predictable price direction
- [ ] For each pattern, state the hypothesis formally
- [ ] Estimate the expected edge (based on EDA statistics)
- [ ] Identify the feature(s) driving the signal

### 3b. Prioritize by expected Sharpe
- [ ] Rank hypotheses by signal strength (from EDA correlation/importance analysis)
- [ ] Consider implementation complexity
- [ ] Consider robustness (does it hold across years?)
- [ ] Select top candidates for Phase 4 implementation

### 3c. Define strategy specifications
- [ ] For each selected hypothesis, define:
  - Entry signal: what triggers long/short/skip?
  - Features used: which columns from `ChallengeState.features`?
  - Parameters: any thresholds or lookback windows?
  - Expected behavior: when should it perform well/poorly?

## Hypothesis Template

```
### H[N]: [Short Name]

**Pattern observed**: [What the EDA showed]
**Hypothesis**: [Formal testable statement]
**Signal**: [How to translate to +1/-1/None]
**Features used**: [Column names from challenge data]
**Expected edge**: [Estimated win rate or Sharpe from EDA]
**Robustness**: [Does it hold across train years?]
**Implementation complexity**: [Low/Medium/High]
**Priority**: [1-5, where 1 is highest]
```

## Hypotheses

*(To be filled in during execution)*
