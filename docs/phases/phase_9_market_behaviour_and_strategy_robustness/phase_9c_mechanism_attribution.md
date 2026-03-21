# Phase 9c: Mechanism Attribution

## Status: PLANNED

## Objective

Determine which parts of the market mechanism are responsible for each observed
behaviour in the current synthetic futures market.

## Target Mechanisms

- sign-based trading rule
- total-profit scoring across the full window
- positive-profit truncation
- linear weight normalisation
- weighted-average forecast aggregation
- EMA dampening via `ema_alpha`
- initialization from `last_settlement_price`

## Key Questions

1. Which mechanism is necessary for active-strategy collapse?
2. Which mechanism creates or amplifies oscillation?
3. Does dampening reduce oscillation by changing the attractor, or only by
   slowing movement?
4. How sensitive are outcomes to initialization versus update dynamics?

## Planned Methods

- `ema_alpha` sweeps under the current strategy pool
- initialization counterfactuals
- strategy-family ablations
- extreme-day masking
- top-cluster removal experiments

## Checklist

- [ ] Define mechanism-isolation experiments
- [ ] Run `ema_alpha` sensitivity sweep
- [ ] Run initialization sensitivity tests
- [ ] Run strategy-family ablations
- [ ] Summarise which mechanisms explain which behaviours
