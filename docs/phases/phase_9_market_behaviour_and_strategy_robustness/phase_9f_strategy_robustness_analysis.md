# Phase 9f: Strategy Robustness Analysis

## Status: PLANNED

## Objective

Distinguish strategies that are merely strong in standalone backtests from those
that remain useful inside the synthetic futures market.

## Working Definition

A robust strategy should do more than earn standalone PnL. It should also be
informative after market repricing and ideally improve the aggregate market.

## Analysis Tracks

- standalone leaderboard vs market-adjusted leaderboard
- contribution to convergence / stability
- contribution to market MAE / RMSE
- redundancy and correlation within the strategy pool
- destabilising versus stabilising strategy families

## Key Questions

1. Which strategies keep their edge after repricing?
2. Which strategies add unique information rather than cluster redundancy?
3. Which strategies help the market find better prices?
4. Which strategies should be pruned, revised, or replaced?

## Checklist

- [ ] Build standalone vs market-adjusted comparison tables
- [ ] Define market-contribution metrics
- [ ] Identify robust, redundant, and destabilising strategies
- [ ] Produce a shortlist of keep / revise / replace candidates
