# Phase 10f: Strategy Robustness Analysis

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

- [ ] Build standalone vs market-adjusted PnL comparison table for all 67 strategies:
  - standalone total PnL, standalone rank
  - market-adjusted total PnL, market-adjusted rank
  - rank change (delta)
- [ ] Define and compute market-contribution metrics:
  - MAE improvement when strategy is included vs excluded (leave-one-out)
  - weight stability across iterations (low variance = more stable)
  - correlation with the strategy's closest cluster peer (redundancy measure)
- [ ] Classify each strategy into one of:
  - **robust**: strong standalone + strong market-adjusted + positive market contribution
  - **standalone-only**: strong standalone but weak market-adjusted
  - **redundant**: similar forecasts to another strategy already in the pool
  - **destabilising**: negative market contribution (MAE worsens when included)
- [ ] Identify the top-5 most robust and top-5 most problematic strategies
- [ ] Write `scripts/phase10f_strategy_robustness.py`
- [ ] Save results to `data/results/phase10/strategy_robustness.csv`
- [ ] Produce a shortlist of keep / revise / replace recommendations for Phase 10g
