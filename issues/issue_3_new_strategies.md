# Issue 3: Implement 5+ New Strategies Exploiting Unused Features

## Summary

Only 25% of available features (8 of 28) are used by existing strategies. Implement 5-7 new theory-backed strategies that exploit the 20+ untapped features -- solar forecasts, commodity prices, cross-border spreads, temperature, volatility regimes, and nuclear availability.

## Motivation

### Feature utilisation gap

The dataset has 28 usable feature columns. Current strategies use only 8:

| Used | Not Used |
|------|----------|
| `forecast_wind_offshore_mw_mean` | `forecast_solar_mw_mean` |
| `forecast_wind_onshore_mw_mean` | `gas_price_usd_mean` |
| `load_forecast_mw_mean` | `carbon_price_usd_mean` |
| `gen_fossil_gas_mw_mean` | `weather_temperature_2m_degc_mean` |
| `gen_fossil_hard_coal_mw_mean` | `gen_nuclear_mw_mean` |
| `gen_fossil_brown_coal_lignite_mw_mean` | `gen_solar_mw_mean` |
| `price_change_eur_mwh` (history) | `price_fr/nl/at/pl/cz/dk_1_eur_mwh_mean` |
| `delivery_date` (weekday) | `flow_fr/nl_net_import_mw_mean` |
| | `price_std`, `price_max`, `price_min` |
| | `weather_wind_speed_10m_kmh_mean` |
| | `weather_shortwave_radiation_wm2_mean` |
| | `load_actual_mw_mean` |

### Market simulation insight

The existing strategy pool is net-long biased. In the futures market equilibrium, **every "smart" strategy loses money** and AlwaysShort becomes #1. New strategies should aim for:
- Orthogonal signals (features unused by the current pool)
- Balanced long/short bias
- Signals that survive equilibrium pricing

## Strategy Specifications

### Strategy 1: SolarForecastStrategy

**Hypothesis**: High solar forecast suppresses midday clearing prices via the merit-order effect (solar is zero marginal cost). Same mechanism as WindForecastStrategy but for solar.

**Feature**: `forecast_solar_mw_mean` (same-day forecast, no lag)

**Logic**:
- `fit()`: Compute median of `forecast_solar_mw_mean` from training data
- `act()`: If solar forecast >= median -> short (-1), else -> long (+1)

**Expected behaviour**: Similar to WindForecastStrategy. Solar has a distinct seasonal profile (strong in summer, weak in winter) which may provide orthogonal signal.

### Strategy 2: CommodityCostStrategy

**Hypothesis**: Rising gas and carbon prices increase the marginal cost of gas-fired generation (the price-setting technology in most hours). Higher marginal cost -> higher clearing prices -> long.

**Features**: `gas_price_usd_mean`, `carbon_price_usd_mean` (lagged 1 day)

**Logic**:
- `fit()`: Compute a combined fuel cost index from training: `fuel_index = gas_price * gas_weight + carbon_price * carbon_weight`. Compute median. Weights can be derived from typical gas plant heat rates (7-8 MWh_th/MWh_e) and emission factors (0.37 tCO2/MWh_e).
- `act()`: If today's fuel index > median -> long (+1), else -> short (-1)

**Expected behaviour**: Should capture the commodity-driven component of price movements that existing strategies completely miss.

### Strategy 3: TemperatureExtremeStrategy

**Hypothesis**: Extreme temperatures (cold in winter, hot in summer) drive demand spikes that push clearing prices up. This is a non-linear signal -- moderate temps have no effect, only extremes matter.

**Feature**: `weather_temperature_2m_degc_mean` (lagged 1 day)

**Logic**:
- `fit()`: Compute 10th and 90th percentiles of temperature from training data
- `act()`: If temp < P10 (extreme cold) OR temp > P90 (extreme heat) -> long (+1). If P10 <= temp <= P90 (moderate) -> short (-1)

**Expected behaviour**: Selective trading. Should have high win rate on extreme days but skip many days. Consider returning None for moderate days to improve hit rate.

### Strategy 4: CrossBorderSpreadStrategy

**Hypothesis**: When yesterday's French and Dutch electricity prices were high relative to DE-LU, the DE-LU market is likely under import pressure, keeping prices elevated. European market coupling means prices converge, so a lagged spread signals directional pressure.

**Features**: `price_fr_eur_mwh_mean`, `price_nl_eur_mwh_mean`, `last_settlement_price`

**Logic**:
- `fit()`: Compute the median spread `(avg(FR, NL) - DE-LU)` from training data
- `act()`: Compute today's spread. If spread > median (neighbours expensive) -> long (+1). If spread < -median (neighbours cheap) -> short (-1). Otherwise -> skip (None)

**Expected behaviour**: Captures the cross-border price transmission mechanism. Should have relatively low correlation with existing strategies since none use neighbour prices.

### Strategy 5: VolatilityRegimeStrategy

**Hypothesis**: High intra-day price volatility (measured by `price_std`) signals market stress and uncertainty. After high-volatility days, prices tend to mean-revert. After low-volatility days, prices tend to continue trending.

**Features**: `price_std`, `price_max`, `price_min` (lagged 1 day)

**Logic**:
- `fit()`: Compute the 75th percentile of `price_std` as the "high volatility" threshold. Compute the median of `price_change_eur_mwh` from training.
- `act()`: If yesterday was high-vol (`price_std > P75`): if price went up, short (mean-revert); if price went down, long (mean-revert). If yesterday was low-vol: follow the trend (momentum).

**Expected behaviour**: Regime-switching strategy. Should capture the well-documented volatility clustering and mean-reversion in electricity markets.

### Strategy 6: NuclearAvailabilityStrategy

**Hypothesis**: Nuclear provides ~20% of DE-LU baseload supply. Sudden drops in nuclear generation (outages, maintenance) tighten supply and push prices up. This is visible as a sharp decline in `gen_nuclear_mw_mean` compared to its recent average.

**Feature**: `gen_nuclear_mw_mean` (lagged 1 day), plus history for rolling average

**Logic**:
- `fit()`: Compute the rolling 14-day mean and std of nuclear generation from training
- `act()`: If yesterday's nuclear is > 1 std below the 14-day rolling mean (supply shortfall) -> long (+1). If nuclear is > 1 std above (surplus) -> short (-1). Otherwise -> skip (None)

**Expected behaviour**: Selective, event-driven strategy. Few trades but high conviction when nuclear drops sharply.

### Strategy 7: RenewablesSurplusStrategy

**Hypothesis**: When combined wind + solar forecasts are very high, the merit-order effect is so strong that prices drop significantly. This is a non-linear effect -- moderate renewables have less impact, but extreme surplus floods the market.

**Features**: `forecast_wind_offshore_mw_mean`, `forecast_wind_onshore_mw_mean`, `forecast_solar_mw_mean`

**Logic**:
- `fit()`: Compute the 80th percentile of combined renewables forecast from training. Also compute 20th percentile.
- `act()`: If combined > P80 (renewables flood) -> short (-1). If combined < P20 (renewables drought) -> long (+1). Otherwise -> skip (None)

**Expected behaviour**: Extreme-signal strategy with high selectivity. Should have strong win rate on trading days.

## Implementation Process (Per Strategy)

Follow TDD:

1. **Write tests first** in `tests/backtest/test_strategy_<name>.py`:
   - Test `fit()` computes correct thresholds from synthetic data
   - Test `act()` returns correct direction for known inputs
   - Test `act()` returns correct direction at boundary values
   - Test `act()` handles missing features gracefully
   - Test `reset()` clears ephemeral state but not fitted parameters
   - Test strategy works with real data slice (integration)
   - Minimum 10 tests per strategy

2. **Implement strategy** in `strategies/<name>.py`:
   - Docstring explaining the economic theory
   - Type hints on all methods
   - Keep under 60 lines

3. **Register** in `strategies/__init__.py`

4. **Verify** by running full test suite

## Files to Create

| File | Type |
|------|------|
| `strategies/solar_forecast.py` | Strategy |
| `strategies/commodity_cost.py` | Strategy |
| `strategies/temperature_extreme.py` | Strategy |
| `strategies/cross_border_spread.py` | Strategy |
| `strategies/volatility_regime.py` | Strategy |
| `strategies/nuclear_availability.py` | Strategy (optional, #6) |
| `strategies/renewables_surplus.py` | Strategy (optional, #7) |
| `tests/backtest/test_strategy_solar_forecast.py` | Tests |
| `tests/backtest/test_strategy_commodity_cost.py` | Tests |
| `tests/backtest/test_strategy_temperature_extreme.py` | Tests |
| `tests/backtest/test_strategy_cross_border_spread.py` | Tests |
| `tests/backtest/test_strategy_volatility_regime.py` | Tests |
| `tests/backtest/test_strategy_nuclear_availability.py` | Tests (optional) |
| `tests/backtest/test_strategy_renewables_surplus.py` | Tests (optional) |

## Files to Modify

- `strategies/__init__.py` -- register new strategies

## Acceptance Criteria

- [ ] At least 5 new strategies implemented and registered
- [ ] Each strategy has 10+ unit tests, all passing
- [ ] Each strategy docstring explains the economic/theoretical rationale
- [ ] No strategy file exceeds 80 lines
- [ ] All strategies use only features available at decision time (no look-ahead bias)
- [ ] All 276+ existing tests still pass
- [ ] At least 2 of the new strategies outperform the existing best (CompositeSignal, 100K PnL) in the standard backtest -- or provide clear analysis if they don't

## Labels

`enhancement`, `strategies`, `priority-high`

## Parallel Safety

Fully safe to work on in parallel with all other issues. Only touches `strategies/` directory and test files. The only shared file is `strategies/__init__.py` which just needs new imports added.
