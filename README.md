# Energy Modelling

Energy market forecasting and trading strategy platform built for the ENTSOE day-ahead electricity market. Includes data collection pipelines, backtesting infrastructure, a hackathon challenge framework, and a synthetic futures market model.

## Terminology

| Term | Meaning |
|------|---------|
| **Backtest** | Evaluates each strategy independently using the previous day's settlement price as entry price. This is a simple, transparent baseline that students interact with directly. |
| **Futures Market** | A synthetic equilibrium model that simulates what happens when all strategies trade against each other simultaneously. Strategy weights shift based on past performance, producing a consensus "market price". This reveals whether a strategy has genuine alpha or merely rides a crowded signal. |
| **Futures Market Simulation** | The accuracy comparison tab — checks how close the futures market consensus price gets to the real day-ahead settlement. |

## Project Structure

```
src/energy_modelling/
  data_collection/   # ENTSOE & Open-Meteo data pipelines (prices, generation, load, flows, forecasts)
  backtest/          # Backtest runner, scoring, synthetic futures market engine
  futures_market/    # Shared data utilities (load_dataset, daily features, settlement computation)
  dashboard/         # Streamlit dashboard (modular: EDA, backtest, futures market, accuracy)
strategies/          # Student strategy submissions (BacktestStrategy subclasses)
tests/               # Pytest test suite
docs/                # Documentation (challenge spec, Kaggle dataset, market status)
```

## Quick Start

Requires Python >= 3.11. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### Collect Data

```bash
# Requires ENTSOE_API_KEY in .env
collect-data --start 2023-01-01 --end 2024-12-31
```

### Run Dashboard

```bash
# Single consolidated dashboard (EDA, backtest, futures market, accuracy)
streamlit run src/energy_modelling/dashboard/app.py
```

### Run Tests

```bash
pytest
```

### Regenerating Results

Re-run all backtests and benchmarks with a single command:

```bash
recompute-all            # all strategies × all benchmarks
recompute-all --benchmarks baseline oracle   # subset of benchmarks
recompute-all --strategies "Always Long" --verbose
```

Results are saved to `data/results/` and picked up automatically by the dashboard.

## CLI Entry Points

| Command | Description |
|---------|-------------|
| `collect-data` | Download ENTSOE + weather data to local parquet files |
| `build-backtest-data` | Build backtest datasets for student distribution |
| `recompute-all` | Regenerate all backtest and benchmark results |

## Key Features

- **Data Collection**: Automated pipelines for ENTSOE prices, generation, load, cross-border flows, NTC, and Open-Meteo weather forecasts.
- **Backtesting**: Run trading strategies against historical day-ahead prices with daily PnL tracking, Sharpe ratios, drawdown analysis, and monthly breakdowns.
- **Hackathon Challenge**: Framework for students to submit strategies implementing the `BacktestStrategy` interface, with automated scoring and leaderboard.
- **Synthetic Futures Market**: Equilibrium model that simulates how strategy profits change when strategies trade against each other, revealing genuine alpha vs. crowded signals.

## Documentation

- [Hackathon Challenge Specification](docs/hackathon_challenge.md)
- [Kaggle Dataset](docs/kaggle.md)
- [Market Implementation Status](docs/market_implementation_status.md)
