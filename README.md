# Energy Modelling

Energy market forecasting and trading strategy platform built for the ENTSOE day-ahead electricity market. Includes data collection pipelines, backtesting infrastructure, a hackathon challenge framework, and a synthetic futures market model.

## Project Structure

```
src/energy_modelling/
  data_collection/   # ENTSOE & Open-Meteo data pipelines (prices, generation, load, flows, forecasts)
  strategy/          # Strategy base class, backtest runner, performance analysis
  challenge/         # Hackathon challenge runner, scoring, synthetic futures market
  dashboard/         # Streamlit dashboard (modular: EDA, backtest, challenge, market, accuracy)
  market_simulation/ # Experimental market simulation utilities
submission/          # Student strategy submissions (outside installable package)
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
# Single consolidated dashboard (EDA, backtest, challenge, market, accuracy)
streamlit run src/energy_modelling/dashboard/app.py
```

### Run Tests

```bash
pytest
```

## CLI Entry Points

| Command | Description |
|---------|-------------|
| `collect-data` | Download ENTSOE + weather data to local parquet files |
| `build-challenge-data` | Build challenge datasets for student distribution |

## Key Features

- **Data Collection**: Automated pipelines for ENTSOE prices, generation, load, cross-border flows, NTC, and Open-Meteo weather forecasts.
- **Backtesting**: Run trading strategies against historical day-ahead prices with daily PnL tracking, Sharpe ratios, drawdown analysis, and monthly breakdowns.
- **Hackathon Challenge**: Framework for students to submit strategies implementing the `ChallengeStrategy` interface, with automated scoring and leaderboard.
- **Synthetic Futures Market**: Equilibrium model that simulates how strategy profits change when strategies trade against each other, revealing genuine alpha vs. crowded signals.

## Documentation

- [Hackathon Challenge Specification](docs/hackathon_challenge.md)
- [Kaggle Dataset](docs/kaggle.md)
- [Market Implementation Status](docs/market_implementation_status.md)
