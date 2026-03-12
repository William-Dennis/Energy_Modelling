# Adding a New Strategy

## What is a strategy?

A strategy observes today's market state and returns a **direction signal**
(+1 long, -1 short, or `None` to skip).  Entry price and position size are
fixed by the market — the strategy has no control over them.

## Steps

1. **Copy the template**

   ```
   cp src/energy_modelling/strategy/template.py \
      src/energy_modelling/strategy/my_strategy.py
   ```

2. **Rename the class** (e.g. `MyStrategy`) and update the module docstring.

3. **Fill in `act()`** — return `Signal(delivery_date=..., direction=+1/-1)`
   or `None` to skip a day.

4. **Use `market.settlement_prices`** in `__init__` if your strategy needs
   future price data (perfect-foresight style).  Otherwise leave the
   constructor empty or accept `market=None`.

5. **Save the file** — the dashboard picks it up automatically on next reload.
   No registry editing required.

## Rules

- `act()` must return `Signal | None` — never a `Trade`.
- Quantity is always **1 MW**; entry price is always the **prior day's DA
  settlement**.  Do not try to set these in your strategy.
- Add unit tests in `tests/strategy/test_my_strategy.py`.
