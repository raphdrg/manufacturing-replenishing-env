"""Supplier price paths: driftless geometric Brownian motion, one path per component.

A real materials planner does not buy at a fixed catalogue price. Quotes move,
and *when* you commit is part of the job. This adds that axis without adding a
second objective: the budget is still a hard constraint, so a volatile market
eats the slack that a careless planner needs.

The model, and why it is this one:

* **Geometric, not arithmetic.** ``p_t = p_{t-1} * exp(sigma * z - sigma^2 / 2)``
  keeps prices strictly positive and makes volatility relative, which is how
  commodity quotes actually behave. A price cannot walk to zero or below.
* **Driftless.** The ``- sigma^2 / 2`` term makes the path a martingale:
  ``E[p_t] = p_0`` for every ``t``. Without it, exponentiating a zero-mean normal
  would drift prices upward and "buy on day 0" would be a free lunch that needs
  no observation at all. With it, the expected price is flat and the only way to
  spend less is to *look at where the price actually went* - which is the
  behaviour this mechanic exists to reward.
* **Exogenous and seeded.** Like the other noise tables, the whole path is drawn
  at reset and stored in theta. The agent's orders never move the market, so
  replay is exact and a GRPO group sharing a seed faces the same market.
* **Visible history, hidden future.** The observation carries today's quote and
  the days already elapsed (``view_prices``). Nothing exposes the future path, so
  no amount of reading tells the agent tomorrow's price - it can only judge
  whether today's quote is cheap relative to what it has seen.

Volatility is a tier parameter: zero on ``easy`` (a fixed catalogue), small on
``medium``, large on ``hard``.
"""

from __future__ import annotations

import math

import numpy as np

#: Prices are quoted to the cent, and never allowed below this.
PRICE_FLOOR = 0.01


def sample_price_paths(
    rng: np.random.Generator,
    base_prices: dict[str, float],
    horizon: int,
    volatility: float,
) -> dict[str, list[float]]:
    """Draw one price path per component over days ``0 .. H-1``.

    Args:
        rng: the scenario's generator; consumed in sorted sku order for determinism.
        base_prices: the day-0 quote of each component (its list price).
        horizon: number of days to generate.
        volatility: daily log-price standard deviation. ``0`` gives a flat path.

    Returns:
        ``sku -> [p_0, ..., p_{H-1}]``, rounded to the cent, with ``p_0`` equal to
        the list price so that master data and the market agree on day 0.
    """
    paths: dict[str, list[float]] = {}
    for sku in sorted(base_prices):
        price = float(base_prices[sku])
        path = [round(price, 4)]
        for _ in range(1, horizon):
            if volatility > 0.0:
                shock = float(rng.normal(0.0, 1.0))
                price = price * math.exp(volatility * shock - 0.5 * volatility**2)
                price = max(PRICE_FLOOR, price)
            path.append(round(price, 4))
        paths[sku] = path
    return paths


def price_stats(path: list[float]) -> dict[str, float]:
    """Descriptive statistics of a realised path, for diagnostics and tests."""
    if not path:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "swing": 0.0}
    lo, hi = min(path), max(path)
    return {
        "min": lo,
        "max": hi,
        "mean": sum(path) / len(path),
        "swing": (hi - lo) / lo if lo > 0 else 0.0,
    }
