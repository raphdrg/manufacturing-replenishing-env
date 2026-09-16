"""The exogenous disturbances: bad records, late suppliers, last-minute orders.

Every table here is drawn once from the seed and is independent of what the
agent does. That is the common-random-numbers trick - it makes replay exact and
lets a GRPO group that shares a seed face literally the same world, so the group
baseline measures policy quality rather than luck.

Four mechanisms, each mirroring something that actually happens in a plant, and
each with its own parameter so an ablation is one config line:

``shrinkage``       unbooked scrap or loss: the truth drops, the record does not.
``short_delivery``  a short shipment booked at the ordered quantity.
``lateness``        a supplier delivering after the date it promised.
``demand_surge``    a rush order that enlarges a production run on the day.

The first two are *record* pathologies and scale with the ERP quality level
``eta``. The last two are behaviour of the world rather than of the data, so they
have their own rates and exist on every tier - a plant where nothing is ever late
and the forecast is always exact is not a planning problem worth training on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import NoiseSwitches, TierConfig


@dataclass(frozen=True)
class NoiseTables:
    """The four exogenous tables, indexed by component (or product) and day."""

    shrinkage: dict[str, list[int]]
    short_frac: dict[str, list[float]]
    slip: dict[str, list[int]]
    demand_surge: dict[str, list[int]]


def sample_record_noise(
    rng: np.random.Generator,
    skus: list[str],
    horizon: int,
    eta: float,
    mean_req: dict[str, float],
    switches: NoiseSwitches,
) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    """Shrinkage and short deliveries. Both are identically zero when ``eta == 0``."""
    shrink: dict[str, list[int]] = {}
    short: dict[str, list[float]] = {}
    for sku in skus:  # the caller guarantees a stable order
        rate = eta * 0.03 * mean_req.get(sku, 0.0) if switches.shrink else 0.0
        shrink[sku] = [int(rng.poisson(rate)) for _ in range(horizon)]

        p_short = 0.3 * eta if switches.short_delivery else 0.0
        short[sku] = [
            round(float(rng.uniform(0.05, 0.20)), 6) if rng.random() < p_short else 0.0
            for _ in range(horizon)
        ]
    return shrink, short


def sample_lateness(
    rng: np.random.Generator, skus: list[str], horizon: int, cfg: TierConfig
) -> dict[str, list[int]]:
    """``ell[i][t]``: days of delay for an order of ``i`` placed on day ``t``.

    Indexed by *placement* day, not arrival day, so the delay is a property of the
    order and is fixed the moment it is placed - which is what makes replay exact.
    The promised date stays visible to the agent either way, so a late order shows
    up as an overdue purchase order and is observable after the fact.
    """
    lo, hi = cfg.late_delivery_days
    prob = cfg.late_delivery_prob
    return {
        sku: [int(rng.integers(lo, hi + 1)) if rng.random() < prob else 0 for _ in range(horizon)]
        for sku in skus
    }


def sample_demand_surges(
    rng: np.random.Generator,
    mps: dict[str, list[int]],
    horizon: int,
    cfg: TierConfig,
) -> dict[str, list[int]]:
    """``surge[p][t]``: extra units added to a planned run on the day it runs.

    The master production schedule is a *forecast*. A customer calling in a rush
    order enlarges the run that was already scheduled, so surges only land on days
    that already have production - a plant does not start a line for nothing. The
    extra quantity appears on the day itself, too late to order material for, so
    the only defence is to have carried stock. It is visible afterwards in the
    production log, which is what makes the surge *rate* learnable.
    """
    lo, hi = cfg.demand_surge_frac
    prob = cfg.demand_surge_prob
    surges: dict[str, list[int]] = {}
    for pid in sorted(mps):
        row = []
        for planned in mps[pid]:
            if planned > 0 and prob > 0 and rng.random() < prob:
                row.append(math.ceil(planned * float(rng.uniform(lo, hi))))
            else:
                row.append(0)
        surges[pid] = row
    return surges
