"""``p(theta | tier, seed, attempt)``: the procedural scenario generator.

All randomness flows from ``SeedSequence([SALT, tier_id, seed, attempt])``; no
global RNG state is ever touched, so a scenario is exactly reproducible from its
reference alone - which is what lets the verifier regenerate theta instead of
trusting it. Changing ``SALT`` reshuffles the entire task distribution, and any
artifact produced before the change stops verifying, because its scenario hash no
longer matches what the generator now produces.

This module samples everything *except* the warehouse capacity and the budget.
Those two are certificates derived from the clairvoyant plan and are set in
:mod:`mrpenv.scenario.certify`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import TierConfig, load_tier_config
from ..core.types import Component, Product, Scenario
from .noise import NoiseTables, sample_demand_surges, sample_lateness, sample_record_noise
from .prices import sample_price_paths


@dataclass(frozen=True)
class RawSample:
    """A scenario with placeholder capacity/budget, plus the slacks for certification."""

    scenario: Scenario
    capacity_slack: float
    budget_slack: float


#: Mixed into every seed so that the three tiers cannot collide and so that the
#: distribution can be deliberately reshuffled by bumping one number.
SALT = 10100


def make_rng(tier_id: int, seed: int, attempt: int) -> np.random.Generator:
    """The single entry point for randomness in this package."""
    return np.random.default_rng(np.random.SeedSequence([SALT, tier_id, seed, attempt]))


def _sample_bom(
    rng: np.random.Generator, cfg: TierConfig, pids: list[str], skus: list[str]
) -> dict[str, dict[str, int]]:
    """Assign components to products so that the structural constraints always hold.

    Constraints: every product uses >= 2 components, every component is
    used by >= 1 product, and at least ``min_shared`` components are used by >= 2
    products. Built constructively rather than by rejection so that generation
    cannot stall.
    """
    order = list(rng.permutation(np.array(skus)))
    n_shared = min(cfg.min_shared, len(skus)) if len(pids) > 1 else 0
    shared, private = order[:n_shared], order[n_shared:]

    users: dict[str, list[str]] = {sku: [] for sku in skus}
    for sku in shared:
        k = int(rng.integers(2, len(pids) + 1))
        chosen = rng.choice(np.array(pids), size=k, replace=False)
        users[str(sku)] = [str(p) for p in chosen]
    for sku in private:
        users[str(sku)] = [str(rng.choice(np.array(pids)))]

    bom: dict[str, dict[str, int]] = {p: {} for p in pids}
    for sku, owners in users.items():
        for p in owners:
            bom[p][sku] = int(rng.choice(np.array(cfg.bom_values)))

    # every product needs at least two distinct components
    for p in pids:
        while len(bom[p]) < min(2, len(skus)):
            sku = str(rng.choice(np.array(skus)))
            if sku not in bom[p]:
                bom[p][sku] = int(rng.choice(np.array(cfg.bom_values)))
    return bom


def _sample_mps(
    rng: np.random.Generator, cfg: TierConfig, pids: list[str], max_lead: int
) -> dict[str, list[int]]:
    """Sample the master production schedule, including the maintenance shutdown."""
    h = cfg.horizon
    mu = {p: int(rng.integers(cfg.batch_mu[0], cfg.batch_mu[1] + 1)) for p in pids}

    mps: dict[str, list[int]] = {}
    for p in pids:
        row = []
        for _ in range(h):
            if rng.random() < cfg.production_prob:
                row.append(math.ceil(mu[p] * float(rng.lognormal(0.0, cfg.batch_sigma))))
            else:
                row.append(0)
        mps[p] = row

    # maintenance shutdown: consecutive zero days for the whole plant
    lo, hi = cfg.shutdown_days
    n_down = int(rng.integers(lo, hi + 1)) if hi > 0 else 0
    down: set[int] = set()
    if n_down > 0 and h > n_down + 2:
        start = int(rng.integers(1, h - n_down))
        down = set(range(start, start + n_down))
        for p in pids:
            for t in down:
                mps[p][t] = 0

    # the task is trivial unless something is produced after the first order could land
    late_days = [t for t in range(max_lead + 1, h) if t not in down]
    if late_days and not any(mps[p][t] > 0 for p in pids for t in late_days):
        t = int(rng.choice(np.array(late_days)))
        p = str(rng.choice(np.array(pids)))
        mps[p][t] = mu[p]

    # and every product must be produced at least once
    for p in pids:
        if sum(mps[p]) == 0:
            free = [t for t in range(h) if t not in down]
            mps[p][int(rng.choice(np.array(free)))] = mu[p]
    return mps


def sample_raw(tier: str, seed: int, attempt: int = 0) -> RawSample:
    """Draw everything except the warehouse capacity and budget certificates."""
    cfg = load_tier_config(tier)
    rng = make_rng(cfg.tier_id, seed, attempt)

    pids = [f"P{i + 1:02d}" for i in range(cfg.n_products)]
    skus = [f"C{i + 1:02d}" for i in range(cfg.n_components)]

    components: list[Component] = []
    for sku in skus:
        lot = int(rng.choice(np.array(cfg.lot_sizes)))
        moq_mult = int(rng.integers(cfg.moq_mult[0], cfg.moq_mult[1] + 1))
        lo, hi = cfg.unit_cost_logu
        cost = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        components.append(
            Component(
                sku=sku,
                lead_time=int(rng.integers(cfg.lead_time[0], cfg.lead_time[1] + 1)),
                lot_size=lot,
                moq=lot * moq_mult,
                unit_cost=round(cost, 4),
                unit_volume=round(float(rng.uniform(*cfg.unit_volume)), 4),
            )
        )
    max_lead = max(c.lead_time for c in components)

    bom = _sample_bom(rng, cfg, pids, skus)
    products = [Product(pid=p, bom=bom[p]) for p in pids]
    mps = _sample_mps(rng, cfg, pids, max_lead)

    eta = round(float(rng.uniform(*cfg.eta)), 4)

    # rush orders enlarge runs that are already scheduled; drawn before the
    # component requirements so those can be computed on the realised demand
    surges = sample_demand_surges(rng, mps, cfg.horizon, cfg)

    # realised requirements r[i][t], needed for the noise rates and the initial stock
    req: dict[str, list[int]] = {sku: [0] * cfg.horizon for sku in skus}
    for p in products:
        for sku, per in p.bom.items():
            for t in range(cfg.horizon):
                req[sku][t] += per * (mps[p.pid][t] + surges[p.pid][t])
    mean_req = {sku: sum(row) / cfg.horizon for sku, row in req.items()}

    shrink, short = sample_record_noise(rng, skus, cfg.horizon, eta, mean_req, cfg.noise)
    slip = sample_lateness(rng, skus, cfg.horizon, cfg)
    tables = NoiseTables(shrinkage=shrink, short_frac=short, slip=slip, demand_surge=surges)

    # supplier quotes wander over the horizon; how much is a tier property
    volatility = round(float(rng.uniform(*cfg.price_volatility)), 4)
    prices = sample_price_paths(
        rng, {c.sku: c.unit_cost for c in components}, cfg.horizon, volatility
    )

    # initial stock covers requirements until the first order could possibly arrive,
    # padded for the delay a first order might suffer
    slip_pad = cfg.late_delivery_days[1] if cfg.late_delivery_prob > 0 else 0
    initial_stock: dict[str, int] = {}
    for c in components:
        cover = min(c.lead_time + slip_pad, cfg.horizon - 1)
        total = sum(req[c.sku][: cover + 1])
        initial_stock[c.sku] = math.ceil((1.0 + float(rng.uniform(0.0, 0.3))) * total)

    scenario = Scenario(
        tier=tier,
        seed=seed,
        attempt=attempt,
        horizon=cfg.horizon,
        products=products,
        components=components,
        mps=mps,
        initial_stock=initial_stock,
        capacity_m3=0.0,
        budget_eur=0.0,
        eta=eta,
        price_volatility=volatility,
        prices=prices,
        shrinkage=tables.shrinkage,
        short_frac=tables.short_frac,
        slip=tables.slip,
        demand_surge=tables.demand_surge,
    )
    return RawSample(
        scenario=scenario,
        capacity_slack=round(float(rng.uniform(*cfg.capacity_slack)), 4),
        budget_slack=cfg.budget_slack,
    )
