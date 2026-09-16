"""The clairvoyant planner: a pure function of theta.

This is the feasibility core of the environment. It reads the hidden noise
tables and produces the *latest-feasible* purchase plan: for each component, at
the first day the true stock would fall short, order on the last day from which
the shipment still lands in time, in the smallest lot multiple whose *short-
delivered* quantity still covers the deficit.

Ordering as late as possible is deliberate: it minimises days of stock on hand,
hence peak warehouse volume, which is exactly what makes the capacity and budget
certificates in :mod:`mrpenv.scenario.certify` tight rather than generous.

It lives in ``scenario/`` rather than ``agents/`` on purpose: the scenario
generator needs it to certify solvability, and the verifier needs the generator.
Keeping it dependency-free means the verifier never transitively imports the
environment's own dynamics. :class:`mrpenv.agents.oracle.OracleAgent` is a thin
tool-call wrapper around this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.errors import Infeasible
from ..core.types import Scenario


@dataclass(frozen=True)
class PlannedPO:
    """A purchase order the planner intends to place on ``day``."""

    sku: str
    qty: int
    day: int
    arrival: int


@dataclass
class ComponentTrajectory:
    """True stock of one component along the horizon."""

    pre_consumption: list[int] = field(default_factory=list)
    end_of_day: list[int] = field(default_factory=list)
    shortfall_day: int | None = None
    shortfall_qty: int = 0


def simulate_component(
    scenario: Scenario, sku: str, pos: list[PlannedPO], stop_at_shortfall: bool = True
) -> ComponentTrajectory:
    """Project the *true* stock of one component under a given set of POs.

    Components do not interact under the true dynamics (no shared capacity
    constraint on consumption), so each can be planned independently.
    """
    # the clairvoyant planner nets against what will *actually* be consumed,
    # including the rush orders the forecast does not show
    req = scenario.realised_requirements()[sku]
    shrink = scenario.shrinkage[sku]
    short = scenario.short_frac[sku]
    arrivals: dict[int, list[int]] = {}
    for po in pos:
        if po.sku == sku:
            arrivals.setdefault(po.arrival, []).append(po.qty)

    traj = ComponentTrajectory()
    x = scenario.initial_stock[sku]
    for t in range(scenario.horizon):
        for qty in arrivals.get(t, []):
            x += math.floor(qty * (1.0 - short[t]))
        traj.pre_consumption.append(x)
        if x < req[t]:
            traj.shortfall_day = t
            traj.shortfall_qty = req[t] - x
            if stop_at_shortfall:
                return traj
        x -= min(x, req[t])
        x -= min(x, shrink[t])
        traj.end_of_day.append(x)
    return traj


def plan_component(
    scenario: Scenario, sku: str, from_day: int = 0, seed_pos: list[PlannedPO] | None = None
) -> list[PlannedPO]:
    """Latest-feasible purchase plan for one component.

    ``from_day`` is the earliest day an order may be placed (the current day when
    the oracle is used as an agent). Raises :class:`Infeasible` if some shortfall
    cannot be covered by any order - the rejection signal of the sampler.
    """
    comp = scenario.component(sku)
    slip = scenario.slip[sku]
    short = scenario.short_frac[sku]
    pos: list[PlannedPO] = list(seed_pos or [])
    max_iter = 5 * scenario.horizon + 20

    for _ in range(max_iter):
        traj = simulate_component(scenario, sku, pos)
        if traj.shortfall_day is None:
            return pos
        day, deficit = traj.shortfall_day, traj.shortfall_qty

        candidates = [t for t in range(from_day, day + 1) if t + comp.lead_time + slip[t] <= day]
        if not candidates:
            raise Infeasible(
                f"{scenario.tier}/{scenario.seed}: {sku} short {deficit} on day {day}, "
                "no order day lands in time"
            )
        order_day = max(candidates)
        arrival = order_day + comp.lead_time + slip[order_day]

        qty = comp.moq
        while math.floor(qty * (1.0 - short[arrival])) < deficit:
            qty += comp.lot_size
        pos.append(PlannedPO(sku=sku, qty=qty, day=order_day, arrival=arrival))

    raise Infeasible(f"{sku}: planner did not converge in {max_iter} iterations")


@dataclass(frozen=True)
class OraclePlan:
    """A certified-feasible plan plus the trajectory statistics it implies."""

    pos: list[PlannedPO]
    spend: float
    peak_utilisation: float
    daily_utilisation: list[float]
    initial_utilisation: float

    def pos_on_day(self, day: int) -> list[PlannedPO]:
        return [po for po in self.pos if po.day == day]


def plan_oracle(scenario: Scenario, from_day: int = 0) -> OraclePlan:
    """Plan every component, then measure spend and the true volume profile."""
    pos: list[PlannedPO] = []
    for sku in scenario.skus:
        pos.extend(plan_component(scenario, sku, from_day=from_day))

    # spend is booked at the quote of the day the order is placed
    spend = sum(scenario.price(po.sku, po.day) * po.qty for po in pos)

    daily = [0.0] * scenario.horizon
    for comp in scenario.components:
        traj = simulate_component(scenario, comp.sku, pos, stop_at_shortfall=False)
        if traj.shortfall_day is not None:  # pragma: no cover - planner guarantees this
            raise Infeasible(f"{comp.sku} still short on day {traj.shortfall_day}")
        for t, x in enumerate(traj.end_of_day):
            daily[t] += comp.unit_volume * x

    initial = sum(c.unit_volume * scenario.initial_stock[c.sku] for c in scenario.components)
    return OraclePlan(
        pos=pos,
        spend=round(spend, 6),
        peak_utilisation=max(daily) if daily else 0.0,
        daily_utilisation=daily,
        initial_utilisation=initial,
    )
