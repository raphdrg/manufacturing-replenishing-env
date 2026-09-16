"""The simulator dynamics, implemented once here for the environment.

The verifier contains a second, independent implementation
(:mod:`mrpenv.verifier.replay`) which must never import this module; a parity
test compares the two on random action sequences. Two implementations that agree
are evidence the dynamics are actually specified, not just coded.

Order of operations on day ``t``:

morning(t)
    1. goods receipts: true stock grows by the *short-delivered* quantity,
       the ERP books the *ordered* quantity;
    2. production backflush: components are consumed for the *realised*
       requirement, which is the forecast plus any rush order that landed today;
       the ERP books that consumption even if the truth cannot cover it, so a
       record can go negative - itself a visible anomaly.
evening(t)
    3. unrecorded shrinkage (scrap/loss): true stock drops, the record does not;
    4. the end-of-day true volume is measured for the capacity check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .types import POStatus, ProductionLogRow, PurchaseOrder, Scenario


@dataclass
class DayRecord:
    """Per-day audit trail the reward is computed from (by the verifier's own replay)."""

    day: int
    pre_consumption: dict[str, int]
    requirement: dict[str, int]
    shortfalls: dict[str, int]
    eod_true_volume: float = 0.0
    capacity_violation: bool = False


@dataclass
class EpisodeState:
    """True + recorded state of one episode."""

    scenario: Scenario
    day: int = 0
    true_stock: dict[str, int] = field(default_factory=dict)
    recorded_stock: dict[str, int] = field(default_factory=dict)
    purchase_orders: list[PurchaseOrder] = field(default_factory=list)
    spend: float = 0.0
    done: bool = False
    invalid_count: int = 0
    po_counter: int = 0
    events: list[str] = field(default_factory=list)
    production_log: list[ProductionLogRow] = field(default_factory=list)
    day_records: list[DayRecord] = field(default_factory=list)

    @property
    def requirements(self) -> dict[str, list[int]]:
        """The *realised* component requirements: what production actually consumes."""
        if not hasattr(self, "_req_cache"):
            self._req_cache: dict[str, list[int]] = self.scenario.realised_requirements()
        return self._req_cache

    def open_pos(self) -> list[PurchaseOrder]:
        return [po for po in self.purchase_orders if po.status is POStatus.OPEN]

    def true_volume(self) -> float:
        return sum(c.unit_volume * self.true_stock[c.sku] for c in self.scenario.components)

    def recorded_volume(self) -> float:
        return sum(
            c.unit_volume * max(0, self.recorded_stock[c.sku]) for c in self.scenario.components
        )


def initial_state(scenario: Scenario) -> EpisodeState:
    """Build ``s_0`` and run the morning of day 0: the first day is already underway."""
    state = EpisodeState(
        scenario=scenario,
        day=0,
        true_stock=dict(scenario.initial_stock),
        recorded_stock=dict(scenario.initial_stock),
    )
    morning(state)
    return state


def morning(state: EpisodeState) -> None:
    """Receipts, then the production backflush, for the current day."""
    sc = state.scenario
    t = state.day

    for po in state.purchase_orders:
        if po.status is POStatus.OPEN and po.actual_arrival == t:
            short = sc.short_frac[po.sku][t]
            true_qty = math.floor(po.qty * (1.0 - short))
            state.true_stock[po.sku] += true_qty
            state.recorded_stock[po.sku] += po.qty  # the ERP books the ordered quantity
            po.status = POStatus.RECEIVED
            state.events.append(
                f"day {t}: goods receipt {po.po_id} {po.sku} qty {po.qty} (recorded)"
            )

    req = state.requirements
    need = {sku: req[sku][t] for sku in sc.skus}
    pre = dict(state.true_stock)
    shortfalls = {sku: max(0, need[sku] - pre[sku]) for sku in sc.skus}

    for product in sc.products:
        forecast = sc.mps[product.pid][t]
        required = sc.demand(product.pid, t, realised=True)
        if required == 0:
            completed = 0
        elif all(shortfalls[sku] == 0 for sku in product.bom):
            completed = required
        else:
            completed = min(
                min(required, pre[sku] // per) for sku, per in product.bom.items() if per > 0
            )
        state.production_log.append(
            ProductionLogRow(
                day=t,
                product=product.pid,
                planned=forecast,
                required=required,
                completed=completed,
            )
        )
        if required > forecast:
            state.events.append(
                f"day {t}: rush order on {product.pid}: {required} needed, {forecast} forecast"
            )
        if required and completed < required:
            state.events.append(
                f"day {t}: production {product.pid} short: {completed}/{required} completed"
            )

    for sku in sc.skus:
        state.true_stock[sku] -= min(state.true_stock[sku], need[sku])
        state.recorded_stock[sku] -= need[sku]  # may go negative: a visible anomaly

    state.day_records.append(
        DayRecord(day=t, pre_consumption=pre, requirement=need, shortfalls=shortfalls)
    )


def evening(state: EpisodeState) -> None:
    """Unrecorded shrinkage, then the end-of-day capacity measurement."""
    sc = state.scenario
    t = state.day
    for sku in sc.skus:
        loss = min(state.true_stock[sku], sc.shrinkage[sku][t])
        state.true_stock[sku] -= loss

    record = state.day_records[-1]
    record.eod_true_volume = state.true_volume()
    record.capacity_violation = record.eod_true_volume > sc.capacity_m3 + 1e-9


def advance_day(state: EpisodeState) -> None:
    """Evening of ``t``, then the morning of ``t+1``. Ends the episode at ``t == H``."""
    if state.done:
        return
    evening(state)
    state.day += 1
    for po in state.purchase_orders:
        # an arrival date at or beyond the horizon never materialises
        if po.status is POStatus.OPEN and state.scenario.horizon <= po.actual_arrival < state.day:
            po.status = POStatus.LAPSED
    if state.day >= state.scenario.horizon:
        state.done = True
        for po in state.purchase_orders:
            if po.status is POStatus.OPEN:
                po.status = POStatus.LAPSED
        state.events.append(f"day {state.day}: horizon reached, episode closed")
    else:
        morning(state)


def fast_forward(state: EpisodeState) -> None:
    """Run out the remaining days with no further actions.

    Only the step-limit truncation uses this: an agent cannot end an episode
    early, so every episode covers the whole horizon and the verifier always sees
    a complete trajectory.
    """
    guard = state.scenario.horizon + 2
    while not state.done and guard > 0:
        advance_day(state)
        guard -= 1


def create_purchase_order(state: EpisodeState, sku: str, qty: int) -> PurchaseOrder:
    """Place a PO today. Spend is booked at placement, as in an ERP commitment."""
    sc = state.scenario
    comp = sc.component(sku)
    t = state.day
    unit_price = sc.price(sku, t)  # today's quote, not the list price
    state.po_counter += 1
    po = PurchaseOrder(
        po_id=f"PO-{state.po_counter:04d}",
        sku=sku,
        qty=qty,
        placed_day=t,
        promised_arrival=t + comp.lead_time,
        actual_arrival=t + comp.lead_time + sc.slip[sku][t],
        unit_price=unit_price,
    )
    state.purchase_orders.append(po)
    state.spend += unit_price * qty
    state.events.append(
        f"day {t}: PO {po.po_id} placed {sku} qty {qty} at {unit_price:.2f} EUR/pc, "
        f"promised day {po.promised_arrival}"
    )
    return po
