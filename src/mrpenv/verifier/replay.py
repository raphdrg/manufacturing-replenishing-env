"""An independent re-implementation of the dynamics, used only for scoring.

This module is written from the specification and must never import
:mod:`mrpenv.core.dynamics` or :mod:`mrpenv.core.env`. Two independent
implementations
that agree is the cheapest available evidence that the dynamics are specified
rather than merely coded - and it means a bug in the environment cannot silently
hand out reward.

The replay trusts only two things: the scenario reference (from which theta is
regenerated) and the ledger of agent actions. Receipts, consumption, shrinkage,
rush orders, prices and stock levels are all recomputed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor

from ..core.types import FinalState, LedgerKind, Scenario


@dataclass
class DaySlice:
    """What the reward needs to know about one simulated day."""

    day: int
    stock_before_consumption: dict[str, int]
    recorded_before_consumption: dict[str, int]
    requirement: dict[str, int]
    volume_end_of_day: float = 0.0


@dataclass
class ReplayedPO:
    po_id: str
    sku: str
    qty: int
    placed_day: int
    arrival_day: int
    unit_price: float = 0.0


@dataclass
class Replay:
    """The outcome of replaying a ledger against a regenerated scenario."""

    scenario: Scenario
    day: int = 0
    reached_horizon: bool = False
    true_stock: dict[str, int] = field(default_factory=dict)
    recorded_stock: dict[str, int] = field(default_factory=dict)
    spend: float = 0.0
    invalid_count: int = 0
    orders: list[ReplayedPO] = field(default_factory=list)
    days: list[DaySlice] = field(default_factory=list)
    structural: list[str] = field(default_factory=list)
    max_record_error: int = 0

    def peak_volume(self) -> float:
        return max((d.volume_end_of_day for d in self.days), default=0.0)


class _Sim:
    """Stateful helper holding the replayed world. Deliberately plain dict arithmetic."""

    def __init__(self, scenario: Scenario) -> None:
        self.sc = scenario
        # the realised requirement: the forecast plus the rush orders that landed
        self.req = scenario.realised_requirements()
        self.list_price = {c.sku: c.unit_cost for c in scenario.components}
        self.unit_volume = {c.sku: c.unit_volume for c in scenario.components}
        self.lead = {c.sku: c.lead_time for c in scenario.components}
        self.out = Replay(
            scenario=scenario,
            true_stock=dict(scenario.initial_stock),
            recorded_stock=dict(scenario.initial_stock),
        )
        self.pending: dict[int, list[ReplayedPO]] = {}
        self._open_morning()

    # -- day machinery ----------------------------------------------------- #

    def _open_morning(self) -> None:
        sc, out = self.sc, self.out
        t = out.day
        if t >= sc.horizon:
            return
        for po in self.pending.pop(t, []):
            delivered = floor(po.qty * (1.0 - sc.short_frac[po.sku][t]))
            out.true_stock[po.sku] += delivered
            out.recorded_stock[po.sku] += po.qty

        before = dict(out.true_stock)
        recorded_before = dict(out.recorded_stock)
        need = {sku: self.req[sku][t] for sku in sc.skus}
        for sku in sc.skus:
            out.true_stock[sku] = max(0, out.true_stock[sku] - need[sku])
            out.recorded_stock[sku] = out.recorded_stock[sku] - need[sku]
        out.days.append(
            DaySlice(
                day=t,
                stock_before_consumption=before,
                recorded_before_consumption=recorded_before,
                requirement=need,
            )
        )
        self._track_record_error()

    def _close_evening(self) -> None:
        sc, out = self.sc, self.out
        t = out.day
        if t >= sc.horizon:
            return
        for sku in sc.skus:
            out.true_stock[sku] = max(0, out.true_stock[sku] - sc.shrinkage[sku][t])
        volume = sum(self.unit_volume[sku] * out.true_stock[sku] for sku in sc.skus)
        self.out.days[-1].volume_end_of_day = volume
        self._track_record_error()

    def _track_record_error(self) -> None:
        out = self.out
        err = max(
            (abs(out.true_stock[sku] - out.recorded_stock[sku]) for sku in self.sc.skus),
            default=0,
        )
        out.max_record_error = max(out.max_record_error, err)

    def advance(self) -> None:
        out = self.out
        if out.reached_horizon:
            return
        self._close_evening()
        out.day += 1
        if out.day >= self.sc.horizon:
            out.reached_horizon = True
        else:
            self._open_morning()

    def run_out(self) -> None:
        guard = self.sc.horizon + 2
        while not self.out.reached_horizon and guard > 0:
            self.advance()
            guard -= 1

    # -- ledger actions ---------------------------------------------------- #

    def quote(self, sku: str, day: int) -> float:
        """The market price of ``sku`` on ``day``, recomputed from theta.

        The ledger records only what was ordered, never what it cost, so the price
        an episode claims to have paid is never an input to the score.
        """
        path = self.sc.prices.get(sku)
        if not path:
            return self.list_price[sku]
        return path[min(max(day, 0), len(path) - 1)]

    def order(self, po_id: str, sku: str, qty: int) -> None:
        out = self.out
        arrival = out.day + self.lead[sku] + self.sc.slip[sku][out.day]
        unit_price = self.quote(sku, out.day)
        po = ReplayedPO(
            po_id=po_id,
            sku=sku,
            qty=qty,
            placed_day=out.day,
            arrival_day=arrival,
            unit_price=unit_price,
        )
        out.orders.append(po)
        if arrival < self.sc.horizon:
            self.pending.setdefault(arrival, []).append(po)
        out.spend += unit_price * qty


def replay(final_state: FinalState, scenario: Scenario) -> Replay:
    """Replay the ledger against a freshly regenerated scenario.

    Structural problems (an ``ADVANCE_DAY`` whose recorded day does not match the
    clock, a backdated order, an unknown sku, an order that breaks MOQ or lot
    size) are collected as reason codes rather than raised, so the verifier can
    report every reason at once.
    """
    sim = _Sim(scenario)
    out = sim.out
    entries = final_state.ledger
    known = set(scenario.skus)

    for i, entry in enumerate(entries):
        kind = entry.kind
        if kind is LedgerKind.GENESIS:
            if i != 0:
                out.structural.append(f"V1_TAMPER:genesis@{i}")
            continue

        if out.reached_horizon:
            out.structural.append(f"V1_TAMPER:action_after_end@{i}")
            continue

        if kind is LedgerKind.INVALID:
            out.invalid_count += 1
            continue

        if entry.day != out.day:
            out.structural.append(f"V1_TAMPER:day@{i}")
            continue

        if kind is LedgerKind.PO_CREATED:
            sku = str(entry.args.get("sku", ""))
            qty = entry.args.get("qty")
            po_id = str(entry.args.get("po_id", f"PO-{i:04d}"))
            if sku not in known:
                out.structural.append(f"V3_INVALID_PO:unknown_sku:{sku}@{i}")
                continue
            if not isinstance(qty, int) or isinstance(qty, bool):
                out.structural.append(f"V3_INVALID_PO:qty_type@{i}")
                continue
            comp = scenario.component(sku)
            if qty < comp.moq:
                out.structural.append(f"V3_INVALID_PO:below_moq:{sku}@{i}")
                continue
            if qty % comp.lot_size != 0:
                out.structural.append(f"V3_INVALID_PO:lot_size:{sku}@{i}")
                continue
            sim.order(po_id, sku, qty)

        elif kind is LedgerKind.ADVANCE_DAY:
            sim.advance()

        elif kind is LedgerKind.TRUNCATE:
            # the harness hit the step limit and fast-forwarded the rest of the horizon
            sim.run_out()

        else:  # pragma: no cover - LedgerKind is exhaustive
            out.structural.append(f"V1_TAMPER:unknown_kind@{i}")

    out.spend = round(out.spend, 6)
    return out
