"""Recorded-only views of the state.

"The ERP is a sensor": the observation is exactly what an ERP user would see -
recorded stock, open purchase orders with their *promised* dates, the forecast
plan, the production log, today's quotes. True stock, the realised demand of
future days and the noise tables are never in it. This is the robotics split
between the simulator's true state and a noisy state estimate.

Everything the agent can learn about the disturbances, it learns from evidence
after the fact: an overdue purchase order, a production run that needed more
than the forecast said, a record that has gone negative.
"""

from __future__ import annotations

from typing import Any

from .dynamics import EpisodeState
from .types import (
    InventoryRow,
    Observation,
    POStatus,
    Scenario,
    StaticView,
    SupplierRow,
)

MAX_EVENTS = 8


def inventory_rows(state: EpisodeState) -> list[InventoryRow]:
    on_order: dict[str, int] = {sku: 0 for sku in state.scenario.skus}
    for po in state.purchase_orders:
        if po.status is POStatus.OPEN:
            on_order[po.sku] += po.qty
    sc = state.scenario
    day = min(state.day, sc.horizon - 1)
    rows: list[InventoryRow] = []
    for c in sc.components:
        today = sc.price(c.sku, day)
        yesterday = sc.price(c.sku, day - 1) if day > 0 else today
        rows.append(
            InventoryRow(
                sku=c.sku,
                recorded_on_hand=state.recorded_stock[c.sku],
                uom=c.uom,
                on_order=on_order[c.sku],
                unit_price_today=round(today, 4),
                unit_price_change=round((today - yesterday) / yesterday, 6) if yesterday else 0.0,
            )
        )
    return rows


def build_observation(state: EpisodeState, steps_used: int = 0, step_limit: int = 0) -> Observation:
    """Assemble the per-step observation. Kept short: static data has its own tools."""
    sc = state.scenario
    utilisation = state.recorded_volume() / sc.capacity_m3 if sc.capacity_m3 > 0 else 0.0
    return Observation(
        day=state.day,
        horizon=sc.horizon,
        done=state.done,
        budget_remaining=round(sc.budget_eur - state.spend, 4),
        spend=round(state.spend, 4),
        capacity_m3=sc.capacity_m3,
        recorded_utilisation=round(utilisation, 4),
        inventory=inventory_rows(state),
        open_purchase_orders=[
            po.public() for po in state.purchase_orders if po.status is POStatus.OPEN
        ],
        production_log=list(state.production_log),
        steps_used=steps_used,
        step_limit=step_limit,
        last_events=state.events[-MAX_EVENTS:],
    )


def static_view(scenario: Scenario) -> StaticView:
    """``view_master_data`` + ``view_plan``, fetched once per episode by an agent."""
    return StaticView(
        horizon=scenario.horizon,
        capacity_m3=scenario.capacity_m3,
        budget_eur=scenario.budget_eur,
        suppliers=[
            SupplierRow(
                sku=c.sku,
                uom=c.uom,
                lead_time_days=c.lead_time,
                lot_size=c.lot_size,
                moq=c.moq,
                unit_cost=c.unit_cost,
                unit_volume=c.unit_volume,
            )
            for c in scenario.components
        ],
        bom={p.pid: dict(p.bom) for p in scenario.products},
        mps={p.pid: list(scenario.mps[p.pid]) for p in scenario.products},
        requirements=scenario.planned_requirements(),
    )


def plan_view(
    scenario: Scenario, from_day: int | None = None, to_day: int | None = None
) -> dict[str, Any]:
    """The MPS rows and derived component requirements for a day window."""
    lo = 0 if from_day is None else max(0, from_day)
    hi = scenario.horizon - 1 if to_day is None else min(scenario.horizon - 1, to_day)
    req = scenario.planned_requirements()
    return {
        "horizon": scenario.horizon,
        "note": (
            "Quantities are the current forecast. A rush order can enlarge a run on the "
            "day it runs; past rush orders are visible in the production log."
        ),
        "from_day": lo,
        "to_day": hi,
        "mps": [
            {"day": t, "product": p.pid, "planned": scenario.mps[p.pid][t]}
            for t in range(lo, hi + 1)
            for p in scenario.products
            if scenario.mps[p.pid][t] > 0
        ],
        "component_requirements": [
            {"day": t, **{sku: req[sku][t] for sku in scenario.skus}} for t in range(lo, hi + 1)
        ],
    }


def price_view(scenario: Scenario, day: int) -> dict[str, Any]:
    """Quoted prices up to and including ``day``. The future path is never returned.

    Deliberately a separate tool rather than part of the observation: reading the
    market costs a turn, so an agent that wants to time its purchases has to
    decide that looking is worth a step - which is the trade-off a planner faces.
    """
    today = min(max(day, 0), scenario.horizon - 1)
    rows: list[dict[str, Any]] = []
    for c in scenario.components:
        history = [round(p, 4) for p in scenario.prices.get(c.sku, [c.unit_cost])[: today + 1]]
        current = history[-1]
        previous = history[-2] if len(history) > 1 else current
        rows.append(
            {
                "sku": c.sku,
                "list_price": c.unit_cost,
                "price_today": current,
                "change_vs_yesterday": round((current - previous) / previous, 6)
                if previous
                else 0.0,
                "min_so_far": min(history),
                "max_so_far": max(history),
                "mean_so_far": round(sum(history) / len(history), 4),
                "history": history,
            }
        )
    return {
        "day": today,
        "prices": rows,
        "note": (
            "Quotes are firm for today only and are booked when the order is placed. "
            "Future prices are unknown."
        ),
    }


def master_data_view(scenario: Scenario) -> dict[str, Any]:
    """Supplier master data, BOM and the three global limits."""
    static = static_view(scenario)
    return {
        "suppliers": [s.model_dump() for s in static.suppliers],
        "bom": static.bom,
        "capacity_m3": scenario.capacity_m3,
        "budget_eur": scenario.budget_eur,
        "price_note": (
            "unit_cost is the list price, which is the day-0 quote. Quotes move daily; "
            "use view_prices for today's price and the history so far."
        ),
        "horizon": scenario.horizon,
    }
