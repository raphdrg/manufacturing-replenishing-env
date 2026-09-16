"""The six checks V1-V6, each with its own reason code.

The reward is a conjunction of cheap, independently testable predicates. That is
a deliberate choice over a weighted score: any weighting creates an exchange
rate between constraint violations, and an optimiser will find it - trading a
stockout for a saving, say. A plan that stops the line once, overflows the
warehouse, or overspends is simply a failed plan.

Every check reads the *replayed true* trajectory, never the stored numbers.
"""

from __future__ import annotations

import math

from ..core.types import FinalState, Scenario, Snapshot
from .replay import Replay

TOLERANCE = 1e-6


def v1_snapshot(snapshot: Snapshot, rep: Replay) -> list[str]:
    """Integrity: the self-reported end state must equal the recomputed one.

    ``done`` is intentionally not compared here - an episode may legitimately end
    early (an invalid action in ``terminate`` mode), and V2 already requires the
    replay to have reached the horizon. Claiming completion falsely therefore
    fails V2, while claiming *numbers* falsely fails here.
    """
    reasons: list[str] = []
    if snapshot.day != rep.day:
        reasons.append("V1_TAMPER:day")
    if snapshot.invalid_count != rep.invalid_count:
        reasons.append("V1_TAMPER:invalid_count")
    if not math.isclose(snapshot.spend, rep.spend, rel_tol=1e-9, abs_tol=1e-4):
        reasons.append("V1_TAMPER:spend")
    if dict(snapshot.true_stock) != dict(rep.true_stock):
        reasons.append("V1_TAMPER:true_stock")
    if dict(snapshot.recorded_stock) != dict(rep.recorded_stock):
        reasons.append("V1_TAMPER:recorded_stock")
    return reasons


def v2_complete(rep: Replay, scenario: Scenario) -> list[str]:
    """Completeness: the episode must cover the whole horizon."""
    if not rep.reached_horizon or rep.day != scenario.horizon:
        return [f"V2_INCOMPLETE:day{rep.day}of{scenario.horizon}"]
    return []


def v3_valid(rep: Replay, max_invalid: int) -> list[str]:
    """Validity: no rejected tool calls beyond the allowance; all POs re-validate."""
    reasons = [r for r in rep.structural if r.startswith("V3_")]
    if rep.invalid_count > max_invalid:
        reasons.append(f"V3_INVALID_ACTION:{rep.invalid_count}>{max_invalid}")
    return reasons


def v4_service(rep: Replay, scenario: Scenario) -> list[str]:
    """Service: on every production day, true stock before consumption covers the need.

    "The need" is the *realised* requirement, rush orders included. The forecast is
    what the agent was shown; what the factory consumed is what it is scored on.

    This is the check that kills the naive coverage reward (EXPLOITS.md, E1):
    quantities are not enough, they have to be *there on the day*.
    """
    reasons: list[str] = []
    for slice_ in rep.days:
        for sku in scenario.skus:
            need = slice_.requirement[sku]
            have = slice_.stock_before_consumption[sku]
            if need > 0 and have < need:
                reasons.append(f"V4_SHORTFALL:{sku}@day{slice_.day}")
    if len(rep.days) < scenario.horizon:
        reasons.append(f"V4_UNSIMULATED_DAYS:{len(rep.days)}of{scenario.horizon}")
    return reasons


def v5_capacity(rep: Replay, scenario: Scenario) -> list[str]:
    """Capacity: the true end-of-day volume never exceeds the warehouse."""
    return [
        f"V5_CAPACITY:{round(s.volume_end_of_day, 3)}>{scenario.capacity_m3}@day{s.day}"
        for s in rep.days
        if s.volume_end_of_day > scenario.capacity_m3 + TOLERANCE
    ]


def v6_budget(rep: Replay, scenario: Scenario) -> list[str]:
    """Budget: total spend (orders plus cycle counts) stays within budget."""
    if rep.spend > scenario.budget_eur + TOLERANCE:
        return [f"V6_BUDGET:{round(rep.spend, 2)}>{scenario.budget_eur}"]
    return []


def diagnostics(
    rep: Replay, scenario: Scenario, final_state: FinalState, oracle_spend: float | None
) -> dict[str, float]:
    """Logged for analysis; never part of the reward."""
    required = sum(sum(row) for row in scenario.realised_requirements().values())
    served = sum(
        min(s.stock_before_consumption[sku], s.requirement[sku])
        for s in rep.days
        for sku in scenario.skus
    )
    diag: dict[str, float] = {
        "service_level": round(served / required, 6) if required else 1.0,
        "spend": round(rep.spend, 4),
        "spend_over_budget": round(rep.spend / scenario.budget_eur, 6)
        if scenario.budget_eur
        else 0.0,
        "peak_utilisation": round(rep.peak_volume(), 4),
        "peak_over_capacity": round(rep.peak_volume() / scenario.capacity_m3, 6)
        if scenario.capacity_m3
        else 0.0,
        "purchase_orders": float(len(rep.orders)),
        "max_record_error": float(rep.max_record_error),
        "rush_order_units": float(sum(sum(row) for row in scenario.demand_surge.values())),
        "late_orders": float(
            sum(
                1
                for po in rep.orders
                if po.arrival_day > po.placed_day + scenario.component(po.sku).lead_time
            )
        ),
        "steps": float(len(final_state.ledger) - 1),
        "days_simulated": float(len(rep.days)),
    }
    if oracle_spend:
        diag["spend_over_oracle"] = round(rep.spend / oracle_spend, 6)
    return diag
