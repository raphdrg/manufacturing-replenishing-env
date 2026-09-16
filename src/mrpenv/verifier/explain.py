"""Turn reason codes into sentences a human can act on.

The codes themselves stay exactly as they are: they are the machine-facing
contract that tests assert on and that an RL pipeline groups failures by, so
they must not drift. This module is a *presentation* layer on top of them, used
by the play UI and the CLI.

Codes are also grouped before they are phrased. A failed episode can carry forty
``V4_SHORTFALL`` lines, which is useful for a dashboard and useless for a person;
one sentence per component, listing the days, says the same thing.
"""

from __future__ import annotations

import re
from collections import defaultdict

_MONEY = "{:,.2f}"


def _fmt_days(days: list[str]) -> str:
    if len(days) == 1:
        return f"day {days[0]}"
    return "days " + ", ".join(days[:-1]) + f" and {days[-1]}"


def _tamper(field: str) -> str:
    if field.startswith("mac"):
        return (
            "The action history has been altered: an entry does not match its signature, "
            "so the ledger cannot be trusted."
        )
    if field.startswith(("chain", "idx", "genesis")):
        return "The action history has been reordered, truncated or added to."
    if field == "scenario_hash":
        return "This result belongs to a different scenario than the one it claims."
    if field == "day":
        return "The reported final day does not match the recomputed one."
    if field in ("true_stock", "recorded_stock"):
        label = "actual" if field == "true_stock" else "recorded"
        return f"The reported {label} stock does not match what replaying the actions produces."
    if field == "spend":
        return "The reported spend does not match what replaying the purchase orders costs."
    if field == "invalid_count":
        return "The reported number of rejected tool calls does not match the ledger."
    return f"The final state does not match the recomputed one ({field})."


def explain(reasons: list[str]) -> list[str]:
    """Human-readable sentences for a list of verifier reason codes.

    Returns one sentence per distinct problem, in the order the checks run, with
    per-component and per-day detail folded in rather than repeated.
    """
    if not reasons:
        return []

    shortfalls: dict[str, list[str]] = defaultdict(list)
    capacity: list[tuple[str, float, float]] = []
    others: list[str] = []
    out: list[str] = []

    for code in reasons:
        if code.startswith("V4_SHORTFALL:"):
            body = code.split(":", 1)[1]
            sku, _, day = body.partition("@day")
            shortfalls[sku].append(day)
        elif code.startswith("V5_CAPACITY:"):
            match = re.match(r"V5_CAPACITY:([\d.]+)>([\d.]+)@day(\d+)", code)
            if match:
                capacity.append((match.group(3), float(match.group(1)), float(match.group(2))))
            else:  # pragma: no cover - defensive
                others.append(code)
        else:
            others.append(code)

    for code in others:
        head, _, body = code.partition(":")
        if head == "V1_TAMPER":
            out.append(_tamper(body))
        elif head == "V2_INCOMPLETE":
            match = re.match(r"day(\d+)of(\d+)", body)
            if match:
                out.append(
                    f"The episode stopped on day {match.group(1)} of {match.group(2)}: "
                    "the whole horizon has to be worked through."
                )
            else:  # pragma: no cover - defensive
                out.append("The episode did not cover the whole horizon.")
        elif head == "V3_INVALID_ACTION":
            count = body.split(">")[0]
            plural = "" if count == "1" else "s"
            out.append(
                f"{count} tool call{plural} {'was' if count == '1' else 'were'} rejected by the "
                "ERP. A single rejected call ends the episode and scores zero."
            )
        elif head == "V3_INVALID_PO":
            kind, _, where = body.partition(":")
            sku = where.split("@")[0] or "a component"
            reason = {
                "below_moq": "was below the supplier's minimum order quantity",
                "lot_size": "was not a multiple of the supplier's lot size",
                "unknown_sku": "named a component that does not exist",
                "qty_type": "had a quantity that was not a whole number",
            }.get(kind, f"was invalid ({kind})")
            out.append(f"A purchase order for {sku} {reason}.")
        elif head == "V4_UNSIMULATED_DAYS":
            match = re.match(r"(\d+)of(\d+)", body)
            if match:
                out.append(
                    f"Only {match.group(1)} of {match.group(2)} days were played, so the rest "
                    "of the plan was never served."
                )
        elif head == "V6_BUDGET":
            spent, _, limit = body.partition(">")
            try:
                out.append(
                    f"Over budget: {_MONEY.format(float(spent))} EUR spent against a budget of "
                    f"{_MONEY.format(float(limit))} EUR."
                )
            except ValueError:  # pragma: no cover - defensive
                out.append(f"Over budget ({body}).")
        else:
            out.append(code)

    for sku, days in shortfalls.items():
        out.append(
            f"Ran out of {sku} on {_fmt_days(days)}: production could not be completed in full."
        )

    if capacity:
        worst = max(capacity, key=lambda item: item[1])
        days = [day for day, _, _ in capacity]
        out.append(
            f"The warehouse overflowed on {_fmt_days(days)}: {worst[1]:,.1f} m3 stored at the "
            f"worst point against {worst[2]:,.1f} m3 of space."
        )
    return out
