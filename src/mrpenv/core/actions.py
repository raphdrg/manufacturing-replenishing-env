"""Tool registry, argument models and action validity.

Every tool call is one environment step - that is how LLM agents count turns,
so that is how the step limit counts them.

Deliberate design choice: **budget and capacity are not enforced here.** A real
ERP happily lets a planner over-order and overfill the warehouse; the
consequences show up later. Enforcing them in the environment would also make
the corresponding verifier checks (V5, V6) untestable, because no trajectory
could ever violate them. The verifier enforces them instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import InvalidAction
from .types import Scenario


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ViewPlanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_day: int | None = Field(default=None, strict=True)
    to_day: int | None = Field(default=None, strict=True)


class CreatePOArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(strict=True)
    qty: int = Field(strict=True)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    write: bool
    parameters: dict[str, Any]


def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: dict[str, ToolSpec] = {
    "view_inventory": ToolSpec(
        name="view_inventory",
        description=(
            "Recorded (ERP) on-hand stock per component, quantity on order, open purchase "
            "orders with their promised arrival days, and recorded volume utilisation. "
            "Free. Records may be inaccurate."
        ),
        args_model=NoArgs,
        write=False,
        parameters=_schema({}),
    ),
    "view_plan": ToolSpec(
        name="view_plan",
        description=(
            "The master production schedule and the derived gross component requirement "
            "per day. Optional day window. Free."
        ),
        args_model=ViewPlanArgs,
        write=False,
        parameters=_schema(
            {
                "from_day": {"type": "integer", "description": "first day to return"},
                "to_day": {"type": "integer", "description": "last day to return"},
            }
        ),
    ),
    "view_master_data": ToolSpec(
        name="view_master_data",
        description=(
            "Bill of materials and supplier master data (lead time, lot size, MOQ, unit "
            "cost, unit volume) plus warehouse capacity, budget and the cycle-count fee. Free."
        ),
        args_model=NoArgs,
        write=False,
        parameters=_schema({}),
    ),
    "view_prices": ToolSpec(
        name="view_prices",
        description=(
            "Today's quoted unit price for every component plus the price history so "
            "far (list price, min, max and mean seen). Supplier quotes move daily and "
            "the price booked is the quote on the day the order is placed. Future "
            "prices are unknown. Free."
        ),
        args_model=NoArgs,
        write=False,
        parameters=_schema({}),
    ),
    "create_purchase_order": ToolSpec(
        name="create_purchase_order",
        description=(
            "Place a purchase order for one component today. Quantity must be a multiple "
            "of the lot size and at least the MOQ. Cost is booked immediately; the goods "
            "arrive after the supplier's lead time, sometimes later and sometimes short."
        ),
        args_model=CreatePOArgs,
        write=True,
        parameters=_schema(
            {
                "sku": {"type": "string", "description": "component sku, e.g. C03"},
                "qty": {"type": "integer", "description": "order quantity in pieces"},
            },
            ["sku", "qty"],
        ),
    ),
    "advance_day": ToolSpec(
        name="advance_day",
        description=(
            "Close the current day and open the next one: goods receipts arrive, then "
            "production consumes its components. The episode ends when the last day of "
            "the horizon has been closed. Free."
        ),
        args_model=NoArgs,
        write=True,
        parameters=_schema({}),
    ),
}

WRITE_TOOLS = frozenset(name for name, spec in TOOLS.items() if spec.write)


def parse_args(tool: str, args: dict[str, Any]) -> BaseModel:
    """Validate the tool name and argument shape. Raises :class:`InvalidAction`."""
    spec = TOOLS.get(tool)
    if spec is None:
        raise InvalidAction("UNKNOWN_TOOL", f"no such tool: {tool!r}")
    try:
        return spec.args_model(**(args or {}))
    except (ValidationError, TypeError) as exc:
        raise InvalidAction(
            "BAD_ARGS", f"{tool}: malformed arguments ({exc.__class__.__name__})"
        ) from None


def validate_po(scenario: Scenario, sku: str, qty: int, day: int) -> None:
    """The ERP's own purchase-order checks: sku, MOQ, lot size, and the clock."""
    try:
        comp = scenario.component(sku)
    except KeyError:
        raise InvalidAction("UNKNOWN_SKU", f"no such component: {sku!r}") from None
    if not isinstance(qty, int) or isinstance(qty, bool):
        raise InvalidAction("BAD_QTY_TYPE", f"qty must be an integer, got {type(qty).__name__}")
    if qty < comp.moq:
        raise InvalidAction("BELOW_MOQ", f"{sku}: qty {qty} below MOQ {comp.moq}")
    if qty % comp.lot_size != 0:
        raise InvalidAction(
            "LOT_SIZE", f"{sku}: qty {qty} is not a multiple of lot size {comp.lot_size}"
        )
    if not 0 <= day <= scenario.horizon - 1:
        raise InvalidAction("BAD_DAY", f"day {day} outside 0..{scenario.horizon - 1}")


# --------------------------------------------------------------------------- #
# Tool schemas for LLM agents
# --------------------------------------------------------------------------- #
#
# Generated from the registry above, so ``GET /tools`` and any agent harness
# cannot drift from what the environment actually validates.


def tool_schemas() -> list[dict[str, Any]]:
    """The tool list in OpenAI ``tools=[...]`` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in TOOLS.values()
    ]


def anthropic_tool_schemas() -> list[dict[str, Any]]:
    """The same tools in Anthropic ``tools=[...]`` format."""
    return [
        {"name": spec.name, "description": spec.description, "input_schema": spec.parameters}
        for spec in TOOLS.values()
    ]
