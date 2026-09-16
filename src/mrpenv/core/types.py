"""Data model for scenarios, episodes, observations and the final-state artifact.

Day-timing convention used everywhere in this package: a day ``t`` has a
*morning* (goods receipts, then production backflush) and an *evening*
(unrecorded shrinkage, then the end-of-day volume measurement). Production
happens on days ``0 .. H-1``; the episode ends at day ``H``, on which nothing
is produced. All volumes are m3, all money is EUR, all quantities are integer
pieces.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Scenario (theta)
# --------------------------------------------------------------------------- #


class Component(BaseModel):
    """A purchased SKU with its single-source supplier master data."""

    model_config = ConfigDict(frozen=True)

    sku: str
    uom: str = "pcs"
    lead_time: int = Field(ge=1, description="promised lead time L_i in days")
    lot_size: int = Field(ge=1, description="order quantity must be a multiple of q_i")
    moq: int = Field(ge=1, description="minimum order quantity m_i, a multiple of q_i")
    unit_cost: float = Field(
        gt=0, description="c_i: the list price in EUR/piece, which is also the day-0 quote"
    )
    unit_volume: float = Field(gt=0, description="v_i in m3/piece")


class Product(BaseModel):
    """A finished product and its bill of materials over purchased components."""

    model_config = ConfigDict(frozen=True)

    pid: str
    bom: dict[str, int] = Field(description="component sku -> qty per finished unit")


class Scenario(BaseModel):
    """The static scenario theta, fully determined by ``(tier, seed, attempt)``.

    The noise tables are exogenous: drawn once from the seed and independent of
    the agent's actions (common random numbers), which is what makes replay
    deterministic.
    """

    model_config = ConfigDict(frozen=True)

    tier: str
    seed: int
    attempt: int = 0

    horizon: int = Field(ge=1, description="H; production on days 0..H-1")
    products: list[Product]
    components: list[Component]
    mps: dict[str, list[int]] = Field(
        description="product -> *forecast* qty per day, length H; this is what the ERP shows"
    )
    initial_stock: dict[str, int]

    capacity_m3: float
    budget_eur: float

    eta: float = Field(ge=0.0, le=1.0, description="ERP record-quality noise level")
    demand_surge: dict[str, list[int]] = Field(
        default_factory=dict,
        description="extra units added to the plan on the day, per product per day",
    )
    price_volatility: float = Field(
        default=0.0, ge=0.0, description="daily log-price sigma of the supplier quotes"
    )
    prices: dict[str, list[float]] = Field(
        default_factory=dict,
        description="quoted unit price per component per day; p[i][0] == unit_cost",
    )
    shrinkage: dict[str, list[int]] = Field(description="sigma[i][t], unrecorded loss at eod t")
    short_frac: dict[str, list[float]] = Field(description="rho[i][t], short-delivery fraction")
    slip: dict[str, list[int]] = Field(description="ell[i][t], delivery slip for a PO placed on t")

    scenario_hash: str = ""

    # -- derived views ----------------------------------------------------- #

    @property
    def skus(self) -> list[str]:
        return [c.sku for c in self.components]

    def price(self, sku: str, day: int) -> float:
        """The quoted unit price of ``sku`` on ``day``.

        Falls back to the list price for scenarios built before prices existed,
        and for a day past the horizon (no order can be placed there anyway).
        """
        path = self.prices.get(sku)
        if not path:
            return self.component(sku).unit_cost
        return path[min(max(day, 0), len(path) - 1)]

    def component(self, sku: str) -> Component:
        for c in self.components:
            if c.sku == sku:
                return c
        raise KeyError(sku)

    def demand(self, pid: str, day: int, realised: bool = False) -> int:
        """Planned (forecast) or realised production of ``pid`` on ``day``.

        The forecast is what the ERP shows and what the planner nets against. The
        realised figure adds the last-minute order that lands on the day itself,
        which is why safety stock is not optional.
        """
        qty = self.mps[pid][day]
        if realised:
            qty += self.demand_surge.get(pid, [0] * self.horizon)[day]
        return qty

    def planned_requirements(self) -> dict[str, list[int]]:
        """``r[i][t]`` from the *forecast* plan: the observable requirement."""
        return self._requirements(realised=False)

    def realised_requirements(self) -> dict[str, list[int]]:
        """``r[i][t]`` actually consumed, including last-minute orders. Hidden until the day."""
        return self._requirements(realised=True)

    def _requirements(self, realised: bool) -> dict[str, list[int]]:
        req: dict[str, list[int]] = {c.sku: [0] * self.horizon for c in self.components}
        for p in self.products:
            for sku, per in p.bom.items():
                row = req[sku]
                for t in range(self.horizon):
                    row[t] += per * self.demand(p.pid, t, realised)
        return req

    def mean_requirement(self, realised: bool = False) -> dict[str, float]:
        """``rbar_i``: average daily gross requirement of each component."""
        req = self._requirements(realised)
        return {sku: sum(row) / self.horizon for sku, row in req.items()}

    def hash_payload(self) -> dict[str, Any]:
        """Everything that defines theta, excluding the hash itself."""
        payload = self.model_dump(mode="json")
        payload.pop("scenario_hash", None)
        return payload


# --------------------------------------------------------------------------- #
# Episode state
# --------------------------------------------------------------------------- #


class POStatus(StrEnum):
    OPEN = "OPEN"
    RECEIVED = "RECEIVED"
    LAPSED = "LAPSED"  # arrival falls at or after H: paid for, never delivered


class PurchaseOrder(BaseModel):
    """A purchase order. ``promised_arrival`` is observable, ``actual_arrival`` is not."""

    model_config = ConfigDict(frozen=False)

    po_id: str
    sku: str
    qty: int
    placed_day: int
    promised_arrival: int
    actual_arrival: int
    status: POStatus = POStatus.OPEN
    unit_price: float = 0.0  # the quote on the day it was placed

    def public(self) -> dict[str, Any]:
        """The ERP view of this PO: no actual arrival date."""
        return {
            "po_id": self.po_id,
            "sku": self.sku,
            "qty": self.qty,
            "placed_day": self.placed_day,
            "promised_arrival": self.promised_arrival,
            "status": str(self.status),
            "unit_price": round(self.unit_price, 4),
        }


class LedgerKind(StrEnum):
    """What the ledger records: agent actions and clock events, nothing derived.

    There is no ``SUBMIT``: an episode cannot be ended early, it always runs the
    full horizon. ``TRUNCATE`` is the harness fast-forwarding an episode that hit
    the step limit, which still produces a complete horizon for the verifier.
    """

    GENESIS = "GENESIS"
    PO_CREATED = "PO_CREATED"
    ADVANCE_DAY = "ADVANCE_DAY"
    TRUNCATE = "TRUNCATE"
    INVALID = "INVALID"


class LedgerEntry(BaseModel):
    """One entry of the append-only, HMAC-chained action ledger."""

    model_config = ConfigDict(frozen=True)

    idx: int
    kind: LedgerKind
    day: int
    args: dict[str, Any] = Field(default_factory=dict)
    prev: str
    mac: str = ""


# --------------------------------------------------------------------------- #
# Observation / static view
# --------------------------------------------------------------------------- #


class InventoryRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    recorded_on_hand: int
    uom: str = "pcs"
    on_order: int = 0
    unit_price_today: float = 0.0
    unit_price_change: float = 0.0  # fraction against yesterday's quote


class ProductionLogRow(BaseModel):
    """One product on one day: what was forecast, what was actually ordered, what ran.

    ``required`` differs from ``planned`` exactly when a last-minute order landed.
    It is visible only from the day it happens, so it is evidence about the *rate*
    of surges rather than a warning about the next one.
    """

    model_config = ConfigDict(frozen=True)

    day: int
    product: str
    planned: int
    required: int
    completed: int


class Observation(BaseModel):
    """Recorded (ERP) data only. True stock and the noise tables never appear here."""

    model_config = ConfigDict(frozen=True)

    day: int
    horizon: int
    done: bool = False
    budget_remaining: float
    spend: float
    capacity_m3: float
    recorded_utilisation: float
    inventory: list[InventoryRow]
    open_purchase_orders: list[dict[str, Any]]
    production_log: list[ProductionLogRow]
    steps_used: int = 0
    step_limit: int = 0
    last_events: list[str] = Field(default_factory=list)


class SupplierRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    uom: str
    lead_time_days: int
    lot_size: int
    moq: int
    unit_cost: float
    unit_volume: float


class StaticView(BaseModel):
    """``view_master_data`` + ``view_plan``: constant for the whole episode.

    Note what is *not* here: the tier. Difficulty is a property of the task
    distribution, not something a materials planner can look up in an ERP, and an
    agent that could read it would learn to condition its risk appetite on the
    label instead of on the evidence. The tier survives only in the scenario
    reference of the final state, which the verifier needs to regenerate theta.
    """

    model_config = ConfigDict(frozen=True)

    horizon: int
    capacity_m3: float
    budget_eur: float
    suppliers: list[SupplierRow]
    bom: dict[str, dict[str, int]]
    mps: dict[str, list[int]]
    requirements: dict[str, list[int]]


# --------------------------------------------------------------------------- #
# Actions / step results
# --------------------------------------------------------------------------- #

ToolName = Literal[
    "view_inventory",
    "view_plan",
    "view_master_data",
    "view_prices",
    "create_purchase_order",
    "advance_day",
]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Gym-like step return. ``reward`` is 0 until the terminal step."""

    model_config = ConfigDict(frozen=True)

    observation: Observation
    tool_output: dict[str, Any]
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Final state artifact
# --------------------------------------------------------------------------- #


class ScenarioRef(BaseModel):
    """Everything the verifier needs to regenerate the scenario and check it is the same one."""

    model_config = ConfigDict(frozen=True)

    tier: str
    seed: int
    scenario_hash: str


class Snapshot(BaseModel):
    """Self-reported end state. The verifier recomputes every field and compares."""

    model_config = ConfigDict(frozen=True)

    day: int
    done: bool
    true_stock: dict[str, int]
    recorded_stock: dict[str, int]
    spend: float
    invalid_count: int


class FinalState(BaseModel):
    """The only input to the verifier."""

    model_config = ConfigDict(frozen=True)

    scenario_ref: ScenarioRef
    ledger: list[LedgerEntry]
    snapshot: Snapshot


class VerifierResultModel(BaseModel):
    """Serialisable mirror of :class:`mrpenv.verifier.verify.VerifierResult`.

    ``reasons`` are stable machine codes; ``explanations`` is the same content in
    sentences, grouped per component and day, for humans.
    """

    model_config = ConfigDict(frozen=True)

    score: int
    reasons: list[str]
    explanations: list[str] = Field(default_factory=list)
    diagnostics: dict[str, float]
