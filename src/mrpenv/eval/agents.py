"""The agents that ship with the environment.

Three policies and the helpers they share, in one module because they are all
variations on the same idea: read the ERP, net the plan, decide when to order.

``RandomAgent``
    The control condition. Legal calls, no planning.
``HeuristicAgent``
    The example policy. Plan-driven netting over recorded data only, with buffers
    and price timing. This is the one to read if you want to see what acting in
    this environment looks like.
``OracleAgent``
    Privileged: it reads the hidden disturbance tables. Not a baseline - it is the
    feasibility certificate of the task distribution, which is why every scenario
    the generator emits is one this agent solves.

Everything except the oracle sees only what :meth:`mrpenv.MRPEnv.step` returns:
recorded stock, promised delivery dates, the forecast plan, today's quotes. None
of them is told which tier it is playing.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from ..core.types import Observation, Scenario, StaticView, ToolCall
from ..scenario.certify import oracle_plan_for
from ..scenario.oracle_plan import OraclePlan, plan_oracle

PricePolicy = Literal["ignore", "opportunistic"]


@runtime_checkable
class Agent(Protocol):
    """What the environment expects of a policy: one tool call per observation."""

    name: str
    privileged: bool

    def act(self, obs: Observation, static: StaticView) -> ToolCall: ...


def round_lot(qty: int | float, lot: int, moq: int) -> int:
    """Smallest orderable quantity that is >= ``qty``: a lot multiple, at least the MOQ."""
    need = max(math.ceil(qty), 1)
    rounded = int(math.ceil(need / lot) * lot)
    return max(rounded, moq)


def supplier_map(static: StaticView) -> dict[str, dict[str, float]]:
    return {
        s.sku: {
            "lead_time": float(s.lead_time_days),
            "lot_size": float(s.lot_size),
            "moq": float(s.moq),
            "unit_cost": s.unit_cost,
            "unit_volume": s.unit_volume,
        }
        for s in static.suppliers
    }


def mean_requirement(static: StaticView) -> dict[str, float]:
    return {sku: sum(row) / static.horizon for sku, row in static.requirements.items()}


def incoming_by_day(obs: Observation, horizon: int) -> dict[str, dict[int, int]]:
    """Open POs bucketed by the day the *records* expect them.

    An overdue PO (promised date already past, still open) is assumed to arrive
    tomorrow - which is what a planner looking at an ERP would assume, and what
    makes delivery slips hurt.
    """
    incoming: dict[str, dict[int, int]] = {}
    for po in obs.open_purchase_orders:
        sku = str(po["sku"])
        eta = int(po["promised_arrival"])
        if eta <= obs.day:
            eta = obs.day + 1
        eta = min(eta, horizon)
        incoming.setdefault(sku, {})
        incoming[sku][eta] = incoming[sku].get(eta, 0) + int(po["qty"])
    return incoming


def project_recorded(
    obs: Observation, static: StaticView, sku: str, extra: dict[int, int] | None = None
) -> list[int]:
    """Projected recorded on-hand stock for ``sku`` on days ``obs.day .. H-1``.

    Netted against the *forecast* requirement, because that is what the ERP shows.
    A rush order that has not happened yet cannot appear here, which is precisely
    why an agent needs buffers rather than better arithmetic.

    Index ``k`` of the result is day ``obs.day + k`` *after* that day's planned
    consumption. Note that the observed on-hand figure is already net of today's
    backflush - today's receipts and today's production both happened this
    morning - so the projection starts from it and only nets the *future* days.
    Receipts are credited on the promised arrival day, which is all the ERP knows.
    """
    horizon = static.horizon
    req = static.requirements[sku]
    incoming = incoming_by_day(obs, horizon).get(sku, {})
    on_hand = next((r.recorded_on_hand for r in obs.inventory if r.sku == sku), 0)

    stock = on_hand
    projection: list[int] = [stock]
    for tau in range(obs.day + 1, horizon):
        stock += incoming.get(tau, 0)
        if extra:
            stock += extra.get(tau, 0)
        stock -= req[tau]
        projection.append(stock)
    return projection


#: Mass on each action family. Reads are kept in, at a low rate, because a random
#: agent that never looks is a slightly different (and weaker) control.
P_ADVANCE = 0.50
P_ORDER = 0.40
P_READ = 0.10

READ_TOOLS = ("view_inventory", "view_plan", "view_master_data", "view_prices")


class RandomAgent:
    """Places legal purchase orders at random and advances the clock at random."""

    name = "random"
    privileged = False

    def __init__(self, seed: int = 0, max_lots: int = 6) -> None:
        self.rng = np.random.default_rng(np.random.SeedSequence([7919, seed]))
        self.max_lots = max_lots

    def act(self, obs: Observation, static: StaticView) -> ToolCall:
        draw = float(self.rng.random())
        if draw < P_ADVANCE or not static.suppliers:
            return ToolCall(tool="advance_day")
        if draw < P_ADVANCE + P_ORDER:
            supplier = static.suppliers[int(self.rng.integers(len(static.suppliers)))]
            lots = int(self.rng.integers(0, self.max_lots + 1))
            qty = supplier.moq + lots * supplier.lot_size
            return ToolCall(tool="create_purchase_order", args={"sku": supplier.sku, "qty": qty})
        return ToolCall(tool=READ_TOOLS[int(self.rng.integers(len(READ_TOOLS)))])


class HeuristicAgent:
    """Daily MRP netting over the *recorded* state."""

    privileged = False

    def __init__(
        self,
        safety_days: float = 0.0,
        adapt: bool = False,
        base_lead_pad: int = 0,
        max_lead_pad: int = 3,
        max_loss_rate: float = 0.35,
        max_surge_rate: float = 0.40,
        price_policy: PricePolicy = "ignore",
        buy_ahead_days: int = 4,
        price_discount: float = 0.04,
        min_price_swing: float = 0.08,
        volume_guard: float = 0.75,
        name: str | None = None,
    ) -> None:
        self.safety_days = safety_days
        self.adapt = adapt
        self.base_lead_pad = base_lead_pad
        self.max_lead_pad = max_lead_pad
        self.max_loss_rate = max_loss_rate
        self.max_surge_rate = max_surge_rate
        self.price_policy: PricePolicy = price_policy
        self.buy_ahead_days = buy_ahead_days
        self.price_discount = price_discount
        self.min_price_swing = min_price_swing
        self.volume_guard = volume_guard
        self.name = name or ("heuristic" if adapt else "mrp")

        self._queue: list[ToolCall] = []
        self._day: int | None = None
        # everything below is estimated from observations, and starts at zero
        self._lead_pad = 0
        self._shortfall_units = 0.0
        self._required_units = 0.0
        self._forecast_units = 0.0
        self._surge_units = 0.0
        self._anomalies = 0
        self._price_history: dict[str, list[float]] = {}
        self._price_seen_day: int | None = None
        self._log_seen = 0

    # -- one decision per day ----------------------------------------------- #

    def act(self, obs: Observation, static: StaticView) -> ToolCall:
        self._learn(obs, static)
        if self._day != obs.day:
            self._day = obs.day
            self._queue = []
        if not self._queue:
            self._queue = [*self._orders(obs, static), ToolCall(tool="advance_day")]
        return self._queue.pop(0)

    # -- learning the world from evidence ----------------------------------- #

    def _learn(self, obs: Observation, static: StaticView) -> None:
        """Update the price, lateness, rush-order and loss estimates from the ERP."""
        if self._price_seen_day != obs.day:
            self._price_seen_day = obs.day
            for quote in obs.inventory:
                if quote.unit_price_today > 0:
                    self._price_history.setdefault(quote.sku, []).append(quote.unit_price_today)

        if not self.adapt:
            return

        # (a) suppliers: an open purchase order past its promised date is evidence
        for po in obs.open_purchase_orders:
            late = obs.day - int(po["promised_arrival"])
            if late > 0:
                self._anomalies += 1
                self._lead_pad = min(self.max_lead_pad, max(self._lead_pad, late))

        # (b) customers and (c) records: the production log carries both. A run that
        # needed more than the forecast is a rush order; a run that came up short is
        # proof the records were overstating what was physically there.
        for log in obs.production_log[self._log_seen :]:
            self._forecast_units += log.planned
            self._required_units += log.required
            if log.required > log.planned:
                self._surge_units += log.required - log.planned
                self._anomalies += 1
            if log.required > log.completed:
                self._shortfall_units += log.required - log.completed
                self._anomalies += 1
        self._log_seen = len(obs.production_log)

        for inv in obs.inventory:
            if inv.recorded_on_hand < 0:
                self._anomalies += 1

    def _surge_rate(self) -> float:
        """Observed rush-order volume as a fraction of the forecast."""
        if self._forecast_units <= 0:
            return 0.0
        return min(self.max_surge_rate, self._surge_units / self._forecast_units)

    def _loss_rate(self) -> float:
        """Observed production shortfall as a fraction of what was required."""
        if self._required_units <= 0:
            return 0.0
        return min(self.max_loss_rate, self._shortfall_units / self._required_units)

    # -- netting ------------------------------------------------------------ #

    def _orders(self, obs: Observation, static: StaticView) -> list[ToolCall]:
        suppliers = supplier_map(static)
        rbar = mean_requirement(static)
        buffered = self.adapt
        surge_rate = self._surge_rate() if buffered else 0.0
        loss_rate = self._loss_rate() if buffered else 0.0
        calls: list[ToolCall] = []

        for sku, sup in suppliers.items():
            pad = (self._lead_pad + self.base_lead_pad) if buffered else 0
            lead = int(sup["lead_time"]) + pad  # order early enough to absorb lateness
            target_day = obs.day + lead  # the earliest day a new order can serve
            if target_day > static.horizon - 1:
                continue

            # cover the planned buffer, the rush orders the forecast does not show,
            # and the stock the records claim but the shelf does not have
            days = self.safety_days if buffered else 0.0
            safety = rbar[sku] * (days + (surge_rate + loss_rate) * (lead + 1))
            projection = project_recorded(obs, static, sku)

            first_short: int | None = None
            for day in range(target_day, static.horizon):
                if projection[day - obs.day] < safety - 1e-9:
                    first_short = day
                    break
            if first_short is None:
                continue  # covered for the rest of the horizon
            if first_short - lead > obs.day and not self._buy_early(obs, sku, first_short, lead):
                continue  # can still wait, and the market gives no reason to hurry

            deficit = (safety - projection[first_short - obs.day]) * (1.0 + loss_rate)
            qty = round_lot(deficit, int(sup["lot_size"]), int(sup["moq"]))
            calls.append(ToolCall(tool="create_purchase_order", args={"sku": sku, "qty": qty}))
        return calls

    # -- timing the market -------------------------------------------------- #

    def _buy_early(self, obs: Observation, sku: str, need_day: int, lead: int) -> bool:
        """Whether to pull a not-yet-due order forward because today's quote is cheap.

        Four conditions, and each one matters:

        * the need is close enough that the material will be consumed soon rather
          than sit in the warehouse for the whole horizon (``buy_ahead_days``);
        * this market has actually moved. The realised swing of the observed path
          is the agent's own estimate of volatility, so on a flat catalogue it
          never trades warehouse space for timing, and it needs no privileged
          knowledge of which tier it is in;
        * today's quote is below what this agent has *seen* so far - the only
          information it has, since the future path is unknowable;
        * the warehouse has room, because buying early converts budget into
          volume, and volume is also a hard constraint.

        Prices are a driftless martingale, so this cannot be an edge on the
        expected price; it is an edge on the *realised* path, which is exactly
        what observing history buys.
        """
        if self.price_policy != "opportunistic":
            return False
        if need_day - lead > obs.day + self.buy_ahead_days:
            return False
        history = self._price_history.get(sku, [])
        if len(history) < 3:
            return False  # not enough of the path seen to call anything cheap
        low, high = min(history), max(history)
        if low <= 0 or (high - low) / low < self.min_price_swing:
            return False  # a market this quiet is not worth paying warehouse space for
        today = history[-1]
        mean_so_far = sum(history) / len(history)
        cheap = today <= mean_so_far * (1.0 - self.price_discount) or today <= low
        return bool(cheap and obs.recorded_utilisation < self.volume_guard)


class OracleAgent:
    """Emits the latest-feasible purchase plan, day by day."""

    name = "oracle (privileged)"
    privileged = True

    def __init__(
        self,
        tier: str | None = None,
        seed: int | None = None,
        scenario: Scenario | None = None,
    ) -> None:
        if scenario is not None:
            self.plan: OraclePlan = plan_oracle(scenario)
        elif tier is not None and seed is not None:
            self.plan = oracle_plan_for(tier, seed)
        else:  # pragma: no cover - programming error
            raise ValueError("OracleAgent needs either a scenario or (tier, seed)")
        self._queue: list[ToolCall] = []
        self._day: int | None = None

    def act(self, obs: Observation, static: StaticView) -> ToolCall:
        if self._day != obs.day:
            self._day = obs.day
            self._queue = [
                ToolCall(tool="create_purchase_order", args={"sku": po.sku, "qty": po.qty})
                for po in self.plan.pos_on_day(obs.day)
            ]
            self._queue.append(ToolCall(tool="advance_day"))
        if not self._queue:
            self._queue = [ToolCall(tool="advance_day")]
        return self._queue.pop(0)
