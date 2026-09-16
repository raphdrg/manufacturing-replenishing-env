"""``MRPEnv``: reset / step / final_state.

The environment owns the truth, hands out recorded observations, and writes every
agent action into the HMAC-chained ledger. It does *not* compute the reward from
its own bookkeeping: the terminal reward is whatever the independent verifier
says when it replays the ledger. That separation is the whole point - the
environment could be buggy or an agent could tamper with the artifact, and the
score would still come from a recomputation.
"""

from __future__ import annotations

from typing import Any

from ..config import EnvConfig, load_env_config
from ..scenario.certify import make_scenario
from . import dynamics as dyn
from .actions import TOOLS, CreatePOArgs, ViewPlanArgs, parse_args, validate_po
from .errors import EpisodeDone, InvalidAction
from .ledger import Ledger, ledger_key
from .observation import (
    build_observation,
    master_data_view,
    plan_view,
    price_view,
    static_view,
)
from .types import (
    FinalState,
    LedgerKind,
    Observation,
    Scenario,
    ScenarioRef,
    Snapshot,
    StaticView,
    StepResult,
    ToolCall,
)

INSTRUCTION = (
    "You are the materials planner of a small plant. Using the tools, place purchase "
    "orders so that every production run can be executed in full on its scheduled day, "
    "warehouse volume never exceeds capacity, and total spend stays within budget. "
    "Three things are not under your control: the production plan is a forecast and a "
    "rush order can enlarge a run on the day it runs, suppliers sometimes deliver after "
    "the date they promised, and quoted prices move from day to day. Inventory records "
    "in the ERP may also be inaccurate. Work through the horizon day by day; the "
    "episode ends when the last day has been closed."
)


class MRPEnv:
    """A single material-replenishment episode."""

    instruction: str = INSTRUCTION

    def __init__(
        self,
        tier: str = "easy",
        seed: int = 0,
        *,
        config: EnvConfig | None = None,
        invalid_action_mode: str | None = None,
        key: bytes | None = None,
    ) -> None:
        cfg = config or load_env_config()
        if invalid_action_mode is not None:
            cfg = cfg.model_copy(update={"invalid_action_mode": invalid_action_mode})
        self.config = cfg
        self._key = key if key is not None else ledger_key()
        self.reset(tier, seed)

    # -- lifecycle --------------------------------------------------------- #

    @classmethod
    def from_scenario(
        cls, scenario: Scenario, *, config: EnvConfig | None = None, key: bytes | None = None
    ) -> MRPEnv:
        """Build an episode on an explicit scenario instead of ``(tier, seed)``.

        Used by the hand-computed dynamics tests, and by any future loader that
        builds theta from real ERP exports. Note that a final state produced this
        way only verifies if the scenario is reproducible from its reference: the
        verifier regenerates theta and compares hashes, exactly as intended.
        """
        env = cls.__new__(cls)
        cfg = config or load_env_config()
        env.config = cfg
        env._key = key if key is not None else ledger_key()
        env.tier = scenario.tier
        env.seed = scenario.seed
        env.scenario = scenario
        env.state = dyn.initial_state(scenario)
        env.ledger = Ledger(key=env._key)
        env.steps_used = 0
        env.step_limit = cfg.step_limit(scenario.horizon)
        env.terminated = False
        env.truncated = False
        env._final_reward = None
        return env

    def reset(self, tier: str | None = None, seed: int | None = None) -> Observation:
        """Sample ``theta ~ p(theta | tier, seed)`` and build ``s_0``."""
        if tier is not None:
            self.tier = tier
        if seed is not None:
            self.seed = seed
        self.scenario: Scenario = make_scenario(self.tier, self.seed)
        self.state = dyn.initial_state(self.scenario)
        self.ledger = Ledger(key=self._key)
        self.steps_used = 0
        self.step_limit = self.config.step_limit(self.scenario.horizon)
        self.terminated = False
        self.truncated = False
        self._final_reward: float | None = None
        return self.observation()

    def observation(self) -> Observation:
        return build_observation(self.state, self.steps_used, self.step_limit)

    def static(self) -> StaticView:
        return static_view(self.scenario)

    @property
    def done(self) -> bool:
        return self.state.done

    # -- stepping ---------------------------------------------------------- #

    def step(self, call: ToolCall | dict[str, Any]) -> StepResult:
        """Apply one tool call. Every call, read or write, is one step."""
        if isinstance(call, dict):
            call = ToolCall(tool=str(call.get("tool", "")), args=dict(call.get("args") or {}))
        if self.state.done:
            raise EpisodeDone("episode has ended; create a new episode")

        self.steps_used += 1
        try:
            output = self._dispatch(call)
        except InvalidAction as exc:
            return self._handle_invalid(call, exc)

        if not self.state.done and self.steps_used >= self.step_limit:
            self.ledger.append(LedgerKind.TRUNCATE, day=self.state.day, args={})
            dyn.fast_forward(self.state)
            self.truncated = True
            output = {**output, "truncated": "step limit reached; remaining days fast-forwarded"}

        return self._result(output)

    def _dispatch(self, call: ToolCall) -> dict[str, Any]:
        args = parse_args(call.tool, call.args)
        state, sc = self.state, self.scenario

        if call.tool == "view_inventory":
            obs = self.observation()
            return {
                "day": obs.day,
                "inventory": [row.model_dump() for row in obs.inventory],
                "open_purchase_orders": obs.open_purchase_orders,
                "recorded_utilisation": obs.recorded_utilisation,
                "budget_remaining": obs.budget_remaining,
            }

        if call.tool == "view_plan":
            assert isinstance(args, ViewPlanArgs)
            return plan_view(sc, args.from_day, args.to_day)

        if call.tool == "view_master_data":
            return master_data_view(sc)

        if call.tool == "view_prices":
            return price_view(sc, state.day)

        if call.tool == "create_purchase_order":
            assert isinstance(args, CreatePOArgs)
            validate_po(sc, args.sku, args.qty, state.day)
            po = dyn.create_purchase_order(state, args.sku, args.qty)
            self.ledger.append(
                LedgerKind.PO_CREATED,
                day=po.placed_day,
                args={"sku": po.sku, "qty": po.qty, "po_id": po.po_id},
            )
            return {
                "po_id": po.po_id,
                "sku": po.sku,
                "qty": po.qty,
                "promised_arrival": po.promised_arrival,
                # the quote booked today, not the list price: they differ as soon as
                # the market has moved, and the agent is charged this one
                "unit_price": round(po.unit_price, 4),
                "cost_eur": round(po.unit_price * po.qty, 4),
                "budget_remaining": round(sc.budget_eur - state.spend, 4),
            }

        if call.tool == "advance_day":
            self.ledger.append(LedgerKind.ADVANCE_DAY, day=state.day, args={})
            dyn.advance_day(state)
            return {"day": state.day, "done": state.done, "events": state.events[-4:]}

        raise InvalidAction("UNKNOWN_TOOL", f"no such tool: {call.tool!r}")  # pragma: no cover

    def _handle_invalid(self, call: ToolCall, exc: InvalidAction) -> StepResult:
        """Log the rejected call; terminate or continue depending on the configured mode."""
        self.state.invalid_count += 1
        self.ledger.append(
            LedgerKind.INVALID,
            day=self.state.day,
            args={"tool": call.tool, "code": exc.code},
        )
        self.state.events.append(f"day {self.state.day}: INVALID {call.tool} ({exc.code})")
        output = {"error": exc.code, "message": exc.message}
        if self.config.invalid_action_mode == "terminate":
            self.state.done = True
            self.terminated = True
        return self._result(output)

    def _result(self, output: dict[str, Any]) -> StepResult:
        terminated = self.state.done
        reward = self.terminal_reward() if terminated else 0.0
        return StepResult(
            observation=self.observation(),
            tool_output=output,
            reward=reward,
            terminated=terminated,
            truncated=self.truncated,
            # info is returned to the agent over HTTP, so it may only carry things the
            # ERP would show. True stock and true volume live behind the debug
            # endpoint, which exists only when MRPENV_DEBUG_REVEAL=1.
            info={
                "day": self.state.day,
                "steps_used": self.steps_used,
                "step_limit": self.step_limit,
                "invalid_count": self.state.invalid_count,
            },
        )

    # -- artifact and reward ----------------------------------------------- #

    def final_state(self) -> FinalState:
        """The artifact the verifier reads. Valid at any time; only meaningful once done."""
        return FinalState(
            scenario_ref=ScenarioRef(
                tier=self.scenario.tier,
                seed=self.scenario.seed,
                scenario_hash=self.scenario.scenario_hash,
            ),
            ledger=self.ledger.entries,
            snapshot=Snapshot(
                day=self.state.day,
                done=self.state.done,
                true_stock=dict(self.state.true_stock),
                recorded_stock=dict(self.state.recorded_stock),
                spend=round(self.state.spend, 6),
                invalid_count=self.state.invalid_count,
            ),
        )

    def terminal_reward(self) -> float:
        """``r_T = verify(final_state).score``: zero until the horizon closes.

        The environment does not score itself from its own bookkeeping - it asks
        the verifier, which replays the ledger with an independent implementation
        of the dynamics. The environment could be buggy and the reward would still
        come from a recomputation. Imported lazily so that importing the verifier
        never pulls the environment in through the package.
        """
        if self._final_reward is None:
            from ..verifier.verify import verify

            self._final_reward = float(verify(self.final_state(), self._key).score)
        return self._final_reward

    def verify(self) -> Any:
        from ..verifier.verify import verify

        return verify(self.final_state(), self._key)

    # -- convenience ------------------------------------------------------- #

    def tool_names(self) -> list[str]:
        return list(TOOLS)

    def run(self, agent: Any, max_steps: int | None = None) -> Any:
        """Drive an agent to the end of the horizon and return the verifier result."""
        limit = max_steps or self.step_limit + 5
        static = self.static()
        obs = self.observation()
        for _ in range(limit):
            if self.state.done:
                break
            call = agent.act(obs, static)
            obs = self.step(call).observation
        while not self.state.done:  # pragma: no cover - the step limit truncates first
            self.step(ToolCall(tool="advance_day"))
        return self.verify()
