"""Rejection sampling with a solvability certificate.

Every scenario the generator emits is provably solvable *and* has binding
constraints. That matters for RL: if some instances were impossible, the reward
would be zero regardless of behaviour and the gradient would carry no signal; if
capacity and budget were generous, the only real constraint would be timing.

The procedure: sample everything except the capacity ``K`` and the budget ``B``,
run the clairvoyant planner against the *realised* demand, resample if it cannot
serve the plan, and otherwise derive those two parameters from the plan's own
trajectory so the optimum sits just inside the constraints.

Note that the planner is clairvoyant about the rush orders too. That is the point
of a feasibility certificate: the task is always solvable by someone who knows
what is coming, so a zero reward always means the policy failed rather than the
instance being impossible.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256

from ..core.errors import Infeasible
from ..core.ledger import canonical_json
from ..core.types import Scenario
from .oracle_plan import OraclePlan, plan_oracle
from .sampler import sample_raw

MAX_ATTEMPTS = 50


def scenario_hash(scenario: Scenario) -> str:
    """``sha256:<hex>`` over the canonical JSON of theta, excluding the hash field.

    This is what makes the scenario reference tamper-evident: the verifier
    regenerates theta from ``(tier, seed)`` and compares hashes, so an artifact
    cannot be relabelled onto an easier task, and one produced by an older
    generator stops verifying rather than being silently mis-scored.
    """
    digest = sha256(canonical_json(scenario.hash_payload()).encode()).hexdigest()
    return f"sha256:{digest}"


def with_hash(scenario: Scenario) -> Scenario:
    return scenario.model_copy(update={"scenario_hash": scenario_hash(scenario)})


def certify(tier: str, seed: int) -> tuple[Scenario, OraclePlan]:
    """Return a certified scenario and the clairvoyant plan that certifies it."""
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        raw = sample_raw(tier, seed, attempt)
        try:
            plan = plan_oracle(raw.scenario)
        except Infeasible as exc:  # unservable draw: resample with attempt += 1
            last = exc
            continue

        capacity = round(
            max(plan.peak_utilisation, plan.initial_utilisation) * (1.0 + raw.capacity_slack), 4
        )
        budget = round(plan.spend * (1.0 + raw.budget_slack), 4)

        scenario = with_hash(
            raw.scenario.model_copy(update={"capacity_m3": capacity, "budget_eur": budget})
        )
        # the certificates must hold for the plan they were derived from
        assert max(plan.daily_utilisation, default=0.0) <= capacity + 1e-9
        assert plan.spend <= budget + 1e-9
        return scenario, plan

    raise Infeasible(f"{tier}/{seed}: no feasible scenario in {MAX_ATTEMPTS} attempts ({last})")


@lru_cache(maxsize=512)
def make_scenario(tier: str, seed: int) -> Scenario:
    """The public generator: ``(tier, seed) -> theta``, deterministic and certified."""
    return certify(tier, seed)[0]


@lru_cache(maxsize=512)
def oracle_plan_for(tier: str, seed: int) -> OraclePlan:
    """The certifying plan, cached for the oracle agent and for diagnostics."""
    return certify(tier, seed)[1]
