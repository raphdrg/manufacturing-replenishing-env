"""``verify(final_state, key) -> VerifierResult``: the separate, final-state-only scorer.

The verifier never trusts a number it is given. It

1. checks the HMAC chain, so the action history cannot have been edited;
2. *regenerates* theta from ``(tier, seed)`` and compares the
   scenario hash, so the task cannot have been swapped for an easier one;
3. replays the actions with its own implementation of the dynamics;
4. requires the replayed numbers to equal the reported snapshot exactly;
5. scores the replayed **true** trajectory with the conjunction V1-V6.

Anything a tamperer writes into the artifact is therefore either overwritten by
the replay or detected as a mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import load_env_config
from ..core.ledger import ledger_key, verify_chain
from ..core.types import FinalState, VerifierResultModel
from ..scenario.certify import make_scenario, oracle_plan_for
from ..scenario.certify import scenario_hash as compute_hash
from .checks import (
    diagnostics,
    v1_snapshot,
    v2_complete,
    v3_valid,
    v4_service,
    v5_capacity,
    v6_budget,
)
from .explain import explain
from .replay import replay


@dataclass(frozen=True)
class VerifierResult:
    score: int
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def explanations(self) -> list[str]:
        """The same failures in plain English, grouped by component and day."""
        return explain(list(self.reasons))

    def as_model(self) -> VerifierResultModel:
        return VerifierResultModel(
            score=self.score,
            reasons=list(self.reasons),
            explanations=self.explanations,
            diagnostics=dict(self.diagnostics),
        )

    def __bool__(self) -> bool:
        return self.score == 1


def verify(
    final_state: FinalState, key: bytes | None = None, max_invalid: int | None = None
) -> VerifierResult:
    """Score a final state. Returns 0 with reason codes, or 1 with no reasons."""
    mac_key = key if key is not None else ledger_key()
    cfg = load_env_config()
    allowance = cfg.max_invalid if max_invalid is None else max_invalid

    # V1a - the action history itself
    reasons: list[str] = list(verify_chain(final_state.ledger, mac_key))

    # V1b - the task must be the task that was handed out. An artifact produced by a
    # different version of the generator fails here too: its hash will not match.
    ref = final_state.scenario_ref
    try:
        scenario = make_scenario(ref.tier, ref.seed)
    except Exception as exc:  # unknown tier/seed, or an infeasible reference
        return VerifierResult(0, [*reasons, f"V1_TAMPER:scenario_ref:{exc.__class__.__name__}"], {})
    if compute_hash(scenario) != ref.scenario_hash:
        return VerifierResult(0, [*reasons, "V1_TAMPER:scenario_hash"], {})

    # 2 - replay with the verifier's own dynamics
    rep = replay(final_state, scenario)
    reasons += [r for r in rep.structural if r.startswith("V1_")]

    # 3..6 - the reward conjunction, all on the replayed true trajectory
    reasons += v1_snapshot(final_state.snapshot, rep)
    reasons += v2_complete(rep, scenario)
    reasons += v3_valid(rep, allowance)
    reasons += v4_service(rep, scenario)
    reasons += v5_capacity(rep, scenario)
    reasons += v6_budget(rep, scenario)

    try:
        oracle_spend: float | None = oracle_plan_for(ref.tier, ref.seed).spend
    except Exception:  # pragma: no cover - the scenario regenerated above
        oracle_spend = None
    diag = diagnostics(rep, scenario, final_state, oracle_spend)

    # de-duplicate while preserving order, so a repeated shortfall reads once per day/sku
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return VerifierResult(score=0 if ordered else 1, reasons=ordered, diagnostics=diag)


def verify_json(payload: dict[str, Any], key: bytes | None = None) -> VerifierResult:
    """Verify an uploaded artifact (used by ``POST /verify`` and the CLI)."""
    return verify(FinalState(**payload), key)
