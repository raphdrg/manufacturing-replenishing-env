"""``mrp-env``: a verifiable material-replenishment environment for LLM agents.

Quick start::

    from mrpenv import MRPEnv, verify

    env = MRPEnv(tier="hard", seed=17)
    env.step({"tool": "create_purchase_order", "args": {"sku": "C03", "qty": 50}})
    while not env.done:                     # an episode always runs the full horizon
        env.step({"tool": "advance_day"})
    print(verify(env.final_state()))

Attribute access here is lazy (PEP 562). That is not a micro-optimisation: it is
what lets ``import mrpenv.verifier`` stay clean of the environment's own
dynamics. If this module imported ``MRPEnv`` eagerly, importing the verifier
would drag ``core.env`` and ``core.dynamics`` in through the parent package, and
the isolation the reward depends on would exist only on paper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .core.env import MRPEnv
    from .core.types import (
        FinalState,
        Observation,
        Scenario,
        StaticView,
        StepResult,
        ToolCall,
    )
    from .scenario.certify import make_scenario
    from .verifier.verify import VerifierResult, verify

_EXPORTS: dict[str, tuple[str, str]] = {
    "MRPEnv": (".core.env", "MRPEnv"),
    "FinalState": (".core.types", "FinalState"),
    "Observation": (".core.types", "Observation"),
    "Scenario": (".core.types", "Scenario"),
    "StaticView": (".core.types", "StaticView"),
    "StepResult": (".core.types", "StepResult"),
    "ToolCall": (".core.types", "ToolCall"),
    "make_scenario": (".scenario.certify", "make_scenario"),
    "verify": (".verifier.verify", "verify"),
    "VerifierResult": (".verifier.verify", "VerifierResult"),
}

__all__ = [
    "FinalState",
    "MRPEnv",
    "Observation",
    "Scenario",
    "StaticView",
    "StepResult",
    "ToolCall",
    "VerifierResult",
    "make_scenario",
    "verify",
]


def __getattr__(name: str) -> Any:
    """Import the requested export on first use (PEP 562)."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module, attribute = target
    return getattr(import_module(module, __name__), attribute)


def __dir__() -> list[str]:
    return sorted(__all__)
