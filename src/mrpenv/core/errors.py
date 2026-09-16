"""Exception types used across the package."""

from __future__ import annotations


class MRPEnvError(Exception):
    """Base class for all environment errors."""


class InvalidAction(MRPEnvError):
    """A tool call that a real ERP would reject (bad sku, MOQ/lot violation, ...).

    Carries a machine-readable ``code`` so the ledger and the verifier can agree
    on why the action was rejected.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class EpisodeDone(MRPEnvError):
    """Raised when a tool call arrives after the episode has terminated."""


class Infeasible(MRPEnvError):
    """Raised by the clairvoyant planner when no purchase plan can serve the plan.

    Used as the rejection signal of the scenario sampler in ``scenario/certify.py``.
    """
