"""The verifier: reads only the final state, recomputes everything else.

Isolation rule: nothing in this package may import ``mrpenv.core.dynamics``,
``mrpenv.core.env`` or ``mrpenv.agents``, so a bug in the environment cannot leak
into the thing that scores it. ``mrpenv/__init__.py`` keeps its exports lazy for
the same reason: an eager import there would drag the environment in through the
parent package.
"""

from __future__ import annotations

from .explain import explain
from .verify import VerifierResult, verify, verify_json

__all__ = ["VerifierResult", "explain", "verify", "verify_json"]
