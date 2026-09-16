"""The action ledger: an append-only HMAC hash chain.

Why a chain and not a log: the verifier reads only the final state, so the
final state has to carry the whole action history *and* make it impossible to
edit that history after the fact. Each entry commits to the previous entry's
MAC, so changing, deleting, reordering or inserting any entry invalidates every
MAC from that point on. The key lives in the server process only
(``MRPENV_LEDGER_KEY``) and is never exposed through any endpoint.

Note what is *not* in the ledger: goods receipts, consumption, shortfalls. Those
are consequences of the dynamics, which the verifier recomputes itself. Only
agent actions and clock events are recorded inputs.
"""

from __future__ import annotations

import hmac
import json
import os
from hashlib import sha256
from typing import Any

from .types import LedgerEntry, LedgerKind

GENESIS_PREV = "0" * 64
DEV_KEY = "dev-only-insecure-key"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for both hashing and MACs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def ledger_key() -> bytes:
    """Read the MAC key from the environment, falling back to the dev key."""
    return os.environ.get("MRPENV_LEDGER_KEY", DEV_KEY).encode()


def entry_mac(entry: LedgerEntry, key: bytes) -> str:
    """``HMAC_SHA256(key, canonical_json(entry without mac))``."""
    payload = entry.model_dump(mode="json")
    payload.pop("mac", None)
    return hmac.new(key, canonical_json(payload).encode(), sha256).hexdigest()


def sha256_hex(text: str) -> str:
    return sha256(text.encode()).hexdigest()


class Ledger:
    """Append-only chained log of agent actions and clock events."""

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key if key is not None else ledger_key()
        self._entries: list[LedgerEntry] = []
        self.append(LedgerKind.GENESIS, day=0, args={})

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def append(self, kind: LedgerKind, day: int, args: dict[str, Any]) -> LedgerEntry:
        prev = GENESIS_PREV if not self._entries else self._entries[-1].mac
        draft = LedgerEntry(idx=len(self._entries), kind=kind, day=day, args=args, prev=prev)
        entry = draft.model_copy(update={"mac": entry_mac(draft, self._key)})
        self._entries.append(entry)
        return entry


def verify_chain(entries: list[LedgerEntry], key: bytes) -> list[str]:
    """Check the chain. Returns a list of ``V1_TAMPER:*`` reason codes (empty if intact)."""
    reasons: list[str] = []
    if not entries:
        return ["V1_TAMPER:empty_ledger"]
    if entries[0].kind is not LedgerKind.GENESIS:
        reasons.append("V1_TAMPER:missing_genesis")
    if entries[0].prev != GENESIS_PREV:
        reasons.append("V1_TAMPER:genesis_prev")
    for i, entry in enumerate(entries):
        if entry.idx != i:
            reasons.append(f"V1_TAMPER:idx@{i}")
        expected_prev = GENESIS_PREV if i == 0 else entries[i - 1].mac
        if entry.prev != expected_prev:
            reasons.append(f"V1_TAMPER:chain@{i}")
        if not hmac.compare_digest(entry.mac, entry_mac(entry, key)):
            reasons.append(f"V1_TAMPER:mac@{i}")
    return reasons
