"""FastAPI server: the environment behind an HTTP API, plus the play UI.

Endpoints mirror the tool API one-to-one, so an agent harness needs no glue
beyond an HTTP client. Three deliberate properties:

* **the episode store is a dict.** An episode is a few kilobytes of Python
  objects and a step is microseconds, so one container serves hundreds of
  concurrent rollouts. That is the structural difference from desktop
  computer-use environments, where a task instance is a whole operating system;
* **the MAC key lives in this process only** and is never returned by any
  endpoint, so an agent cannot forge a final state. Note that this only holds
  across the process boundary: an agent running in-process can read the key from
  the environment, so tamper resistance is a property of *deploying* this server,
  not of the code;
* **true stock is unreachable.** ``/debug/{id}/truth`` is mounted solely when
  ``MRPENV_DEBUG_REVEAL=1``; in a normal run the route does not exist, and no
  other response carries the hidden state - not even the ``info`` block.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .core.actions import anthropic_tool_schemas, tool_schemas
from .core.env import MRPEnv
from .core.errors import EpisodeDone
from .core.ledger import DEV_KEY
from .core.types import FinalState, Observation, StaticView
from .verifier.verify import verify

log = logging.getLogger("mrpenv.server")
STATIC_DIR = Path(__file__).parent / "static"

# --------------------------------------------------------------------------- #
# Episode store
# --------------------------------------------------------------------------- #


class StoreFull(RuntimeError):
    """Raised when the episode cap is reached; the API turns this into a 429."""


@dataclass
class EpisodeHandle:
    """One live episode plus its lock and bookkeeping."""

    episode_id: str
    env: MRPEnv
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_used_at = time.monotonic()


class EpisodeStore:
    """``episode_id -> EpisodeHandle`` with a TTL and a capacity limit."""

    def __init__(
        self, ttl_seconds: float = 3600.0, max_episodes: int = 1000, sweep_seconds: float = 60.0
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_episodes = max_episodes
        self.sweep_seconds = sweep_seconds
        self._episodes: dict[str, EpisodeHandle] = {}
        self._sweeper: asyncio.Task[None] | None = None

    def create(self, tier: str, seed: int, invalid_action_mode: str | None = None) -> EpisodeHandle:
        self.sweep()
        if len(self._episodes) >= self.max_episodes:
            raise StoreFull(f"episode cap reached ({self.max_episodes})")
        env = MRPEnv(tier=tier, seed=seed, invalid_action_mode=invalid_action_mode)
        handle = EpisodeHandle(episode_id=uuid.uuid4().hex[:16], env=env)
        self._episodes[handle.episode_id] = handle
        return handle

    def get(self, episode_id: str) -> EpisodeHandle:
        handle = self._episodes.get(episode_id)
        if handle is None:
            raise KeyError(episode_id)
        handle.touch()
        return handle

    def drop(self, episode_id: str) -> None:
        self._episodes.pop(episode_id, None)

    def sweep(self) -> int:
        """Drop idle episodes. Returns how many were removed."""
        now = time.monotonic()
        stale = [
            eid
            for eid, h in self._episodes.items()
            if now - h.last_used_at > self.ttl_seconds and not h.lock.locked()
        ]
        for eid in stale:
            self._episodes.pop(eid, None)
        return len(stale)

    def __len__(self) -> int:
        return len(self._episodes)

    async def _sweep_forever(self) -> None:
        while True:  # pragma: no cover - only exercised by a live server
            await asyncio.sleep(self.sweep_seconds)
            self.sweep()

    def start(self) -> None:
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_forever())

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None


# --------------------------------------------------------------------------- #
# Request and response models
# --------------------------------------------------------------------------- #


class CreateEpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Literal["easy", "medium", "hard"] = "easy"
    seed: int = 0
    invalid_action_mode: Literal["terminate", "penalize"] | None = None


class CreateEpisodeResponse(BaseModel):
    episode_id: str
    observation: Observation
    instruction: str
    static: StaticView
    step_limit: int


class StepRequest(BaseModel):
    # extra args are allowed through so that malformed tool calls reach the
    # environment's own validator and are logged as INVALID, exactly as an
    # LLM's bad tool call would be - rejecting them at the HTTP layer would
    # hide the very failure mode the invalid-action policy exists for
    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: Observation
    tool_output: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class VerifyResponse(BaseModel):
    score: int
    reasons: list[str]  # stable machine codes
    explanations: list[str] = Field(default_factory=list)  # the same thing in English
    diagnostics: dict[str, float]


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


def debug_reveal_enabled() -> bool:
    return os.environ.get("MRPENV_DEBUG_REVEAL", "0") == "1"


def create_app(store: EpisodeStore | None = None) -> FastAPI:
    # note: `store or EpisodeStore()` would be wrong - an empty store is falsy
    episodes = EpisodeStore() if store is None else store

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if os.environ.get("MRPENV_LEDGER_KEY", DEV_KEY) == DEV_KEY:
            log.warning(
                "MRPENV_LEDGER_KEY is unset: using the public dev key. Ledger MACs are "
                "NOT trustworthy in this configuration - set a real key before using "
                "scores from this server for training or evaluation."
            )
        if debug_reveal_enabled():
            log.warning("MRPENV_DEBUG_REVEAL=1: /debug/{id}/truth is mounted and leaks true stock")
        episodes.start()
        try:
            yield
        finally:
            await episodes.stop()

    app = FastAPI(
        title="Material Restocking Environment",
        summary="A verifiable material-replenishment environment for LLM agents",
        lifespan=lifespan,
    )
    app.state.store = episodes

    # -- meta --------------------------------------------------------------- #

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "episodes": len(episodes),
            "debug_reveal": debug_reveal_enabled(),
        }

    @app.get("/tools")
    def tools(dialect: str = "openai") -> dict[str, Any]:
        schemas = anthropic_tool_schemas() if dialect == "anthropic" else tool_schemas()
        return {"dialect": dialect, "tools": schemas}

    # -- episodes ----------------------------------------------------------- #

    @app.post("/episodes", response_model=CreateEpisodeResponse)
    def create_episode(body: CreateEpisodeRequest) -> CreateEpisodeResponse:
        try:
            handle = episodes.create(body.tier, body.seed, body.invalid_action_mode)
        except StoreFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:  # an unsamplable scenario is a server-side problem
            raise HTTPException(
                status_code=500, detail=f"scenario generation failed: {exc}"
            ) from exc
        env = handle.env
        return CreateEpisodeResponse(
            episode_id=handle.episode_id,
            observation=env.observation(),
            instruction=env.instruction,
            static=env.static(),
            step_limit=env.step_limit,
        )

    def _handle(episode_id: str) -> Any:
        try:
            return episodes.get(episode_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no such episode: {episode_id}") from None

    @app.post("/episodes/{episode_id}/step", response_model=StepResponse)
    async def step(episode_id: str, body: StepRequest) -> StepResponse:
        handle = _handle(episode_id)
        async with handle.lock:
            try:
                result = handle.env.step({"tool": body.tool, "args": body.args})
            except EpisodeDone as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        return StepResponse(**result.model_dump())

    @app.get("/episodes/{episode_id}/observation")
    def observation(episode_id: str) -> Any:
        return _handle(episode_id).env.observation()

    @app.get("/episodes/{episode_id}/static")
    def static_data(episode_id: str) -> Any:
        return _handle(episode_id).env.static()

    @app.post("/episodes/{episode_id}/verify", response_model=VerifyResponse)
    def verify_episode(episode_id: str) -> VerifyResponse:
        handle = _handle(episode_id)
        if not handle.env.done:
            raise HTTPException(status_code=409, detail="episode is not finished")
        return VerifyResponse(**handle.env.verify().as_model().model_dump())

    @app.get("/episodes/{episode_id}/final_state")
    def final_state(episode_id: str) -> Any:
        handle = _handle(episode_id)
        if not handle.env.done:
            raise HTTPException(status_code=409, detail="episode is not finished")
        return handle.env.final_state()

    @app.delete("/episodes/{episode_id}")
    def delete_episode(episode_id: str) -> dict[str, str]:
        _handle(episode_id)
        episodes.drop(episode_id)
        return {"status": "deleted"}

    # -- offline verification ------------------------------------------------ #

    @app.post("/verify", response_model=VerifyResponse)
    def verify_artifact(payload: dict[str, Any]) -> VerifyResponse:
        try:
            artifact = FinalState(**payload)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"malformed final state: {exc}") from exc
        return VerifyResponse(**verify(artifact).as_model().model_dump())

    # -- human play UI -------------------------------------------------------- #

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        page = STATIC_DIR / "index.html"
        if not page.is_file():  # pragma: no cover - only if the package is mangled
            return HTMLResponse("<h1>mrp-env</h1><p>UI asset missing.</p>", status_code=200)
        return FileResponse(page, media_type="text/html")

    @app.get("/favicon.ico")
    def favicon() -> Any:
        return JSONResponse(status_code=204, content=None)

    # -- debug (opt-in only) -------------------------------------------------- #

    if debug_reveal_enabled():

        @app.get("/debug/{episode_id}/truth")
        def truth(episode_id: str) -> dict[str, Any]:
            handle = _handle(episode_id)
            state = handle.env.state
            return {
                "day": state.day,
                "true_stock": state.true_stock,
                "recorded_stock": state.recorded_stock,
                "true_volume": round(state.true_volume(), 4),
                "capacity_m3": handle.env.scenario.capacity_m3,
                "spend": round(state.spend, 4),
            }

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    return app


app = create_app()


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the server with uvicorn (used by ``mrpenv serve``)."""
    import uvicorn

    uvicorn.run("mrpenv.server:app", host=host, port=port, reload=reload, log_level="info")
