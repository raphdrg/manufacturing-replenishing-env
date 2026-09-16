"""YAML configuration loading and validation.

Two kinds of config exist: ``configs/env.yaml`` (how the environment treats the
agent: invalid-action policy, step limit) and ``configs/tiers/*.yaml``
(the task distribution ``p(theta | tier)``). Both are validated
into frozen pydantic models and cached, so a config typo fails loudly at load
time rather than silently changing the task distribution.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

TIERS: tuple[str, ...] = ("easy", "medium", "hard")


def config_dir() -> Path:
    """Locate the ``configs/`` directory.

    Order: ``$MRPENV_CONFIG_DIR``, then the repository root found by walking up
    from this file, then ``./configs`` (the container's working directory).
    """
    env = os.environ.get("MRPENV_CONFIG_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if (candidate / "tiers" / "easy.yaml").is_file():
            return candidate
    cwd = Path.cwd() / "configs"
    if (cwd / "tiers" / "easy.yaml").is_file():
        return cwd
    raise FileNotFoundError("could not locate configs/; set MRPENV_CONFIG_DIR")


class EnvConfig(BaseModel):
    """Environment-level knobs."""

    model_config = ConfigDict(frozen=True)

    invalid_action_mode: Literal["terminate", "penalize"] = "terminate"
    max_invalid: int = 0
    step_limit_factor: int = 12
    step_limit_offset: int = 40

    def step_limit(self, horizon: int) -> int:
        """``T_max = 12H + 40`` tool calls by default."""
        return self.step_limit_factor * horizon + self.step_limit_offset


class NoiseSwitches(BaseModel):
    """Independent ablation switches for the two ERP record pathologies.

    Late deliveries and demand surges are not record pathologies - they are
    supplier and customer behaviour - so they have their own tier parameters
    rather than riding on ``eta``.
    """

    model_config = ConfigDict(frozen=True)

    shrink: bool = True
    short_delivery: bool = True


class TierConfig(BaseModel):
    """``p(theta | tier)``: the sampling ranges of one tier."""

    model_config = ConfigDict(frozen=True)

    name: str
    tier_id: int
    n_products: int = Field(ge=1)
    n_components: int = Field(ge=2)
    min_shared: int = 0
    horizon: int = Field(ge=2)
    bom_values: list[int]
    production_prob: float
    batch_mu: tuple[int, int]
    batch_sigma: float = 0.3
    shutdown_days: tuple[int, int]
    lead_time: tuple[int, int]
    lot_sizes: list[int]
    moq_mult: tuple[int, int]
    unit_cost_logu: tuple[float, float]
    unit_volume: tuple[float, float]
    price_volatility: tuple[float, float] = (0.0, 0.0)
    eta: tuple[float, float]
    late_delivery_prob: float = 0.0
    late_delivery_days: tuple[int, int] = (1, 2)
    demand_surge_prob: float = 0.0
    demand_surge_frac: tuple[float, float] = (0.0, 0.0)
    capacity_slack: tuple[float, float]
    budget_slack: float
    noise: NoiseSwitches = NoiseSwitches()


@cache
def load_env_config(path: str | None = None) -> EnvConfig:
    p = Path(path) if path else config_dir() / "env.yaml"
    data = yaml.safe_load(p.read_text()) or {}
    return EnvConfig(**data)


@cache
def load_tier_config(tier: str) -> TierConfig:
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {TIERS}")
    data = yaml.safe_load((config_dir() / "tiers" / f"{tier}.yaml").read_text())
    return TierConfig(**data)
