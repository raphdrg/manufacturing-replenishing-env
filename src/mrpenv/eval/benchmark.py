"""Run the agents over the task distribution and report what happened.

One entry point, :func:`run_benchmark`, rolls every agent over the same seeds on
every tier and returns a summary. The comparison is only meaningful because the
seeds are shared: two rows differ by policy and nothing else.

Success is the binary verifier score. Rates carry Wilson 95% intervals rather
than the normal approximation, because the interesting cells sit near 0 and 1
where the normal interval leaves the unit interval and stops meaning anything.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ..config import TIERS
from ..core.env import MRPEnv
from .agents import Agent, HeuristicAgent, OracleAgent, RandomAgent

AgentFactory = Callable[[str, int], Agent]

#: The agents the benchmark reports, in the order they are printed. ``Random`` is
#: the floor, ``Heuristic`` is the policy to read, ``Oracle`` is the ceiling and
#: exists to prove every scenario is solvable.
AGENTS: dict[str, AgentFactory] = {
    "Random": lambda tier, seed: RandomAgent(seed=seed),
    "Heuristic": lambda tier, seed: HeuristicAgent(
        safety_days=1.5, base_lead_pad=1, adapt=True, price_policy="opportunistic"
    ),
    "Oracle (privileged)": lambda tier, seed: OracleAgent(tier, seed),
}

Z = 1.959963984540054  # two-sided 95%
#: Below this many seeds the comparisons cannot separate the agents, so the
#: invariants are not worth checking: two rows tie and report a false alarm.
MIN_SEEDS_FOR_INVARIANTS = 20


@dataclass(frozen=True)
class Cell:
    """One agent on one tier."""

    successes: int
    episodes: int
    service: float
    spend_ratio: float

    @property
    def rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        """Wilson 95% interval for the success rate."""
        n = self.episodes
        if n == 0:
            return (0.0, 0.0)
        p = self.rate
        denom = 1.0 + Z * Z / n
        centre = (p + Z * Z / (2 * n)) / denom
        half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Benchmark:
    """The whole result: ``cells[(agent, tier)]`` plus the shared configuration."""

    cells: dict[tuple[str, str], Cell]
    tiers: list[str]
    agents: list[str]
    episodes: int
    elapsed: float

    def cell(self, agent: str, tier: str) -> Cell:
        return self.cells[(agent, tier)]

    def rate(self, agent: str, tier: str) -> float:
        return self.cell(agent, tier).rate

    def overall(self, agent: str) -> Cell:
        rows = [self.cells[(agent, tier)] for tier in self.tiers]
        return Cell(
            successes=sum(r.successes for r in rows),
            episodes=sum(r.episodes for r in rows),
            service=sum(r.service for r in rows) / len(rows),
            spend_ratio=sum(r.spend_ratio for r in rows) / len(rows),
        )


def run_benchmark(
    episodes: int = 100,
    tiers: Sequence[str] = TIERS,
    agents: dict[str, AgentFactory] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Benchmark:
    """Roll every agent over ``episodes`` seeds on every tier."""
    table = agents or AGENTS
    seeds = range(episodes)
    cells: dict[tuple[str, str], Cell] = {}
    started = time.perf_counter()

    for label, factory in table.items():
        for tier in tiers:
            scores, service, spend = [], [], []
            for seed in seeds:
                env = MRPEnv(tier, seed)
                result = env.run(factory(tier, seed))
                scores.append(result.score)
                service.append(result.diagnostics["service_level"])
                spend.append(result.diagnostics["spend_over_budget"])
            cells[(label, tier)] = Cell(
                successes=sum(scores),
                episodes=len(scores),
                service=sum(service) / len(service),
                spend_ratio=sum(spend) / len(spend),
            )
        if progress:
            progress(f"{label}: {len(tiers) * episodes} episodes")

    return Benchmark(
        cells=cells,
        tiers=list(tiers),
        agents=list(table),
        episodes=episodes,
        elapsed=time.perf_counter() - started,
    )


def format_table(result: Benchmark) -> str:
    """The table view of the figure: success rate with Wilson 95% intervals."""
    columns = [*result.tiers, "all"]
    width = max(len(a) for a in result.agents) + 2
    lines = ["agent".ljust(width) + "".join(c.ljust(20) for c in columns)]
    lines.append("-" * len(lines[0]))
    for agent in result.agents:
        cells = []
        for tier in columns:
            cell = result.overall(agent) if tier == "all" else result.cell(agent, tier)
            lo, hi = cell.interval
            cells.append(f"{cell.rate:.2f} [{lo:.2f},{hi:.2f}]".ljust(20))
        lines.append(agent.ljust(width) + "".join(cells))
    lines += [
        "",
        f"{result.episodes} episodes per tier per agent, identical seeds for every agent. "
        f"{len(result.agents) * len(result.tiers) * result.episodes} episodes "
        f"in {result.elapsed:.1f}s.",
    ]
    return "\n".join(lines)


def format_diagnostics(result: Benchmark) -> str:
    """Means of two diagnostics, which are logged and never rewarded."""
    lines = ["agent".ljust(22) + "tier".ljust(10) + "service".ljust(10) + "spend/budget"]
    for agent in result.agents:
        for tier in result.tiers:
            cell = result.cell(agent, tier)
            lines.append(
                agent.ljust(22)
                + tier.ljust(10)
                + f"{cell.service:.3f}".ljust(10)
                + f"{cell.spend_ratio:.3f}"
            )
    return "\n".join(lines)


def check_invariants(result: Benchmark) -> list[str]:
    """Sanity checks: if one fails, the environment is broken rather than hard.

    A generator that produced impossible instances, or a verifier that handed out
    reward for nothing, shows up here rather than in a pass rate that merely looks
    difficult.
    """
    if result.episodes < MIN_SEEDS_FOR_INVARIANTS:
        return []

    problems: list[str] = []
    for tier in result.tiers:
        if "Random" in result.agents and result.rate("Random", tier) != 0.0:
            problems.append(
                f"random scores {result.rate('Random', tier):.2f} on {tier}: a policy that "
                "ignores the plan must not pass a conjunctive reward"
            )
        oracle = "Oracle (privileged)"
        if oracle in result.agents and result.rate(oracle, tier) != 1.0:
            problems.append(
                f"the oracle scores {result.rate(oracle, tier):.2f} on {tier}: every emitted "
                "scenario is supposed to be certified solvable"
            )
        if "Heuristic" in result.agents and not (
            result.rate("Heuristic", tier) > result.rate("Random", tier)
        ):
            problems.append(f"the heuristic does not beat random on {tier}")

    if "Heuristic" in result.agents and {"easy", "hard"} <= set(result.tiers):
        if not result.rate("Heuristic", "easy") > result.rate("Heuristic", "hard"):
            problems.append(
                "the heuristic does no better on easy than on hard: the tiers are not "
                "ordered by difficulty"
            )
        if result.rate("Heuristic", "easy") < 0.90:
            problems.append(
                f"the heuristic scores {result.rate('Heuristic', 'easy'):.2f} on easy: the "
                "easy tier must be solvable by a competent planner"
            )
    return problems


def success_rates(result: Benchmark) -> dict[str, dict[str, float]]:
    """``agent -> tier -> rate``, for the figure and for programmatic use."""
    return {
        agent: {
            **{tier: result.rate(agent, tier) for tier in result.tiers},
            "all": result.overall(agent).rate,
        }
        for agent in result.agents
    }


def iter_scores(result: Benchmark) -> Iterable[tuple[str, str, float]]:
    for (agent, tier), cell in result.cells.items():
        yield agent, tier, cell.rate
