"""``mrpenv`` command line.

mrpenv serve                     the HTTP API and the browser play UI
mrpenv play                      play an episode yourself, in the terminal
mrpenv run                       watch a built-in agent play one episode
mrpenv eval                      the benchmark: table, figure, sanity checks
mrpenv exploits                  the two documented reward exploits
mrpenv verify final_state.json   score an artifact offline
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import TIERS, load_env_config
from .core.env import MRPEnv
from .core.types import FinalState, ToolCall

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
# the benchmark table needs about 100 columns; rich falls back to 80 when stdout
# is a pipe, which would wrap it into something unreadable in logs
console = Console(width=max(100, shutil.get_terminal_size((120, 40)).columns))


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="bind address (use 0.0.0.0 in a container)"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="auto-reload on source changes"),
) -> None:
    """Run the HTTP API and the browser play UI."""
    from .server import serve as _serve

    console.print(f"[bold]Material Restocking Environment[/] on http://{host}:{port}")
    console.print("open that address in a browser to play the environment by hand")
    _serve(host=host, port=port, reload=reload)


# --------------------------------------------------------------------------- #
# play
# --------------------------------------------------------------------------- #

REPL_HELP = """\
[bold]commands[/]
  po <sku> <qty>   place a purchase order        inv      recorded inventory + open POs
  next             advance one day               prices   today's quotes and the history
  runout           advance to the last day       plan     forecast plan and requirements
  verify           score the final state         master   supplier master data
  help             this list                     quit     leave
"""


def _render(obs: Any, static: Any) -> None:
    table = Table(title=f"day {obs.day} / {obs.horizon}", title_style="bold")
    for column in ("sku", "recorded", "on order", "lead", "MOQ/lot", "EUR/pc today"):
        table.add_column(column, justify="left" if column == "sku" else "right")
    suppliers = {s.sku: s for s in static.suppliers}
    for row in obs.inventory:
        supplier = suppliers[row.sku]
        table.add_row(
            row.sku,
            str(row.recorded_on_hand),
            str(row.on_order),
            str(supplier.lead_time_days),
            f"{supplier.moq}/{supplier.lot_size}",
            f"{row.unit_price_today:.2f}",
            style="red" if row.recorded_on_hand < 0 else None,
        )
    console.print(table)
    console.print(
        f"spend [bold]{obs.spend:.2f}[/] / {static.budget_eur:.2f} EUR   "
        f"recorded volume {obs.recorded_utilisation * 100:.0f}% of {static.capacity_m3:.2f} m3   "
        f"steps {obs.steps_used}/{obs.step_limit}"
    )
    if obs.open_purchase_orders:
        console.print(
            "open POs: "
            + ", ".join(
                f"{po['po_id']}:{po['sku']}x{po['qty']}@day{po['promised_arrival']}"
                for po in obs.open_purchase_orders
            )
        )
    for line in obs.last_events[-4:]:
        console.print(f"  [dim]{line}[/]")


def _verdict(env: MRPEnv) -> None:
    result = env.verify()
    style = "green" if result.score else "red"
    console.print(f"[{style}]verifier score {result.score}[/]")
    for sentence in result.explanations:
        console.print(f"  - {sentence}")


@app.command()
def play(
    tier: str = typer.Option("medium", help=f"one of {', '.join(TIERS)}"),
    seed: int = typer.Option(0),
) -> None:
    """Play an episode from the terminal."""
    env = MRPEnv(tier, seed)
    static = env.static()
    console.print(Panel(env.instruction, title=f"{tier} / seed {seed}", expand=False))
    console.print(REPL_HELP)
    _render(env.observation(), static)

    while True:
        try:
            raw = console.input("[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not raw:
            continue
        parts = raw.split()
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("quit", "exit", "q"):
            return
        if cmd in ("help", "h", "?"):
            console.print(REPL_HELP)
            continue
        if cmd == "verify":
            _verdict(env)
            continue
        if cmd == "master":
            console.print(static.model_dump())
            continue
        if cmd == "runout":
            while not env.done:
                env.step(ToolCall(tool="advance_day"))
            _render(env.observation(), static)
            _verdict(env)
            return

        call: ToolCall | None = None
        if cmd == "inv":
            call = ToolCall(tool="view_inventory")
        elif cmd == "next":
            call = ToolCall(tool="advance_day")
        elif cmd == "prices":
            call = ToolCall(tool="view_prices")
        elif cmd == "plan":
            call = ToolCall(tool="view_plan")
        elif cmd == "po" and len(args) == 2:
            try:
                qty: Any = int(args[1])
            except ValueError:
                qty = args[1]
            call = ToolCall(tool="create_purchase_order", args={"sku": args[0].upper(), "qty": qty})
        if call is None:
            console.print("[yellow]unrecognised command[/]; try 'help'")
            continue

        result = env.step(call)
        if "error" in result.tool_output:
            console.print(f"[red]{result.tool_output['error']}[/]: {result.tool_output['message']}")
        elif cmd not in ("next", "inv"):
            console.print(result.tool_output)
        _render(result.observation, static)
        if result.terminated or result.truncated:
            _verdict(env)
            return


# --------------------------------------------------------------------------- #
# run one agent
# --------------------------------------------------------------------------- #


@app.command(name="run")
def run_agent(
    tier: str = typer.Option("hard", help=f"one of {', '.join(TIERS)}"),
    seed: int = typer.Option(7),
    agent: str = typer.Option("heuristic", help="heuristic | mrp | oracle | random"),
    show: int = typer.Option(10, help="how many of the last events to print"),
) -> None:
    """Roll one episode with a built-in agent and print what happened.

    The quickest way to see the environment work end to end: an agent reading only
    the ERP's records, and the verifier's verdict on the hidden true stock.
    """
    from .eval.agents import HeuristicAgent, OracleAgent, RandomAgent

    builders: dict[str, Callable[[], Any]] = {
        "heuristic": lambda: HeuristicAgent(
            safety_days=1.5, base_lead_pad=1, adapt=True, price_policy="opportunistic"
        ),
        "mrp": lambda: HeuristicAgent(safety_days=0.0, adapt=False),
        "oracle": lambda: OracleAgent(tier, seed),
        "random": lambda: RandomAgent(seed=seed),
    }
    if agent not in builders:
        console.print(f"[red]unknown agent {agent!r}[/]; try one of {', '.join(builders)}")
        raise typer.Exit(2)

    env = MRPEnv(tier, seed)
    console.print(Panel(env.instruction, title=f"{tier} / seed {seed}", expand=False))
    result = env.run(builders[agent]())

    static, obs = env.static(), env.observation()
    console.print(
        f"[bold]{agent}[/] used {obs.steps_used} of {obs.step_limit} tool calls over "
        f"{static.horizon} days, placed {int(result.diagnostics['purchase_orders'])} purchase "
        f"orders and spent {result.diagnostics['spend']:.2f} of {static.budget_eur:.2f} EUR"
    )
    for line in obs.last_events[-show:]:
        console.print(f"  [dim]{line}[/]")
    console.print()
    _verdict(env)
    console.print()
    console.print(
        "service {service_level:.0%} | peak volume {peak_over_capacity:.0%} of capacity | "
        "spend {spend_over_budget:.0%} of budget | late deliveries {late_orders:.0f} | "
        "rush-order units {rush_order_units:.0f} | max record error "
        "{max_record_error:.0f}".format(**result.diagnostics)
    )


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #


@app.command(name="eval")
def eval_cmd(
    episodes: int = typer.Option(100, help="episodes per tier per agent"),
    figure: str = typer.Option("benchmark.png", help="where to write the figure"),
    diagnostics: bool = typer.Option(False, help="also print the logged diagnostics"),
) -> None:
    """The benchmark: every agent over every tier, with the figure and sanity checks."""
    from .eval.benchmark import (
        AGENTS,
        MIN_SEEDS_FOR_INVARIANTS,
        check_invariants,
        format_diagnostics,
        format_table,
        run_benchmark,
    )

    console.rule("[bold]Material Restocking Environment")
    cfg = load_env_config()
    console.print(
        f"invalid_action_mode={cfg.invalid_action_mode}  max_invalid={cfg.max_invalid}  "
        f"step_limit=12H+40"
    )
    console.print(
        f"running {len(AGENTS)} agents x {len(TIERS)} tiers x {episodes} seeds "
        f"= {len(AGENTS) * len(TIERS) * episodes} episodes"
    )

    result = run_benchmark(episodes=episodes, progress=lambda m: console.print(f"  [dim]{m}[/]"))
    console.print()
    console.print(format_table(result))
    if diagnostics:
        console.print()
        console.print(format_diagnostics(result))

    if figure:
        from .eval.plot import plot_benchmark

        console.print(f"\nwrote {plot_benchmark(result, figure)}")

    console.print()
    problems = check_invariants(result)
    if result.episodes < MIN_SEEDS_FOR_INVARIANTS:
        console.print(
            f"[dim]({result.episodes} episodes is too few to check the sanity invariants; "
            f"use at least {MIN_SEEDS_FOR_INVARIANTS})[/]"
        )
    elif problems:
        console.print("[red]sanity invariants violated - the environment is broken, not hard:[/]")
        for problem in problems:
            console.print(f"  [red]x[/] {problem}")
        raise typer.Exit(1)
    else:
        console.print(
            "[green]all invariants hold[/]: random never passes, the oracle always does, "
            "the heuristic beats random on every tier, and difficulty rises with the tier"
        )


# --------------------------------------------------------------------------- #
# exploits
# --------------------------------------------------------------------------- #


@app.command()
def exploits() -> None:
    """Show the two naive rewards scoring 1 where the verifier scores 0."""
    from .eval.exploits import (
        LateBulkOrderAgent,
        TrustRecordsAgent,
        coverage_reward,
        recorded_service_reward,
    )

    env = MRPEnv("medium", 3)
    result = env.run(LateBulkOrderAgent())
    console.print(
        Panel.fit(
            json.dumps(
                {
                    "naive reward": "coverage: did we buy enough of everything?",
                    "wrong solution": "order the whole requirement on the last day",
                    "tier/seed": "medium/3",
                    "naive score": coverage_reward(env.final_state()),
                    "verifier score": result.score,
                    "verifier says": next(iter(result.explanations), ""),
                },
                indent=2,
            ),
            title="E1 quantity coverage",
        )
    )

    for seed in range(50):
        env = MRPEnv("hard", seed)
        result = env.run(TrustRecordsAgent())
        if recorded_service_reward(env.final_state()) == 1 and result.score == 0:
            console.print(
                Panel.fit(
                    json.dumps(
                        {
                            "naive reward": "service checked against the ERP's records",
                            "wrong solution": "plan correctly and believe the stock figures",
                            "tier/seed": f"hard/{seed}",
                            "naive score": 1,
                            "verifier score": 0,
                            "verifier says": next(iter(result.explanations), ""),
                            "max record error": result.diagnostics["max_record_error"],
                        },
                        indent=2,
                    ),
                    title="E2 trust the records",
                )
            )
            return
    console.print("[yellow]E2 did not reproduce in seeds 0-49[/]")


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


@app.command()
def verify(path: Path = typer.Argument(..., help="path to a final_state.json")) -> None:
    """Verify a final-state artifact offline."""
    from .verifier.verify import verify as _verify

    result = _verify(FinalState(**json.loads(path.read_text())))
    style = "green" if result.score else "red"
    console.print(f"[{style}]score {result.score}[/]")
    for sentence in result.explanations:
        console.print(f"  - {sentence}")
    console.print(result.diagnostics)
    raise typer.Exit(0 if result.score == 1 else 1)


if __name__ == "__main__":  # pragma: no cover
    app()
