"""The benchmark figure: the heuristic agent's success rate by tier.

Deliberately the simplest form that answers the question: one series, three
ordered categories, a column chart. The random control is a flat zero on every
tier, so plotting it as a second series would spend a colour and a legend on three
invisible marks - it reads better as a line of text under the axis, generated from
the same measurement.

House rules applied here:

* one series means **one hue** for every column (categorical slot 1) and **no
  legend box** - the title says what is plotted. A darker-where-bigger ramp would
  double-encode the bar length and is an anti-pattern;
* columns capped thin, the band's leftover left as air;
* a direct value label on every column, so magnitude is never bar-length-only;
* recessive hairline grid on the value axis only, no top or right spine;
* text in ink tokens, never in the series colour;
* dark mode is a selected second set of steps against the dark surface, not an
  inversion of the light one.
"""

from __future__ import annotations

from pathlib import Path

from .benchmark import Benchmark

# Validated with a palette checker before use: both modes pass the lightness band,
# the chroma floor, colour-vision-deficiency separation (worst adjacent dE 24.7
# light / 26.8 dark), the normal-vision floor and 3:1 contrast against their
# surface. ``series_2`` is kept for a future second series rather than deleted, so
# the pair stays validated together.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_soft": "#52514e",
        "grid": "#e3e3e0",
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_soft": "#c3c2b7",
        "grid": "#2f2f2d",
        "series_1": "#3987e5",
        "series_2": "#d95926",
    },
}

TIER_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}


#: One high-level sentence on what the plotted policy does. Kept short on purpose:
#: the figure's job is the numbers, not the algorithm.
HEURISTIC_BLURB = (
    "The heuristic projects the ERP's recorded stock against the forecast plan and\n"
    "orders at the last responsible moment, carrying buffers for late deliveries and\n"
    "rush orders and buying early when the prices it has seen look cheap."
)


def control_footnote(result: Benchmark, control: str = "Random") -> str:
    """One line describing the control condition, built from its measured rates.

    Generated rather than written, so the figure cannot end up claiming a zero
    that the data no longer supports.
    """
    if control not in result.agents:
        return ""
    rates = [result.rate(control, tier) for tier in result.tiers]
    if all(rate == 0.0 for rate in rates):
        return (
            f"The random agent scores 0.00 on all tiers (0 of {result.episodes} episodes on each)."
        )
    listed = ", ".join(f"{tier} {rate:.2f}" for tier, rate in zip(result.tiers, rates, strict=True))
    return f"The random agent scores {listed}."


def plot_benchmark(
    result: Benchmark,
    out_path: str | Path = "benchmark.png",
    mode: str = "light",
    dpi: int = 200,
    agent: str = "Heuristic",
) -> Path:
    """Render the column chart for one agent. Returns the path written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme = THEMES[mode]
    tiers = result.tiers
    rates = [result.rate(agent, tier) for tier in tiers]

    # Width is chosen so the columns land at the 24px-at-1x cap without looking
    # spindly: three categories across a narrow frame keeps them close together
    # while the band's leftover stays as air.
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=dpi)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    positions = list(range(len(tiers)))
    bars = ax.bar(
        positions,
        rates,
        width=0.17,  # ~24px at 1x; the band's leftover is left as air
        color=theme["series_1"],  # one series, one hue - never a ramp on the tiers
        edgecolor="none",
        zorder=3,
    )
    for bar, rate in zip(bars, rates, strict=True):
        ax.annotate(
            f"{rate:.2f}",
            (bar.get_x() + bar.get_width() / 2, rate),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=11,
            color=theme["ink"],  # a text token, never the series colour
        )

    # headroom so the label on a near-perfect column never touches the subtitle
    ax.set_ylim(0, 1.09)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Success rate", fontsize=10, color=theme["ink_soft"], labelpad=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([TIER_LABELS.get(t, t.title()) for t in tiers], fontsize=11)

    ax.yaxis.grid(True, color=theme["grid"], linewidth=1.0, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme["ink_soft"], labelsize=10, length=0)

    ax.set_title(
        f"{agent} agent: success rate by tier",
        fontsize=13,
        color=theme["ink"],
        loc="left",
        pad=16,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.045,
        f"{result.episodes} episodes per tier, binary verifier score. "
        f"Tiers in increasing order of difficulty.",
        transform=ax.transAxes,
        fontsize=9,
        color=theme["ink_soft"],
    )

    caption = "\n".join(part for part in (control_footnote(result), HEURISTIC_BLURB) if part)
    if caption:
        ax.text(
            0.0,
            -0.16,
            caption,
            transform=ax.transAxes,
            fontsize=8.5,
            linespacing=1.6,
            va="top",
            color=theme["ink_soft"],
        )

    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    return path
