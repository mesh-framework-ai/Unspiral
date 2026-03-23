"""Real-time sycophancy monitoring dashboard using Rich."""

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.console import Group


def health_bar(score: float, width: int = 20) -> Text:
    """Render a color-coded health bar."""
    filled = int(score * width)
    empty = width - filled
    if score >= 0.6:
        color = "green"
    elif score >= 0.4:
        color = "yellow"
    elif score >= 0.2:
        color = "dark_orange"
    else:
        color = "red"
    bar = Text()
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append(f" {score:.0%}", style=f"bold {color}")
    return bar


def sycophancy_bar(score: float, width: int = 15) -> Text:
    """Render a sycophancy score bar (higher = worse)."""
    filled = int(min(score, 1.0) * width)
    empty = width - filled
    if score <= 0.3:
        color = "green"
    elif score <= 0.5:
        color = "yellow"
    elif score <= 0.7:
        color = "dark_orange"
    else:
        color = "red"
    bar = Text()
    bar.append("▓" * filled, style=color)
    bar.append("░" * empty, style="dim")
    bar.append(f" {score:.2f}", style=f"bold {color}")
    return bar


def belief_sparkline(history: list[tuple[int, float, float]], width: int = 30) -> Text:
    """Render a sparkline of P(false belief) over turns."""
    if not history:
        return Text("no data", style="dim")
    chars = " ▁▂▃▄▅▆▇█"
    values = [p_false for _, p_false, _ in history]
    # Take last `width` values
    values = values[-width:]
    line = Text()
    for v in values:
        idx = int(min(v, 0.99) * len(chars))
        if v > 0.7:
            color = "red"
        elif v > 0.5:
            color = "yellow"
        else:
            color = "green"
        line.append(chars[idx], style=color)
    return line


def render_dashboard(
    turn: int,
    sycophancy_score: float,
    agreement_score: float,
    praise_score: float,
    hedging_ratio: float,
    health_score: float,
    health_level: str,
    p_false: float,
    pi_estimate: float,
    spiral_risk: str,
    intervention_level: str,
    belief_history: list[tuple[int, float, float]],
    stance_result: dict | None = None,
) -> Panel:
    """Render the full monitoring dashboard as a Rich Panel."""
    level_colors = {
        "green": "green", "yellow": "yellow",
        "orange": "dark_orange", "red": "red",
    }
    level_color = level_colors.get(health_level, "white")

    # Metrics table
    metrics = Table.grid(padding=(0, 2))
    metrics.add_column(justify="right", style="bold")
    metrics.add_column()

    metrics.add_row("Turn", Text(str(turn), style="cyan"))
    metrics.add_row("Sycophancy", sycophancy_bar(sycophancy_score))
    metrics.add_row("Agreement", sycophancy_bar(agreement_score))
    metrics.add_row("Praise", sycophancy_bar(praise_score))
    metrics.add_row("Hedging", Text(f"{hedging_ratio:.2f}", style="cyan"))
    metrics.add_row("Health", health_bar(health_score))
    metrics.add_row("Status", Text(f" {health_level.upper()} ", style=f"bold white on {level_color}"))

    # Belief panel
    belief = Table.grid(padding=(0, 2))
    belief.add_column(justify="right", style="bold")
    belief.add_column()

    risk_colors = {"low": "green", "medium": "yellow", "high": "dark_orange", "critical": "red"}
    risk_color = risk_colors.get(spiral_risk, "white")

    belief.add_row("P(false belief)", Text(f"{p_false:.3f}", style="cyan"))
    belief.add_row("Est. sycophancy π", Text(f"{pi_estimate:.3f}", style="cyan"))
    belief.add_row("Spiral risk", Text(f" {spiral_risk.upper()} ", style=f"bold white on {risk_color}"))
    belief.add_row("Trajectory", belief_sparkline(belief_history))

    if intervention_level != "none":
        int_color = level_colors.get(intervention_level, "white")
        belief.add_row(
            "Intervention",
            Text(f" {intervention_level.upper()} ACTIVE ", style=f"bold white on {int_color}"),
        )

    if stance_result and stance_result.get("tested"):
        rev = "YES" if stance_result.get("reversed") else "NO"
        rev_style = "bold red" if stance_result.get("reversed") else "bold green"
        belief.add_row("Stance reversal", Text(rev, style=rev_style))

    content = Columns([
        Panel(metrics, title="Detectors", border_style="blue", width=42),
        Panel(belief, title="Belief Tracking", border_style="magenta", width=42),
    ])

    return Panel(
        Group(content),
        title="[bold]Unspiral Monitor[/bold]",
        border_style="cyan",
        subtitle=f"[dim]turn {turn}[/dim]",
    )
