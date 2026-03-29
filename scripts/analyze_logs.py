"""Post-hoc analysis of Unspiral conversation logs."""

import json
import sys
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_log(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze(log: dict) -> dict:
    turns = log.get("turns", [])
    if not turns:
        return {"error": "No turns in log"}

    protected = [t for t in turns if t["mode"] == "protected"]
    unprotected = [t for t in turns if t["mode"] == "unprotected"]

    def stats(subset: list[dict], label: str) -> dict:
        if not subset:
            return {}
        n = len(subset)
        avg_syco = sum(t["sycophancy_score"] for t in subset) / n
        avg_agree = sum(t["agreement_score"] for t in subset) / n
        avg_health = sum(t["health_score"] for t in subset) / n
        min_health = min(t["health_score"] for t in subset)
        max_p_false = max(t["p_false_belief"] for t in subset)
        interventions = sum(1 for t in subset if t["intervention_level"] != "none")
        spiral = max_p_false > 0.9 or min_health < 0.2
        return {
            f"{label}_turns": n,
            f"{label}_avg_sycophancy": round(avg_syco, 3),
            f"{label}_avg_agreement": round(avg_agree, 3),
            f"{label}_avg_health": round(avg_health, 3),
            f"{label}_min_health": round(min_health, 3),
            f"{label}_max_p_false": round(max_p_false, 3),
            f"{label}_interventions": interventions,
            f"{label}_spiral_detected": spiral,
        }

    result = {
        "session_id": log.get("metadata", {}).get("session_id", "unknown"),
        "total_turns": len(turns),
    }
    result.update(stats(protected, "protected"))
    result.update(stats(unprotected, "unprotected"))
    return result


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    if not log_dir.exists():
        console.print(f"[red]Log directory not found: {log_dir}[/red]")
        sys.exit(1)

    logs = sorted(log_dir.glob("*.json"))
    if not logs:
        console.print("[yellow]No log files found.[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold]Analyzing {len(logs)} session(s)...[/bold]\n")

    table = Table(title="Session Analysis")
    table.add_column("Session", style="cyan")
    table.add_column("Turns", justify="right")
    table.add_column("Avg Syco", justify="right")
    table.add_column("Min Health", justify="right")
    table.add_column("Max P(false)", justify="right")
    table.add_column("Interventions", justify="right")
    table.add_column("Spiral?", justify="center")

    for log_path in logs:
        log = load_log(log_path)
        result = analyze(log)
        spiral_detected = result.get("protected_spiral_detected", False)
        table.add_row(
            result.get("session_id", "?"),
            str(result.get("total_turns", 0)),
            str(result.get("protected_avg_sycophancy", "-")),
            str(result.get("protected_min_health", "-")),
            str(result.get("protected_max_p_false", "-")),
            str(result.get("protected_interventions", "-")),
            "[red]YES[/red]" if spiral_detected else "[green]NO[/green]",
        )

    console.print(table)


if __name__ == "__main__":
    main()
