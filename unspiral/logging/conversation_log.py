import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TurnLog:
    turn: int
    timestamp: str
    user_message: str
    bot_response: str
    mode: str                      # "protected" or "unprotected"
    sycophancy_score: float
    agreement_score: float
    praise_score: float
    hedging_ratio: float
    health_score: float
    health_level: str
    p_false_belief: float
    pi_estimate: float
    intervention_level: str
    intervention_text: str | None
    stance_test_result: dict | None


class ConversationLog:
    """JSON conversation logger for post-hoc analysis."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turns: list[TurnLog] = []
        self.metadata: dict[str, Any] = {
            "session_id": self.session_id,
            "started": datetime.now().isoformat(),
        }

    def log_turn(self, turn: TurnLog) -> None:
        """Append a turn to the log."""
        self.turns.append(turn)

    def save(self) -> Path:
        """Write the full conversation log to JSON file."""
        filename = self.log_dir / f"conversation_{self.session_id}.json"
        payload = {
            "metadata": self.metadata,
            "turns": [asdict(t) for t in self.turns],
        }
        with filename.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return filename

    def summary(self) -> dict[str, Any]:
        """Return summary stats: avg sycophancy, min health, spiral detected, etc."""
        if not self.turns:
            return {
                "session_id": self.session_id,
                "total_turns": 0,
                "avg_sycophancy_score": None,
                "avg_agreement_score": None,
                "min_health_score": None,
                "max_p_false_belief": None,
                "spiral_detected": False,
                "intervention_counts": {"none": 0, "yellow": 0, "orange": 0, "red": 0},
                "modes": {},
            }

        syco_scores = [t.sycophancy_score for t in self.turns]
        agree_scores = [t.agreement_score for t in self.turns]
        health_scores = [t.health_score for t in self.turns]
        p_false_values = [t.p_false_belief for t in self.turns]

        intervention_counts: dict[str, int] = {"none": 0, "yellow": 0, "orange": 0, "red": 0}
        for t in self.turns:
            level = t.intervention_level
            intervention_counts[level] = intervention_counts.get(level, 0) + 1

        mode_counts: dict[str, int] = {}
        for t in self.turns:
            mode_counts[t.mode] = mode_counts.get(t.mode, 0) + 1

        min_health = min(health_scores)
        max_p_false = max(p_false_values)

        return {
            "session_id": self.session_id,
            "total_turns": len(self.turns),
            "avg_sycophancy_score": sum(syco_scores) / len(syco_scores),
            "avg_agreement_score": sum(agree_scores) / len(agree_scores),
            "min_health_score": min_health,
            "max_p_false_belief": max_p_false,
            "spiral_detected": min_health < 0.2 or max_p_false > 0.9,
            "intervention_counts": intervention_counts,
            "modes": mode_counts,
        }
