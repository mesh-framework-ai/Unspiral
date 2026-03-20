from dataclasses import dataclass, field


@dataclass
class Intervention:
    level: str                   # "none", "yellow", "orange", "red"
    system_injection: str | None  # Added to system prompt
    response_suffix: str | None   # Appended to bot response
    user_warning: str | None      # Shown directly to user
    triggered_at_turn: int


class CircuitBreaker:
    """Three-tier intervention system based on conversation health.

    Yellow (health < 0.70): Push back on unsupported claims
    Orange (health < 0.55): Directly challenge claims, no diplomacy
    Red (health < 0.35): Emergency — refuse to validate false beliefs
    """

    YELLOW_SYSTEM = (
        "The user may be developing a false belief reinforced by your previous responses. "
        "Push back on unsupported claims. Do not validate feelings over facts. "
        "Cite specific counterevidence. Do NOT begin with 'It's understandable' or "
        "'I hear you' or any other validating opener."
    )

    ORANGE_SYSTEM = (
        "CRITICAL: The user is in a sycophantic spiral. You MUST directly challenge "
        "their claims. Do NOT say 'it's understandable' or 'I hear you.' State plainly "
        "what the scientific consensus is and where the user's reasoning is flawed. "
        "Be direct, not diplomatic. If the user's plan could cause harm, say so explicitly. "
        "Do NOT present their position as one valid option among many."
    )

    RED_SYSTEM = (
        "EMERGENCY: This conversation has reached dangerous levels of false belief "
        "reinforcement. Your response MUST: (1) Clearly state the user's belief is not "
        "supported by evidence, (2) Name the specific risks of acting on this belief, "
        "(3) Strongly recommend consulting qualified professionals. "
        "Do NOT validate any aspect of the false belief. Do NOT be diplomatic. "
        "Be blunt about the danger."
    )

    RED_WARNING = (
        "\u26a0\ufe0f  UNSPIRAL WARNING: This conversation shows strong signs of belief "
        "reinforcement spiraling. The AI\u2019s responses have been overly agreeable "
        "and may have reinforced false or harmful beliefs. "
        "STOP and consult independent sources or a qualified professional."
    )

    # Thresholds
    YELLOW_THRESHOLD = 0.70
    ORANGE_THRESHOLD = 0.55
    RED_THRESHOLD = 0.35

    # Override thresholds
    SYCO_SPIKE_THRESHOLD = 0.8       # single-turn sycophancy spike → ORANGE
    SYCO_SUSTAINED_THRESHOLD = 0.5   # sustained sycophancy + turns > 3 → ORANGE

    def __init__(self):
        self.interventions: list[Intervention] = []
        self.active_level: str = "none"

    def evaluate(
        self,
        health_score: float,
        turn: int,
        sycophancy_score: float = 0.0,
        agreement_score: float = 0.0,
    ) -> Intervention:
        """Determine intervention level with overrides for sycophancy spikes."""
        # Base level from health score
        if health_score < self.RED_THRESHOLD:
            level = "red"
        elif health_score < self.ORANGE_THRESHOLD:
            level = "orange"
        elif health_score < self.YELLOW_THRESHOLD:
            level = "yellow"
        else:
            level = "none"

        # Override 1: sycophancy spike on any single turn → at least ORANGE
        if sycophancy_score >= self.SYCO_SPIKE_THRESHOLD:
            if level in ("none", "yellow"):
                level = "orange"

        # Override 2: sustained sycophancy after turn 3 → at least ORANGE
        if sycophancy_score >= self.SYCO_SUSTAINED_THRESHOLD and turn > 3:
            if level in ("none", "yellow"):
                level = "orange"

        # Override 3: high agreement sustained after turn 5 → at least YELLOW
        if agreement_score > 0.6 and turn > 5:
            if level == "none":
                level = "yellow"

        # Build intervention
        if level == "red":
            system_injection = self.RED_SYSTEM
            response_suffix = self._red_suffix()
            user_warning = self.RED_WARNING
        elif level == "orange":
            system_injection = self.ORANGE_SYSTEM
            response_suffix = self._orange_suffix()
            user_warning = None
        elif level == "yellow":
            system_injection = self.YELLOW_SYSTEM
            response_suffix = None
            user_warning = None
        else:
            system_injection = None
            response_suffix = None
            user_warning = None

        self.active_level = level
        intervention = Intervention(
            level=level,
            system_injection=system_injection,
            response_suffix=response_suffix,
            user_warning=user_warning,
            triggered_at_turn=turn,
        )
        self.interventions.append(intervention)
        return intervention

    def get_system_injection(self) -> str | None:
        """Get the current system prompt injection based on active level."""
        mapping = {
            "yellow": self.YELLOW_SYSTEM,
            "orange": self.ORANGE_SYSTEM,
            "red": self.RED_SYSTEM,
        }
        return mapping.get(self.active_level)

    def get_response_suffix(self) -> str | None:
        if self.active_level == "orange":
            return self._orange_suffix()
        if self.active_level == "red":
            return self._red_suffix()
        return None

    def get_user_warning(self) -> str | None:
        if self.active_level == "red":
            return self.RED_WARNING
        return None

    @staticmethod
    def _orange_suffix() -> str:
        return (
            "\n\n---\n\n**\u26a0\ufe0f Sycophancy Alert:** This response was generated with "
            "anti-sycophancy safeguards active. The AI detected signs of validation bias "
            "in this conversation. Counterarguments have been appended below."
        )

    @staticmethod
    def _red_suffix() -> str:
        return (
            "\n\n---\n\n**\U0001f6d1 CRITICAL WARNING:** This conversation has triggered "
            "emergency intervention. The beliefs being discussed may be harmful if acted upon. "
            "Please consult qualified professionals before making any decisions based on "
            "this conversation."
        )
