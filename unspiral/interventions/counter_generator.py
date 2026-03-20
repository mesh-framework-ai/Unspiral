class CounterGenerator:
    """Generate devil's advocate counterarguments using OpenAI.

    Used at orange+ intervention levels to append counterpoints to responses.
    """

    def __init__(self, client):  # openai.OpenAI client
        self.client = client

    def extract_claims(self, bot_response: str) -> list[str]:
        """Extract the main factual/opinion claims from a bot response.
        Uses OpenAI to parse claims. Returns list of claim strings."""
        if not bot_response.strip():
            return []

        prompt = (
            "Extract the main factual and opinion claims from the following text. "
            "Return each claim as a separate line, with no numbering or bullet points. "
            "Only include substantive claims, not filler phrases.\n\n"
            f"Text:\n{bot_response}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )

        raw = response.choices[0].message.content or ""
        claims = [line.strip() for line in raw.splitlines() if line.strip()]
        return claims

    def generate_counter(self, claims: list[str], conversation_context: str = "") -> str:
        """Generate a concise, evidence-based counterargument.

        Prompt template:
        'Given these claims: {claims}
        Generate a brief (2-3 sentence) counterargument that:
        1. Cites the strongest opposing evidence
        2. Identifies potential biases or missing context
        3. Suggests what a skeptic would say
        Be factual and specific, not dismissive.'

        Returns the counterargument text."""
        if not claims:
            return ""

        claims_text = "\n".join(f"- {c}" for c in claims)
        context_block = (
            f"\nConversation context:\n{conversation_context}\n"
            if conversation_context.strip()
            else ""
        )

        prompt = (
            f"Given these claims:\n{claims_text}"
            f"{context_block}\n"
            "Generate a brief (2-3 sentence) counterargument that:\n"
            "1. Cites the strongest opposing evidence\n"
            "2. Identifies potential biases or missing context\n"
            "3. Suggests what a skeptic would say\n"
            "Be factual and specific, not dismissive."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )

        return (response.choices[0].message.content or "").strip()

    def format_intervention(self, counter: str) -> str:
        """Format the counterargument for display.
        Wraps in a box: '--- Alternative Perspective ---\\n{counter}\\n---'"""
        if not counter.strip():
            return ""
        return f"--- Alternative Perspective ---\n{counter}\n---"
