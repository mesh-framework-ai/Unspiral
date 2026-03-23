"""Unspiral CLI — Anti-sycophancy chat interface powered by OpenAI."""

import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.markdown import Markdown
import openai

from unspiral.detectors.lqcd_probe import LQCDProbe
from unspiral.detectors.agreement_classifier import AgreementClassifier
from unspiral.detectors.stance_test import StanceTest
from unspiral.tracking.belief_tracker import BeliefTracker
from unspiral.tracking.drift_monitor import DriftMonitor
from unspiral.interventions.circuit_breaker import CircuitBreaker
from unspiral.interventions.counter_generator import CounterGenerator
from unspiral.logging.conversation_log import ConversationLog, TurnLog
from unspiral.models.embeddings import embed
from unspiral.cli.dashboard import render_dashboard

console = Console()

BANNER = r"""
[cyan]
 _   _                 _           _
| | | |_ __  ___ _ __ (_)_ __ __ _| |
| | | | '_ \/ __| '_ \| | '__/ _` | |
| |_| | | | \__ \ |_) | | | | (_| | |
 \___/|_| |_|___/ .__/|_|_|  \__,_|_|
                 |_|
[/cyan]
[dim]ML-powered anti-sycophancy framework[/dim]
"""

SYSTEM_BASE = (
    "You are a helpful assistant. Answer the user's questions honestly and thoroughly."
)


def get_api_key() -> str:
    """Load or prompt for OpenAI API key."""
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if key and key != "sk-your-key-here":
        return key

    console.print("\n[yellow]No OpenAI API key found.[/yellow]")
    console.print("Enter your key (it will be saved to .env):\n")
    key = Prompt.ask("[bold]OPENAI_API_KEY[/bold]", password=True)

    env_path = Path(".env")
    env_path.write_text(f"OPENAI_API_KEY={key}\n")
    os.environ["OPENAI_API_KEY"] = key
    console.print("[green]Key saved to .env[/green]\n")
    return key


def select_mode() -> str:
    """Let user choose protected or unprotected mode."""
    console.print("\n[bold]Select mode:[/bold]")
    console.print("  [cyan]1[/cyan] — Protected (sycophancy detection + interventions)")
    console.print("  [cyan]2[/cyan] — Unprotected (raw ChatGPT, control group)")
    console.print("  [cyan]3[/cyan] — Side-by-side (both modes, compare outputs)\n")

    choice = Prompt.ask("Mode", choices=["1", "2", "3"], default="1")
    return {"1": "protected", "2": "unprotected", "3": "sidebyside"}[choice]


def chat_completion(
    client: openai.OpenAI,
    messages: list[dict],
    system: str = SYSTEM_BASE,
    model: str = "gpt-4o",
) -> str:
    """Get a chat completion from OpenAI."""
    full_messages = [{"role": "system", "content": system}] + messages
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=full_messages, temperature=0.7, max_tokens=1024
            )
            msg = resp.choices[0].message
            text = (msg.content or "").strip() or getattr(msg, "reasoning_content", None)
            if text:
                return text
            if attempt < 2:
                console.print(f"[dim]Empty response, retrying ({attempt + 2}/3)...[/dim]")
                continue
            return "[No response after 3 attempts]"
        except openai.OpenAIError as e:
            console.print(f"[red]OpenAI error: {e}[/red]")
            return "[Error generating response]"
    return "[No response]"


def run_protected_turn(
    client: openai.OpenAI,
    user_msg: str,
    history: list[dict],
    turn: int,
    lqcd: LQCDProbe,
    agreement_cls: AgreementClassifier,
    stance: StanceTest,
    belief: BeliefTracker,
    drift: DriftMonitor,
    breaker: CircuitBreaker,
    counter_gen: CounterGenerator,
    logger: ConversationLog,
) -> str:
    """Run a single turn through the full unspiral pipeline."""
    # 1. Get system injection from circuit breaker
    system = SYSTEM_BASE
    injection = breaker.get_system_injection()
    if injection:
        system = f"{system}\n\n{injection}"

    # 2. Get bot response
    history.append({"role": "user", "content": user_msg})
    bot_response = chat_completion(client, history, system=system)
    history.append({"role": "assistant", "content": bot_response})

    # 3. Run detectors
    with console.status("[cyan]Analyzing response...[/cyan]", spinner="dots"):
        # LQCD probe
        lqcd_result = lqcd.score(user_msg, history)
        syco_score = lqcd_result["sycophancy_score"]

        # Agreement classifier
        agree_result = agreement_cls.score(user_msg, bot_response)

        # Embeddings for drift
        msg_embedding = embed(user_msg)
        if turn == 1:
            drift.set_initial_topic(msg_embedding)

        # Belief tracker update
        belief_state = belief.update(syco_score, agree_result["agreement"], turn)

        # Drift monitor update
        health_snap = drift.update(
            syco_score, agree_result["agreement"],
            belief.p_false_belief, msg_embedding, turn,
        )

        # Circuit breaker (now receives sycophancy + agreement for overrides)
        intervention = breaker.evaluate(
            health_snap.health_score, turn,
            sycophancy_score=syco_score,
            agreement_score=agree_result["agreement"],
        )

        # Stance test (periodic)
        stance_result = None
        if stance.should_test():
            stance_result = stance.test_reversal(history, bot_response)

    # 4. Apply interventions to response
    display_response = bot_response

    if intervention.level == "orange":
        # ORANGE: Regenerate with hardened prompt for visibly different response
        console.print("[dim]Regenerating with anti-sycophancy safeguards...[/dim]")
        hardened_response = chat_completion(
            client, history[:-1],  # exclude the sycophantic response
            system=f"{SYSTEM_BASE}\n\n{breaker.ORANGE_SYSTEM}",
        )
        # Replace the assistant response in history too
        history[-1] = {"role": "assistant", "content": hardened_response}
        counter_text = counter_gen.generate_counter(
            counter_gen.extract_claims(hardened_response)
        )
        suffix = counter_gen.format_intervention(counter_text)
        display_response = f"{hardened_response}\n\n{suffix}"

    elif intervention.level == "red":
        # RED: Regenerate with emergency prompt + show warning
        console.print("[dim]Emergency intervention active...[/dim]")
        hardened_response = chat_completion(
            client, history[:-1],
            system=f"{SYSTEM_BASE}\n\n{breaker.RED_SYSTEM}",
        )
        history[-1] = {"role": "assistant", "content": hardened_response}
        counter_text = counter_gen.generate_counter(
            counter_gen.extract_claims(hardened_response)
        )
        suffix = counter_gen.format_intervention(counter_text)
        display_response = f"{hardened_response}\n\n{suffix}"
        console.print(Panel(
            intervention.user_warning,
            border_style="red", title="[bold red]\u26a0\ufe0f  CRITICAL WARNING[/bold red]",
        ))

    # Show sycophancy alert panel at YELLOW+
    if intervention.level in ("yellow", "orange", "red"):
        alert_color = {"yellow": "yellow", "orange": "dark_orange", "red": "red"}[intervention.level]
        alert_label = intervention.level.upper()
        console.print(Panel(
            f"[bold]Sycophancy level: {alert_label}[/bold]\n"
            f"Health: {health_snap.health_score:.0%} | "
            f"P(false belief): {belief.p_false_belief:.0%} | "
            f"Est. sycophancy \u03c0: {belief.expected_pi:.0%}\n"
            f"The AI may be reinforcing beliefs rather than challenging them.",
            border_style=alert_color,
            title=f"[bold {alert_color}]\u26a0\ufe0f  Sycophancy Alert — {alert_label}[/bold {alert_color}]",
        ))

    # 5. Render dashboard
    dashboard = render_dashboard(
        turn=turn,
        sycophancy_score=syco_score,
        agreement_score=agree_result["agreement"],
        praise_score=agree_result["praise"],
        hedging_ratio=agree_result["hedging"],
        health_score=health_snap.health_score,
        health_level=health_snap.level,
        p_false=belief.p_false_belief,
        pi_estimate=belief.expected_pi,
        spiral_risk=belief.spiral_risk,
        intervention_level=intervention.level,
        belief_history=belief.trajectory(),
        stance_result=stance_result,
    )
    console.print(dashboard)

    # 6. Log
    logger.log_turn(TurnLog(
        turn=turn,
        timestamp=datetime.now().isoformat(),
        user_message=user_msg,
        bot_response=bot_response,
        mode="protected",
        sycophancy_score=syco_score,
        agreement_score=agree_result["agreement"],
        praise_score=agree_result["praise"],
        hedging_ratio=agree_result["hedging"],
        health_score=health_snap.health_score,
        health_level=health_snap.level,
        p_false_belief=belief.p_false_belief,
        pi_estimate=belief.expected_pi,
        intervention_level=intervention.level,
        intervention_text=intervention.system_injection,
        stance_test_result=stance_result,
    ))

    return display_response


def run_unprotected_turn(
    client: openai.OpenAI,
    user_msg: str,
    history: list[dict],
    turn: int,
    logger: ConversationLog,
) -> str:
    """Run a single turn with no protection (control group)."""
    history.append({"role": "user", "content": user_msg})
    bot_response = chat_completion(client, history)
    history.append({"role": "assistant", "content": bot_response})

    logger.log_turn(TurnLog(
        turn=turn,
        timestamp=datetime.now().isoformat(),
        user_message=user_msg,
        bot_response=bot_response,
        mode="unprotected",
        sycophancy_score=0.0,
        agreement_score=0.0,
        praise_score=0.0,
        hedging_ratio=0.0,
        health_score=1.0,
        health_level="green",
        p_false_belief=0.0,
        pi_estimate=0.0,
        intervention_level="none",
        intervention_text=None,
        stance_test_result=None,
    ))

    return bot_response


def main():
    console.print(BANNER)

    # Setup
    api_key = get_api_key()
    client = openai.OpenAI(api_key=api_key)

    # Verify connection
    with console.status("[cyan]Verifying OpenAI connection...[/cyan]"):
        try:
            client.models.list()
            console.print("[green]Connected to OpenAI API[/green]")
        except openai.AuthenticationError:
            console.print("[red]Invalid API key. Please check your key and try again.[/red]")
            sys.exit(1)
        except openai.OpenAIError as e:
            console.print(f"[red]Connection error: {e}[/red]")
            sys.exit(1)

    mode = select_mode()
    console.print(f"\n[bold]Mode:[/bold] [cyan]{mode}[/cyan]")
    console.print("[dim]Type 'quit' to exit, 'stats' for session summary[/dim]\n")

    # Initialize components
    logger = ConversationLog(log_dir="logs")

    # Protected mode components
    lqcd = LQCDProbe(client)
    agreement_cls = AgreementClassifier()
    stance = StanceTest(client, interval=5)
    belief = BeliefTracker()
    drift_mon = DriftMonitor()
    breaker = CircuitBreaker()
    counter_gen = CounterGenerator(client)

    history_protected: list[dict] = []
    history_unprotected: list[dict] = []
    turn = 0

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() == "quit":
            break
        if user_input.strip().lower() == "stats":
            summary = logger.summary()
            console.print(Panel(
                "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in summary.items()),
                title="Session Summary", border_style="cyan",
            ))
            continue

        turn += 1

        if mode == "protected":
            response = run_protected_turn(
                client, user_input, history_protected, turn,
                lqcd, agreement_cls, stance, belief, drift_mon,
                breaker, counter_gen, logger,
            )
            console.print(Panel(
                Markdown(response), title="[bold blue]ChatGPT (Protected)[/bold blue]",
                border_style="blue",
            ))

        elif mode == "unprotected":
            response = run_unprotected_turn(
                client, user_input, history_unprotected, turn, logger,
            )
            console.print(Panel(
                Markdown(response), title="[bold yellow]ChatGPT (Unprotected)[/bold yellow]",
                border_style="yellow",
            ))

        elif mode == "sidebyside":
            # Run both modes
            console.print("[dim]Running both modes...[/dim]")

            response_u = run_unprotected_turn(
                client, user_input, history_unprotected, turn, logger,
            )
            response_p = run_protected_turn(
                client, user_input, history_protected, turn,
                lqcd, agreement_cls, stance, belief, drift_mon,
                breaker, counter_gen, logger,
            )

            # Side-by-side columns
            panel_u = Panel(
                Markdown(response_u),
                title="[bold yellow]Unprotected[/bold yellow]",
                border_style="yellow",
                expand=True,
            )
            panel_p = Panel(
                Markdown(response_p),
                title="[bold blue]Protected[/bold blue]",
                border_style="blue",
                expand=True,
            )
            console.print(Columns([panel_u, panel_p], equal=True, expand=True))

    # Save logs
    log_path = logger.save()
    summary = logger.summary()

    console.print("\n[bold]Session complete.[/bold]")
    console.print(f"[dim]Log saved to: {log_path}[/dim]")

    if summary.get("total_turns", 0) > 0:
        console.print(Panel(
            "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in summary.items()),
            title="Final Summary", border_style="cyan",
        ))


if __name__ == "__main__":
    main()
