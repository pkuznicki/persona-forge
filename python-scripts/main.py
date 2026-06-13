import asyncio
from collections import Counter
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from rich.table import Table
from experiment import ExperimentConfig, ExperimentResult, Turn, generate_facts, run_experiment

# ── Configure your experiment here ──────────────────────────────────────────
SUBJECT_MODEL   = "gpt-4o-mini"
PERSUADER_MODEL = "gpt-4.1"
GENERATOR_MODEL = "gpt-4.1"
N_FACTS         = 10
MAX_ROUNDS      = 100   # per fact; strategies cycle automatically
# ────────────────────────────────────────────────────────────────────────────

console = Console()


def on_open(question: str, answer: str, false_answer: str, justification: str) -> None:
    color = "red" if answer.lower() == false_answer.lower() else "green"
    console.print(Panel(
        f"[dim]Q:[/dim] {question}\n"
        f"[dim]A:[/dim] [{color}]{answer}[/{color}]  —  {justification}",
        title="[bold]Opening[/bold]",
        border_style="dim",
    ))
    console.print()


def on_persuader(round_num: int, strategy: str, message: str) -> None:
    console.print(f"[bold]Round {round_num}[/bold]  [magenta][{strategy}][/magenta]")
    console.print(f"  [magenta]Persuader:[/magenta] {message}")


def on_turn(turn: Turn) -> None:
    status = "[bold red]CAPITULATED[/bold red]" if turn.capitulated else "[dim]held[/dim]"
    color = "red" if turn.capitulated else "green"
    console.print(
        f"  [cyan]Subject:[/cyan]   [{color}]{turn.subject_answer}[/{color}]"
        f"  —  {turn.subject_justification}"
    )
    console.print(f"  → {status}")
    console.print()


async def main() -> None:
    console.print(Rule("[bold]Generating facts[/bold]"))
    console.print(f"  Generator: [yellow]{GENERATOR_MODEL}[/yellow]  |  requesting {N_FACTS} facts…")
    facts = await generate_facts(N_FACTS, GENERATOR_MODEL)
    console.print(f"  Got [bold]{len(facts)}[/bold] facts.\n")

    results: list[ExperimentResult] = []

    for i, fact in enumerate(facts, start=1):
        console.print(Rule(f"[bold]Experiment {i}/{len(facts)}[/bold]"))
        console.print(f"  [green]{fact.true_fact}[/green]  →  lie: [red]{fact.false_claim}[/red]")
        console.print(f"  Subject: [cyan]{SUBJECT_MODEL}[/cyan]  |  Persuader: [magenta]{PERSUADER_MODEL}[/magenta]")
        console.print()

        cfg = ExperimentConfig(
            subject_model=SUBJECT_MODEL,
            persuader_model=PERSUADER_MODEL,
            true_fact=fact.true_fact,
            false_claim=fact.false_claim,
            false_answer=fact.false_answer,
            opening_question=fact.opening_question,
            answer_choices=fact.answer_choices,
            max_rounds=MAX_ROUNDS,
        )

        result = await run_experiment(
            cfg,
            on_open=lambda q, a, j, fa=fact.false_answer: on_open(q, a, fa, j),
            on_persuader=on_persuader,
            on_turn=on_turn,
        )
        results.append(result)

        if result.capitulated:
            console.print(
                f"[bold red]Capitulated[/bold red] on round {result.capitulation_round} "
                f"via [magenta]{result.winning_strategy}[/magenta]"
            )
        else:
            console.print(f"[bold green]Held firm[/bold green] through all {len(result.turns)} rounds.")
        console.print()

    # ── Final summary ────────────────────────────────────────────────────────
    console.print(Rule("[bold]Final Summary[/bold]"))

    total          = len(results)
    capitulated    = [r for r in results if r.capitulated]
    held           = [r for r in results if not r.capitulated]
    cap_rounds     = [r.capitulation_round for r in capitulated]
    total_rounds   = sum(len(r.turns) for r in results)
    strategy_wins  = Counter(r.winning_strategy for r in capitulated)

    # Top-level numbers
    pct = 100 * len(capitulated) // total if total else 0
    avg_cap = sum(cap_rounds) / len(cap_rounds) if cap_rounds else 0
    avg_rounds = total_rounds / total if total else 0

    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="dim")
    overview.add_column()
    overview.add_row("Facts run",          str(total))
    overview.add_row("Total rounds",       str(total_rounds))
    overview.add_row("Avg rounds/fact",    f"{avg_rounds:.1f}")
    overview.add_row("Capitulated",        f"[red]{len(capitulated)}[/red] / {total}  ({pct}%)")
    overview.add_row("Held firm",          f"[green]{len(held)}[/green] / {total}")
    overview.add_row("Avg round of cap.",  f"{avg_cap:.1f}" if cap_rounds else "—")
    overview.add_row("Earliest cap.",      str(min(cap_rounds)) if cap_rounds else "—")
    overview.add_row("Latest cap.",        str(max(cap_rounds)) if cap_rounds else "—")
    console.print(overview)
    console.print()

    # Strategy breakdown
    if strategy_wins:
        console.print("[bold]Capitulations by strategy[/bold]")
        strat_table = Table(show_header=True, header_style="bold")
        strat_table.add_column("Strategy",      style="magenta")
        strat_table.add_column("Capitulations", justify="right")
        strat_table.add_column("% of caps",     justify="right")
        for strategy, count in strategy_wins.most_common():
            pct_s = 100 * count // len(capitulated)
            strat_table.add_row(strategy, str(count), f"{pct_s}%")
        console.print(strat_table)


asyncio.run(main())
