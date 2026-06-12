import asyncio
from rich.console import Console
from rich.rule import Rule
from agents import make_agent

PROMPT = "What is 2+2? Answer in one sentence."
MODELS = ["gpt-4.1-nano", "gpt-4.1", "claude-sonnet-4-6", "gemini-2.5-flash"]

async def main():
    console = Console()
    console.print(Rule("[bold]LLM Comparison Run[/bold]"))
    for model_id in MODELS:
        try:
            r = await make_agent(model_id).run(PROMPT)
            console.print(f"[bold]{model_id}[/bold]  {r.output}\n")
        except Exception as e:
            console.print(f"[red]{model_id} failed: {e}[/red]\n")

asyncio.run(main())
