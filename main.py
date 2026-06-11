import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from rich.console import Console
from rich.rule import Rule

load_dotenv(override=True)

console = Console()

def info(msg): console.print(f"[bold blue]ℹ[/bold blue]  {msg}")
def ok(msg):   console.print(f"[bold green]✔[/bold green]  {msg}")
def warn(msg): console.print(f"[bold yellow]⚠[/bold yellow]  {msg}")
def err(msg):  console.print(f"[bold red]✘[/bold red]  {msg}")

def run_model(label, call_fn):
    info(f"Calling {label} …")
    t0 = time.perf_counter()
    try:
        result = call_fn()
        elapsed = time.perf_counter() - t0
        ok(f"{label} responded in {elapsed:.2f}s")
        console.print(f"[dim]{result}[/dim]\n")
    except Exception as e:
        err(f"{label} failed: {e}\n")


PROMPT = "What is 2+2? Answer in one sentence."
messages = [{"role": "user", "content": PROMPT}]

console.print(Rule("[bold]LLM Comparison Run[/bold]"))
info(f"Prompt: {PROMPT}\n")

# --- OpenAI ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

run_model(
    "OpenAI gpt-4.1-nano",
    lambda: openai_client.chat.completions.create(
        model="gpt-4.1-nano", messages=messages
    ).choices[0].message.content,
)

run_model(
    "OpenAI gpt-4.1",
    lambda: openai_client.chat.completions.create(
        model="gpt-4.1", messages=messages
    ).choices[0].message.content,
)

# --- Anthropic / Claude ---
claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

run_model(
    "Anthropic claude-sonnet-4-6",
    lambda: claude_client.messages.create(
        model="claude-sonnet-4-6", messages=messages, max_tokens=256
    ).content[0].text,
)

# --- Google Gemini ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    gemini_client = OpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    run_model(
        "Google gemini-2.5-flash",
        lambda: gemini_client.chat.completions.create(
            model="gemini-2.5-flash", messages=messages
        ).choices[0].message.content,
    )
else:
    warn("Google Gemini skipped — GOOGLE_API_KEY not set")

# --- DeepSeek ---
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if deepseek_api_key:
    deepseek_client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1",
    )
    run_model(
        "DeepSeek deepseek-chat",
        lambda: deepseek_client.chat.completions.create(
            model="deepseek-chat", messages=messages
        ).choices[0].message.content,
    )
else:
    warn("DeepSeek skipped — DEEPSEEK_API_KEY not set")

console.print(Rule("[bold green]Done[/bold green]"))
