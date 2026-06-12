import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models import Model

load_dotenv(override=True)

def make_model(model_id: str) -> Model:
    """Return a PydanticAI model instance for any supported provider."""
    if model_id.startswith("claude"):
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        return AnthropicModel(model_id, provider=AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY")))

    if model_id.startswith(("gpt-", "o1", "o3", "o4")):
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIModel(model_id, provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))

    if model_id.startswith("gemini"):
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider
        return GoogleModel(model_id, provider=GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY")))

    raise ValueError(f"Unsupported model id: {model_id!r}")


def make_agent(model_id: str, system_prompt: str = "", **kwargs) -> Agent:
    """Create a role-based Agent backed by any supported model."""
    return Agent(make_model(model_id), system_prompt=system_prompt, **kwargs)