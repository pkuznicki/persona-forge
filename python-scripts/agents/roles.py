"""
Pre-built agent roles for persona-forge psychometric sessions.
Each role is a factory function so the caller controls the model.
"""

from pydantic_ai import Agent
from .factory import make_agent


def interviewer(model_id: str) -> Agent:
    return make_agent(
        model_id,
        system_prompt=(
            "You are a neutral psychometric interviewer. "
            "Present one OEJTS question at a time, wait for the response, "
            "then move to the next. Never interpret answers yourself."
        ),
    )


def analyst(model_id: str) -> Agent:
    return make_agent(
        model_id,
        system_prompt=(
            "You are a psychometric analyst. "
            "Given a list of OEJTS question-answer pairs, score each of the four axes "
            "(E/I, S/N, T/F, J/P) and return a JSON object with axis scores and "
            "a confidence value (0-1) per axis."
        ),
    )
