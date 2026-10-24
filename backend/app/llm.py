"""Groq chat model factory (free tier). Built lazily so the app can start and /health works without a key."""
import os

from . import config


def make_llm(model: str | None = None, temperature: float = 0.0, max_tokens: int = 1500):
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_groq import ChatGroq

    kwargs = {}
    effort = os.getenv("GROQ_REASONING_EFFORT", "low")  # gpt-oss models 'think' first (those tokens count toward max_tokens); 'low' saves free-tier tokens
    if effort:
        kwargs["reasoning_effort"] = effort  # langchain-groq >=0.3 has a dedicated field; it rejects this inside model_kwargs
    return ChatGroq(model=model or config.GROQ_MODEL, temperature=temperature, max_tokens=max_tokens,
                    api_key=config.GROQ_API_KEY, max_retries=4, **kwargs)
