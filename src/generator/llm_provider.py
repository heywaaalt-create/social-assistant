"""Swappable LLM provider abstraction for content generation."""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import anthropic
import openai


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(
        self,
        prompt: str,
        context: dict[str, Any],
        persona: dict[str, Any],
    ) -> str: ...


def _build_system_prompt(persona: dict[str, Any]) -> str:
    """Build a system prompt string from a persona dict."""
    name = persona.get("name", "")
    tone = persona.get("tone", "")
    language = persona.get("language", "")

    guidelines = persona.get("style_guidelines", [])
    guidelines_text = "\n".join(f"- {g}" for g in guidelines)

    forbidden = persona.get("forbidden_words", [])
    forbidden_text = ", ".join(forbidden)

    return (
        f"你是 {name} 的社群小編。\n"
        f"語氣：{tone}\n"
        f"語言：{language}\n"
        f"風格指南：\n{guidelines_text}\n"
        f"禁用詞彙：{forbidden_text}"
    )


def _build_user_message(prompt: str, context: dict[str, Any]) -> str:
    """Build a user message combining the prompt and JSON context."""
    context_json = json.dumps(context, ensure_ascii=False, indent=2)
    return f"{prompt}\n\n素材資料：\n{context_json}"


class ClaudeProvider:
    """Anthropic Claude LLM provider."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None = None,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        self.model = model

    async def generate(
        self,
        prompt: str,
        context: dict[str, Any],
        persona: dict[str, Any],
    ) -> str:
        system_prompt = _build_system_prompt(persona)
        user_message = _build_user_message(prompt, context)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OpenAIProvider:
    """OpenAI GPT LLM provider."""

    def __init__(
        self,
        client: openai.AsyncOpenAI | None = None,
        model: str = "gpt-4o",
    ) -> None:
        self.client = client or openai.AsyncOpenAI()
        self.model = model

    async def generate(
        self,
        prompt: str,
        context: dict[str, Any],
        persona: dict[str, Any],
    ) -> str:
        system_prompt = _build_system_prompt(persona)
        user_message = _build_user_message(prompt, context)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content


def create_provider(name: str) -> LLMProvider:
    """Factory function to create an LLM provider by name.

    Args:
        name: Provider name. One of "claude" or "openai".

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider name is unknown.
    """
    if name == "claude":
        return ClaudeProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name!r}. Supported: 'claude', 'openai'.")
