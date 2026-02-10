"""Content generator: creates posts and replies using LLM + persona."""

from typing import Any

from src.generator.llm_provider import LLMProvider
from src.models.schemas import (
    ActionSuggestion,
    ActionType,
    ContentItem,
    DraftContent,
    DraftType,
)

_POST_PROMPT = (
    "根據以下話題，以品牌人設撰寫一篇 Threads 貼文。\n"
    "要求：\n"
    "- 繁體中文\n"
    "- 口語化、像在跟朋友聊天\n"
    "- 不超過 {max_length} 字\n"
    "- 不要使用 hashtag\n"
    "- 只輸出貼文內容，不要加任何前綴說明\n\n"
    "話題：{title}\n"
    "摘要：{summary}\n"
    "切入角度：{angle}"
)

_REPLY_PROMPT = (
    "根據以下討論串，以品牌人設撰寫一則自然的回覆。\n"
    "要求：\n"
    "- 繁體中文\n"
    "- 像朋友之間的對話\n"
    "- 不超過 {max_length} 字\n"
    "- 視情況提供有價值的資訊或自然互動\n"
    "- 只輸出回覆內容\n\n"
    "原文：{summary}\n"
    "切入角度：{angle}"
)


class ContentGenerator:
    def __init__(
        self,
        settings: dict[str, Any],
        persona: dict[str, Any],
        llm: LLMProvider,
    ):
        self._settings = settings
        self._persona = persona
        self._llm = llm
        self._forbidden = set(persona.get("forbidden_words", []))
        self._max_retries = settings.get("max_retries", 3)

    def _contains_forbidden(self, text: str) -> bool:
        return any(w in text for w in self._forbidden)

    async def generate(
        self,
        item: ContentItem,
        suggestion: ActionSuggestion,
        target_thread_id: str | None = None,
    ) -> DraftContent:
        is_post = suggestion.action_type == ActionType.POST
        max_length = self._settings.get(
            "max_post_length" if is_post else "max_reply_length", 500
        )
        template = _POST_PROMPT if is_post else _REPLY_PROMPT
        prompt = template.format(
            max_length=max_length,
            title=item.title,
            summary=item.summary,
            angle=suggestion.suggested_angle,
        )
        context = {
            "source": item.source.value,
            "url": item.url,
            "tags": item.tags,
        }

        body = ""
        for _ in range(self._max_retries):
            body = await self._llm.generate(prompt, context, self._persona)
            body = body.strip()
            if len(body) <= max_length and not self._contains_forbidden(body):
                break

        return DraftContent(
            content_item_id=item.id,
            draft_type=DraftType.POST if is_post else DraftType.REPLY,
            body=body[:max_length],
            target_thread_id=target_thread_id,
        )
