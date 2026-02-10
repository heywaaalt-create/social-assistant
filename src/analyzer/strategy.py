"""Strategy analyzer: scores content and decides actions."""

import math
from datetime import datetime, timezone
from typing import Any

from src.generator.llm_provider import LLMProvider
from src.models.schemas import ActionSuggestion, ActionType, ContentItem


class StrategyAnalyzer:
    def __init__(
        self,
        settings: dict[str, Any],
        persona: dict[str, Any],
        llm: LLMProvider,
    ):
        self._settings = settings
        self._persona = persona
        self._llm = llm

    def compute_heat_score(self, item: ContentItem) -> float:
        """Score 0-10 based on engagement and recency."""
        engagement = item.likes + item.replies * 3
        # Log scale for engagement (0-7 points)
        eng_score = min(7.0, math.log1p(engagement) * 1.2)

        # Time decay (0-3 points, full score if < 6h, decays over 7 days)
        if item.published_at:
            age_hours = (
                datetime.now(timezone.utc) - item.published_at
            ).total_seconds() / 3600
        else:
            age_hours = 48  # default assume 2 days old
        time_score = max(0.0, 3.0 * math.exp(-age_hours / 48))

        return min(10.0, eng_score + time_score)

    def check_risk(self, item: ContentItem) -> list[str]:
        """Return list of risk flag descriptions."""
        flags: list[str] = []
        risk_keywords = self._settings.get("risk_keywords", [])
        text = f"{item.title} {item.summary}".lower()
        for kw in risk_keywords:
            if kw in text:
                flags.append(f"包含風險關鍵字：{kw}")
        return flags

    async def _score_relevance(self, item: ContentItem) -> float:
        """Use LLM to score relevance 0-10."""
        prompt = (
            f"請評估以下內容與「藝術展覽平台」的相關程度，只回覆一個 0 到 10 的數字。\n"
            f"標題：{item.title}\n摘要：{item.summary}"
        )
        raw = await self._llm.generate(prompt, {}, self._persona)
        try:
            return float(raw.strip().split()[0])
        except (ValueError, IndexError):
            return 5.0

    async def analyze(self, items: list[ContentItem]) -> list[ActionSuggestion]:
        min_rel = self._settings.get("min_relevance_score", 5.0)
        min_heat = self._settings.get("min_heat_score", 3.0)
        suggestions: list[ActionSuggestion] = []

        for item in items:
            risk_flags = self.check_risk(item)
            if risk_flags:
                continue

            heat = self.compute_heat_score(item)
            if heat < min_heat:
                continue

            relevance = await self._score_relevance(item)
            if relevance < min_rel:
                continue

            # Decide action type
            if heat >= 7.0 and relevance >= 7.0:
                action = ActionType.POST
            elif heat >= 4.0 or relevance >= 6.0:
                action = ActionType.REPLY
            else:
                continue

            suggestions.append(
                ActionSuggestion(
                    content_item_id=item.id,
                    action_type=action,
                    relevance_score=relevance,
                    heat_score=heat,
                    suggested_angle=f"針對「{item.title[:30]}」分享觀點",
                    priority=int(heat + relevance),
                    risk_flags=risk_flags,
                )
            )

        suggestions.sort(key=lambda s: s.priority, reverse=True)
        return suggestions
