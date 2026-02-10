"""Threads API collector."""

from datetime import datetime

import httpx

from src.collector.base import BaseCollector
from src.models.schemas import ContentItem, ContentSource

THREADS_API_BASE = "https://graph.threads.net/v1.0"


class ThreadsCollector(BaseCollector):
    def __init__(self, access_token: str, user_id: str):
        self._access_token = access_token
        self._user_id = user_id

    def parse_response(self, data: dict) -> list[ContentItem]:
        items: list[ContentItem] = []
        for post in data.get("data", []):
            text = post.get("text", "")
            title = text[:50] + ("..." if len(text) > 50 else "")

            published_at = None
            if ts := post.get("timestamp"):
                try:
                    published_at = datetime.fromisoformat(
                        ts.replace("+0000", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            tags = [w.lstrip("#") for w in text.split() if w.startswith("#")]

            items.append(
                ContentItem(
                    source=ContentSource.THREADS,
                    url=post.get("permalink", ""),
                    title=title,
                    summary=text,
                    author=post.get("username", ""),
                    published_at=published_at,
                    likes=post.get("like_count", 0),
                    replies=post.get("reply_count", 0),
                    tags=tags,
                )
            )
        return items

    async def collect(self, keywords: list[str] | None = None) -> list[ContentItem]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{THREADS_API_BASE}/{self._user_id}/threads",
                params={
                    "fields": "id,text,username,timestamp,like_count,reply_count,permalink",
                    "access_token": self._access_token,
                    "limit": 25,
                },
            )
            response.raise_for_status()
            return self.parse_response(response.json())
