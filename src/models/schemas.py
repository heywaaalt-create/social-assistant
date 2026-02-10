"""Pydantic models for the social-assistant pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentSource(StrEnum):
    THREADS = "threads"
    NEWS = "news"
    OFFICIAL = "official"
    ARTOGO = "artogo"


class ActionType(StrEnum):
    POST = "post"
    REPLY = "reply"
    SKIP = "skip"


class DraftType(StrEnum):
    POST = "post"
    REPLY = "reply"


class DraftStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    PUBLISHED = "published"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ContentItem(BaseModel):
    """A piece of content collected from any source."""

    id: str = Field(default_factory=_uuid)
    source: ContentSource
    url: str
    title: str
    summary: str = ""
    author: str = ""
    published_at: datetime | None = None
    likes: int = 0
    replies: int = 0
    tags: list[str] = Field(default_factory=list)
    collected_at: datetime = Field(default_factory=_now)


class ActionSuggestion(BaseModel):
    """An AI-generated suggestion for how to respond to a content item."""

    id: str = Field(default_factory=_uuid)
    content_item_id: str
    action_type: ActionType
    relevance_score: int = Field(ge=0, le=10)
    heat_score: int = Field(ge=0, le=10)
    suggested_angle: str = ""
    priority: int = 0
    risk_flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class DraftContent(BaseModel):
    """A draft post or reply awaiting review."""

    id: str = Field(default_factory=_uuid)
    content_item_id: str
    draft_type: DraftType
    body: str
    target_thread_id: str | None = None
    status: DraftStatus = DraftStatus.PENDING
    rejection_reason: str = ""
    published_thread_id: str | None = None
    published_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)
