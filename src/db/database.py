"""Async SQLite database layer for the social-assistant pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from src.models.schemas import (
    ContentItem,
    ContentSource,
    DraftContent,
    DraftStatus,
    DraftType,
)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS content_items (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    url             TEXT NOT NULL,
    title           TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    published_at    TEXT,
    likes           INTEGER NOT NULL DEFAULT 0,
    replies         INTEGER NOT NULL DEFAULT 0,
    tags            TEXT NOT NULL DEFAULT '[]',
    collected_at    TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_content_items_url ON content_items(url);

CREATE TABLE IF NOT EXISTS drafts (
    id                  TEXT PRIMARY KEY,
    content_item_id     TEXT NOT NULL,
    draft_type          TEXT NOT NULL,
    body                TEXT NOT NULL,
    target_thread_id    TEXT,
    status              TEXT NOT NULL DEFAULT 'pending',
    rejection_reason    TEXT NOT NULL DEFAULT '',
    published_thread_id TEXT,
    published_at        TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (content_item_id) REFERENCES content_items(id)
);

CREATE INDEX IF NOT EXISTS idx_drafts_status ON drafts(status);
"""


def _dt_to_str(dt: datetime | None) -> str | None:
    """Convert a datetime to ISO-8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    """Convert an ISO-8601 string back to a datetime, or None."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


class Database:
    """Async SQLite database for content items and drafts."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Open the connection and create tables if needed."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Content items
    # ------------------------------------------------------------------

    async def save_content_item(self, item: ContentItem) -> None:
        """Insert a content item into the database."""
        assert self._conn is not None
        await self._conn.execute(
            """
            INSERT INTO content_items
                (id, source, url, title, summary, author, published_at,
                 likes, replies, tags, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.source.value,
                item.url,
                item.title,
                item.summary,
                item.author,
                _dt_to_str(item.published_at),
                item.likes,
                item.replies,
                json.dumps(item.tags, ensure_ascii=False),
                _dt_to_str(item.collected_at),
            ),
        )
        await self._conn.commit()

    async def get_content_item(self, item_id: str) -> ContentItem | None:
        """Retrieve a content item by its id, or None if not found."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT * FROM content_items WHERE id = ?", (item_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_content_item(row)

    async def url_exists(self, url: str) -> bool:
        """Check whether a URL already exists in the database."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT 1 FROM content_items WHERE url = ?", (url,)
        ) as cursor:
            row = await cursor.fetchone()
        return row is not None

    async def get_recent_content_items(self, limit: int = 50) -> list[ContentItem]:
        """Return the most recently collected content items."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT * FROM content_items ORDER BY collected_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_content_item(row) for row in rows]

    # ------------------------------------------------------------------
    # Drafts
    # ------------------------------------------------------------------

    async def save_draft(self, draft: DraftContent) -> None:
        """Insert a draft into the database."""
        assert self._conn is not None
        await self._conn.execute(
            """
            INSERT INTO drafts
                (id, content_item_id, draft_type, body, target_thread_id,
                 status, rejection_reason, published_thread_id, published_at,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft.id,
                draft.content_item_id,
                draft.draft_type.value,
                draft.body,
                draft.target_thread_id,
                draft.status.value,
                draft.rejection_reason,
                draft.published_thread_id,
                _dt_to_str(draft.published_at),
                _dt_to_str(draft.created_at),
            ),
        )
        await self._conn.commit()

    async def get_draft(self, draft_id: str) -> DraftContent | None:
        """Retrieve a draft by its id, or None if not found."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (draft_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_draft(row)

    async def get_drafts_by_status(self, status: DraftStatus) -> list[DraftContent]:
        """Return all drafts with the given status."""
        assert self._conn is not None
        async with self._conn.execute(
            "SELECT * FROM drafts WHERE status = ? ORDER BY created_at DESC",
            (status.value,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_draft(row) for row in rows]

    async def update_draft_status(
        self,
        draft_id: str,
        status: DraftStatus,
        **kwargs: str | None,
    ) -> None:
        """Update a draft's status and optionally other text fields."""
        assert self._conn is not None
        # Build SET clause dynamically for extra keyword args
        set_parts = ["status = ?"]
        params: list[str | None] = [status.value]
        for key, value in kwargs.items():
            set_parts.append(f"{key} = ?")
            params.append(value)
        params.append(draft_id)

        sql = f"UPDATE drafts SET {', '.join(set_parts)} WHERE id = ?"
        await self._conn.execute(sql, params)
        await self._conn.commit()

    # ------------------------------------------------------------------
    # Row mapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_content_item(row: aiosqlite.Row) -> ContentItem:
        """Convert a database row to a ContentItem model."""
        return ContentItem(
            id=row["id"],
            source=ContentSource(row["source"]),
            url=row["url"],
            title=row["title"],
            summary=row["summary"],
            author=row["author"],
            published_at=_str_to_dt(row["published_at"]),
            likes=row["likes"],
            replies=row["replies"],
            tags=json.loads(row["tags"]),
            collected_at=_str_to_dt(row["collected_at"]),
        )

    @staticmethod
    def _row_to_draft(row: aiosqlite.Row) -> DraftContent:
        """Convert a database row to a DraftContent model."""
        return DraftContent(
            id=row["id"],
            content_item_id=row["content_item_id"],
            draft_type=DraftType(row["draft_type"]),
            body=row["body"],
            target_thread_id=row["target_thread_id"],
            status=DraftStatus(row["status"]),
            rejection_reason=row["rejection_reason"],
            published_thread_id=row["published_thread_id"],
            published_at=_str_to_dt(row["published_at"]),
            created_at=_str_to_dt(row["created_at"]),
        )
