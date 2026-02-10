"""Tests for async SQLite database layer."""

import pytest

from src.db.database import Database
from src.models.schemas import (
    ContentItem,
    ContentSource,
    DraftContent,
    DraftStatus,
    DraftType,
)


@pytest.fixture
async def db(tmp_db_path):
    """Create and initialize a test database."""
    database = Database(tmp_db_path)
    await database.init()
    yield database
    await database.close()


class TestDatabaseInit:
    async def test_init_creates_tables(self, db):
        """Verify that init() creates the content_items and drafts tables."""
        async with db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cursor:
            rows = await cursor.fetchall()
        table_names = [row[0] for row in rows]
        assert "content_items" in table_names
        assert "drafts" in table_names


class TestContentItemOperations:
    async def test_save_and_get_content_item(self, db):
        """Save a content item and retrieve it by id."""
        item = ContentItem(
            source=ContentSource.THREADS,
            url="https://threads.net/t/123",
            title="Test Thread",
            summary="A test thread summary",
            tags=["art", "gallery"],
        )
        await db.save_content_item(item)

        retrieved = await db.get_content_item(item.id)
        assert retrieved is not None
        assert retrieved.id == item.id
        assert retrieved.source == ContentSource.THREADS
        assert retrieved.url == "https://threads.net/t/123"
        assert retrieved.title == "Test Thread"
        assert retrieved.summary == "A test thread summary"
        assert retrieved.tags == ["art", "gallery"]

    async def test_get_content_item_not_found(self, db):
        """Getting a non-existent item returns None."""
        result = await db.get_content_item("non-existent-id")
        assert result is None

    async def test_url_exists(self, db):
        """Verify url_exists returns correct boolean."""
        item = ContentItem(
            source=ContentSource.NEWS,
            url="https://example.com/article",
            title="Article",
        )
        assert await db.url_exists("https://example.com/article") is False
        await db.save_content_item(item)
        assert await db.url_exists("https://example.com/article") is True
        assert await db.url_exists("https://example.com/other") is False

    async def test_get_recent_content_items(self, db):
        """Verify recent items are returned in order, respecting limit."""
        for i in range(5):
            item = ContentItem(
                source=ContentSource.THREADS,
                url=f"https://threads.net/t/{i}",
                title=f"Thread {i}",
            )
            await db.save_content_item(item)

        recent = await db.get_recent_content_items(limit=3)
        assert len(recent) == 3

        all_items = await db.get_recent_content_items(limit=50)
        assert len(all_items) == 5


class TestDraftOperations:
    async def test_save_and_get_draft(self, db):
        """Save a draft and retrieve it by id."""
        item = ContentItem(
            source=ContentSource.THREADS,
            url="https://threads.net/t/456",
            title="Source Thread",
        )
        await db.save_content_item(item)

        draft = DraftContent(
            content_item_id=item.id,
            draft_type=DraftType.POST,
            body="This is a draft post",
        )
        await db.save_draft(draft)

        retrieved = await db.get_draft(draft.id)
        assert retrieved is not None
        assert retrieved.id == draft.id
        assert retrieved.content_item_id == item.id
        assert retrieved.draft_type == DraftType.POST
        assert retrieved.body == "This is a draft post"
        assert retrieved.status == DraftStatus.PENDING

    async def test_get_draft_not_found(self, db):
        """Getting a non-existent draft returns None."""
        result = await db.get_draft("non-existent-id")
        assert result is None

    async def test_get_pending_drafts(self, db):
        """Save 3 pending drafts and get all of them."""
        item = ContentItem(
            source=ContentSource.THREADS,
            url="https://threads.net/t/789",
            title="Source Thread",
        )
        await db.save_content_item(item)

        for i in range(3):
            draft = DraftContent(
                content_item_id=item.id,
                draft_type=DraftType.POST,
                body=f"Draft post {i}",
            )
            await db.save_draft(draft)

        pending = await db.get_drafts_by_status(DraftStatus.PENDING)
        assert len(pending) == 3
        for d in pending:
            assert d.status == DraftStatus.PENDING

    async def test_update_draft_status(self, db):
        """Update a draft's status and verify the change."""
        item = ContentItem(
            source=ContentSource.THREADS,
            url="https://threads.net/t/update",
            title="Source Thread",
        )
        await db.save_content_item(item)

        draft = DraftContent(
            content_item_id=item.id,
            draft_type=DraftType.POST,
            body="Draft to be approved",
        )
        await db.save_draft(draft)

        await db.update_draft_status(draft.id, DraftStatus.APPROVED)
        updated = await db.get_draft(draft.id)
        assert updated is not None
        assert updated.status == DraftStatus.APPROVED

    async def test_update_draft_status_with_kwargs(self, db):
        """Update draft status with additional fields like rejection_reason."""
        item = ContentItem(
            source=ContentSource.THREADS,
            url="https://threads.net/t/reject",
            title="Source Thread",
        )
        await db.save_content_item(item)

        draft = DraftContent(
            content_item_id=item.id,
            draft_type=DraftType.POST,
            body="Draft to be rejected",
        )
        await db.save_draft(draft)

        await db.update_draft_status(
            draft.id, DraftStatus.REJECTED, rejection_reason="Off-brand tone"
        )
        updated = await db.get_draft(draft.id)
        assert updated is not None
        assert updated.status == DraftStatus.REJECTED
        assert updated.rejection_reason == "Off-brand tone"

    async def test_get_drafts_by_status_filters_correctly(self, db):
        """Verify filtering by status only returns matching drafts."""
        item = ContentItem(
            source=ContentSource.THREADS,
            url="https://threads.net/t/filter",
            title="Source Thread",
        )
        await db.save_content_item(item)

        # Create drafts with different statuses
        draft1 = DraftContent(
            content_item_id=item.id,
            draft_type=DraftType.POST,
            body="Pending draft",
        )
        draft2 = DraftContent(
            content_item_id=item.id,
            draft_type=DraftType.POST,
            body="Another draft",
        )
        await db.save_draft(draft1)
        await db.save_draft(draft2)

        # Approve one
        await db.update_draft_status(draft1.id, DraftStatus.APPROVED)

        pending = await db.get_drafts_by_status(DraftStatus.PENDING)
        approved = await db.get_drafts_by_status(DraftStatus.APPROVED)
        assert len(pending) == 1
        assert len(approved) == 1
        assert pending[0].id == draft2.id
        assert approved[0].id == draft1.id
