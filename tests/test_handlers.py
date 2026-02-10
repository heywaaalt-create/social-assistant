"""Tests for Slack action handlers."""

from unittest.mock import AsyncMock

from src.reviewer.handlers import handle_approve, handle_reject, handle_edit_submit
from src.models.schemas import DraftContent, DraftType, DraftStatus


def _make_draft() -> DraftContent:
    return DraftContent(
        id="draft-1",
        content_item_id="item-1",
        draft_type=DraftType.POST,
        body="推薦展覽",
    )


async def test_handle_approve():
    db = AsyncMock()
    draft = _make_draft()
    db.get_draft = AsyncMock(return_value=draft)
    db.update_draft_status = AsyncMock()

    publisher = AsyncMock()
    publisher.publish = AsyncMock(return_value="thread-123")

    await handle_approve("draft-1", db, publisher)

    db.update_draft_status.assert_any_call("draft-1", DraftStatus.APPROVED)
    publisher.publish.assert_called_once_with(draft)
    db.update_draft_status.assert_any_call(
        "draft-1", DraftStatus.PUBLISHED, published_thread_id="thread-123"
    )


async def test_handle_reject():
    db = AsyncMock()
    db.update_draft_status = AsyncMock()

    await handle_reject("draft-1", db, reason="語氣不對")

    db.update_draft_status.assert_called_once_with(
        "draft-1", DraftStatus.REJECTED, rejection_reason="語氣不對"
    )


async def test_handle_edit_submit():
    db = AsyncMock()
    draft = _make_draft()
    db.get_draft = AsyncMock(return_value=draft)
    db.update_draft_status = AsyncMock()

    publisher = AsyncMock()
    publisher.publish = AsyncMock(return_value="thread-456")

    await handle_edit_submit("draft-1", "修改後的內容", db, publisher)

    publisher.publish.assert_called_once()
    published_draft = publisher.publish.call_args[0][0]
    assert published_draft.body == "修改後的內容"
