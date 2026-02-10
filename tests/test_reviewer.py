"""Tests for the Slack reviewer."""

from unittest.mock import MagicMock

from src.reviewer.slack_bot import SlackReviewer, build_review_blocks
from src.models.schemas import DraftContent, DraftType, ContentItem, ContentSource


def _make_draft() -> DraftContent:
    return DraftContent(
        content_item_id="item-1",
        draft_type=DraftType.POST,
        body="最近去看了台北當代藝術館的新展，沉浸式的體驗真的很棒！",
    )


def _make_item() -> ContentItem:
    return ContentItem(
        source=ContentSource.NEWS,
        url="https://example.com/exhibit",
        title="台北當代藝術館新展",
        summary="沉浸式體驗展覽",
    )


def test_build_review_blocks():
    draft = _make_draft()
    item = _make_item()
    blocks = build_review_blocks(draft, item)
    assert len(blocks) >= 3
    actions = [b for b in blocks if b.get("type") == "actions"]
    assert len(actions) == 1
    buttons = actions[0]["elements"]
    assert len(buttons) == 3  # approve, edit, reject


async def test_send_for_review():
    mock_client = MagicMock()
    mock_client.chat_postMessage = MagicMock(
        return_value={"ok": True, "ts": "1234567890.123"}
    )
    reviewer = SlackReviewer(client=mock_client, channel="#social-review")
    draft = _make_draft()
    item = _make_item()
    ts = await reviewer.send_for_review(draft, item)
    assert ts == "1234567890.123"
    mock_client.chat_postMessage.assert_called_once()
