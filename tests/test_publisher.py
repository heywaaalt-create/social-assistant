"""Tests for the Threads publisher."""

from unittest.mock import AsyncMock, patch

from src.publisher.threads import ThreadsPublisher
from src.models.schemas import DraftContent, DraftType


def _make_draft(draft_type=DraftType.POST, target=None) -> DraftContent:
    return DraftContent(
        content_item_id="item-1",
        draft_type=draft_type,
        body="推薦大家去看這個展覽！",
        target_thread_id=target,
    )


async def test_publish_post():
    publisher = ThreadsPublisher(access_token="fake", user_id="12345")

    mock_create_resp = AsyncMock()
    mock_create_resp.json = lambda: {"id": "container-1"}
    mock_create_resp.raise_for_status = lambda: None

    mock_publish_resp = AsyncMock()
    mock_publish_resp.json = lambda: {"id": "thread-post-1"}
    mock_publish_resp.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=[mock_create_resp, mock_publish_resp]
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        thread_id = await publisher.publish(_make_draft())

    assert thread_id == "thread-post-1"
    assert mock_client.post.call_count == 2


async def test_publish_reply():
    publisher = ThreadsPublisher(access_token="fake", user_id="12345")

    mock_create_resp = AsyncMock()
    mock_create_resp.json = lambda: {"id": "container-2"}
    mock_create_resp.raise_for_status = lambda: None

    mock_publish_resp = AsyncMock()
    mock_publish_resp.json = lambda: {"id": "reply-1"}
    mock_publish_resp.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        side_effect=[mock_create_resp, mock_publish_resp]
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    draft = _make_draft(DraftType.REPLY, target="parent-thread-1")
    with patch("httpx.AsyncClient", return_value=mock_client):
        thread_id = await publisher.publish(draft)

    assert thread_id == "reply-1"
    # Verify reply_to_id was included
    create_call = mock_client.post.call_args_list[0]
    assert "reply_to_id" in str(create_call)
