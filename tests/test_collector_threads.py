"""Tests for the Threads API collector."""

from unittest.mock import AsyncMock, patch

from src.collector.threads import ThreadsCollector
from src.models.schemas import ContentSource

SAMPLE_THREADS_RESPONSE = {
    "data": [
        {
            "id": "123456",
            "text": "剛看完北美館的新展，推薦大家去看！#展覽 #北美館",
            "username": "art_lover_tw",
            "timestamp": "2026-02-09T10:30:00+0000",
            "like_count": 42,
            "reply_count": 5,
            "permalink": "https://www.threads.net/@art_lover_tw/post/123456",
        },
        {
            "id": "789012",
            "text": "有人去過故宮的亞洲藝術展嗎？",
            "username": "museum_fan",
            "timestamp": "2026-02-08T15:00:00+0000",
            "like_count": 18,
            "reply_count": 12,
            "permalink": "https://www.threads.net/@museum_fan/post/789012",
        },
    ]
}


async def test_threads_collector_parse_response():
    collector = ThreadsCollector(access_token="fake", user_id="fake")
    items = collector.parse_response(SAMPLE_THREADS_RESPONSE)
    assert len(items) == 2
    assert items[0].source == ContentSource.THREADS
    assert items[0].likes == 42
    assert items[0].replies == 5
    assert "北美館" in items[0].title


async def test_threads_collector_collect():
    collector = ThreadsCollector(access_token="fake", user_id="fake")

    mock_response = AsyncMock()
    mock_response.json = lambda: SAMPLE_THREADS_RESPONSE
    mock_response.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        items = await collector.collect()

    assert len(items) == 2
