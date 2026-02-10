"""Tests for the news collector."""

from unittest.mock import AsyncMock, patch

from src.collector.news import NewsCollector
from src.models.schemas import ContentSource

SAMPLE_HTML = """
<html><body>
<article class="post-item">
  <h2><a href="/exhibit/123">台北當代藝術館新展開幕</a></h2>
  <p class="excerpt">探討數位藝術與身體的關係，展期至三月底</p>
  <span class="author">王小明</span>
  <time datetime="2026-02-08">2026-02-08</time>
</article>
<article class="post-item">
  <h2><a href="/exhibit/124">故宮南院特展登場</a></h2>
  <p class="excerpt">亞洲藝術的當代對話</p>
  <span class="author">李大華</span>
  <time datetime="2026-02-07">2026-02-07</time>
</article>
</body></html>
"""


def _make_collector() -> NewsCollector:
    return NewsCollector(
        source_name="TestNews",
        base_url="https://example.com",
        article_selector="article.post-item",
        title_selector="h2 a",
        excerpt_selector="p.excerpt",
        author_selector="span.author",
        date_selector="time",
    )


async def test_news_collector_parse():
    collector = _make_collector()
    items = collector.parse_html(SAMPLE_HTML, "https://example.com")
    assert len(items) == 2
    assert items[0].title == "台北當代藝術館新展開幕"
    assert items[0].source == ContentSource.NEWS
    assert "exhibit/123" in items[0].url
    assert items[0].author == "王小明"
    assert items[1].title == "故宮南院特展登場"


async def test_news_collector_collect():
    collector = _make_collector()

    mock_response = AsyncMock()
    mock_response.text = SAMPLE_HTML
    mock_response.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        items = await collector.collect()

    assert len(items) == 2
    assert items[0].source == ContentSource.NEWS
