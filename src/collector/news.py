"""News website collector using httpx + BeautifulSoup."""

from datetime import datetime, timezone
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from src.collector.base import BaseCollector
from src.models.schemas import ContentItem, ContentSource


class NewsCollector(BaseCollector):
    def __init__(
        self,
        source_name: str,
        base_url: str,
        article_selector: str,
        title_selector: str,
        excerpt_selector: str = "",
        author_selector: str = "",
        date_selector: str = "",
    ):
        self.source_name = source_name
        self.base_url = base_url
        self.article_selector = article_selector
        self.title_selector = title_selector
        self.excerpt_selector = excerpt_selector
        self.author_selector = author_selector
        self.date_selector = date_selector

    def parse_html(self, html: str, base_url: str) -> list[ContentItem]:
        soup = BeautifulSoup(html, "html.parser")
        articles = soup.select(self.article_selector)
        items: list[ContentItem] = []
        for article in articles:
            title_el = article.select_one(self.title_selector)
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            href = title_el.get("href", "")
            url = urljoin(base_url, href) if href else base_url

            excerpt = ""
            if self.excerpt_selector:
                el = article.select_one(self.excerpt_selector)
                if el:
                    excerpt = el.get_text(strip=True)

            author = ""
            if self.author_selector:
                el = article.select_one(self.author_selector)
                if el:
                    author = el.get_text(strip=True)

            published_at = None
            if self.date_selector:
                el = article.select_one(self.date_selector)
                if el:
                    dt_str = el.get("datetime") or el.get_text(strip=True)
                    try:
                        published_at = datetime.fromisoformat(dt_str).replace(
                            tzinfo=timezone.utc
                        )
                    except (ValueError, TypeError):
                        pass

            items.append(
                ContentItem(
                    source=ContentSource.NEWS,
                    url=url,
                    title=title,
                    summary=excerpt,
                    author=author,
                    published_at=published_at,
                )
            )
        return items

    async def collect(self) -> list[ContentItem]:
        async with httpx.AsyncClient(
            headers={"User-Agent": "ARTOGO-Social-Assistant/0.1"},
            follow_redirects=True,
            timeout=30,
        ) as client:
            response = await client.get(self.base_url)
            response.raise_for_status()
            return self.parse_html(response.text, self.base_url)
