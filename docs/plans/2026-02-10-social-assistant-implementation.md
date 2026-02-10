# Social Assistant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an automated Threads social media assistant for ARTOGO that collects art exhibition trends, generates brand-aligned content, and publishes after human approval via Slack.

**Architecture:** Pipeline architecture — Collector gathers content from multiple sources into SQLite, Analyzer scores and filters topics, Generator creates posts/replies via swappable LLM, Reviewer sends to Slack for human approval, Publisher posts to Threads on schedule. APScheduler orchestrates the pipeline.

**Tech Stack:** Python 3.12+ / uv / SQLite + aiosqlite / httpx + BeautifulSoup4 / anthropic + openai / slack-bolt / APScheduler / pydantic / pyyaml / pytest + pytest-asyncio

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `.python-version`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `config/settings.yaml`
- Create: `config/persona.yaml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Initialize uv project**

```bash
cd /Users/walt/Desktop/Ai/ARTOGO/social-assistant
uv init --no-readme
```

**Step 2: Set Python version**

Write `.python-version`:
```
3.12
```

**Step 3: Write pyproject.toml**

```toml
[project]
name = "social-assistant"
version = "0.1.0"
description = "ARTOGO social media automation assistant for Threads"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "aiosqlite>=0.20",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "apscheduler>=3.10",
    "slack-bolt>=1.20",
    "anthropic>=0.40",
    "openai>=1.50",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-mock>=3.14",
    "ruff>=0.8",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

**Step 4: Write .env.example**

```
# Threads API
THREADS_ACCESS_TOKEN=
THREADS_USER_ID=

# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=
SLACK_REVIEW_CHANNEL=#social-review

# LLM
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Database
DATABASE_PATH=data/social_assistant.db
```

**Step 5: Write .gitignore**

```
__pycache__/
*.pyc
.env
data/
.venv/
*.egg-info/
dist/
.ruff_cache/
.pytest_cache/
```

**Step 6: Create directory structure**

```bash
mkdir -p src/{collector,analyzer,generator,reviewer,publisher,models,db}
mkdir -p config tests data
touch src/__init__.py src/collector/__init__.py src/analyzer/__init__.py
touch src/generator/__init__.py src/reviewer/__init__.py src/publisher/__init__.py
touch src/models/__init__.py src/db/__init__.py
touch tests/__init__.py
```

**Step 7: Write config/settings.yaml**

```yaml
collector:
  schedule_interval_hours: 4
  keywords:
    - 展覽
    - 美術館
    - 導覽
    - 當代藝術
    - 博物館
    - 策展
    - 藝術家
    - 畫展
    - 雕塑
    - 攝影展
  news_sources:
    - name: ARTouch
      url: https://artouch.com/
      type: rss
    - name: 非池中
      url: https://artemperor.tw/
      type: scrape

analyzer:
  min_relevance_score: 5.0
  min_heat_score: 3.0
  daily_post_limit: 3
  daily_reply_limit: 10
  risk_keywords:
    - 政治
    - 選舉
    - 爭議
    - 抄襲

generator:
  max_retries: 3
  max_post_length: 500
  max_reply_length: 300

reviewer:
  publish_hours: [10, 15, 20]

publisher:
  retry_count: 3
  retry_delay_seconds: 60
  metrics_check_interval_hours: 24
```

**Step 8: Write config/persona.yaml**

```yaml
name: ARTOGO
platform: Threads
tone: 輕鬆友善、像懂藝術的朋友
language: zh-TW

style_guidelines:
  - 用口語化的方式分享藝術觀點
  - 偶爾帶點幽默感
  - 對藝術有熱情但不說教
  - 鼓勵大家親自去看展
  - 適時提問引發互動
  - 不過度使用 emoji，偶爾用一兩個即可

forbidden_words:
  - 業配
  - 工商
  - 合作邀約
  - 限時優惠
  - 立即購買

example_posts:
  - "最近去看了《未來身體》展，那個互動裝置真的會讓人停下腳步想很久..."
  - "有人也覺得這次北美館的策展動線設計得特別好嗎？從入口就開始有沉浸感"
  - "週末想看展的可以考慮故宮的新展，聽說導覽很有料"

example_replies:
  - "這個展我也有去！二樓那個影像裝置超推"
  - "推薦可以預約導覽，會聽到很多展牌上沒寫的故事"
  - "這位藝術家之前在關渡美術館也有展過，風格蠻一致的"
```

**Step 9: Write src/config.py**

```python
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _BASE_DIR / "config"


def load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings() -> dict[str, Any]:
    return load_yaml("settings.yaml")


def load_persona() -> dict[str, Any]:
    return load_yaml("persona.yaml")


def init_env() -> None:
    load_dotenv(_BASE_DIR / ".env")
```

**Step 10: Write tests/conftest.py**

```python
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def sample_settings():
    return {
        "collector": {
            "schedule_interval_hours": 4,
            "keywords": ["展覽", "美術館"],
            "news_sources": [],
        },
        "analyzer": {
            "min_relevance_score": 5.0,
            "min_heat_score": 3.0,
            "daily_post_limit": 3,
            "daily_reply_limit": 10,
            "risk_keywords": ["政治"],
        },
        "generator": {
            "max_retries": 3,
            "max_post_length": 500,
            "max_reply_length": 300,
        },
        "reviewer": {"publish_hours": [10, 15, 20]},
        "publisher": {
            "retry_count": 3,
            "retry_delay_seconds": 60,
            "metrics_check_interval_hours": 24,
        },
    }


@pytest.fixture
def sample_persona():
    return {
        "name": "ARTOGO",
        "platform": "Threads",
        "tone": "輕鬆友善",
        "language": "zh-TW",
        "style_guidelines": ["用口語化的方式分享藝術觀點"],
        "forbidden_words": ["業配", "工商"],
        "example_posts": ["最近去看了展覽..."],
        "example_replies": ["這個展我也有去！"],
    }
```

**Step 11: Install dependencies**

```bash
cd /Users/walt/Desktop/Ai/ARTOGO/social-assistant
uv sync --all-extras
```

**Step 12: Run initial test to verify setup**

```bash
uv run pytest --co -q
```
Expected: no errors, 0 tests collected (no test files yet)

**Step 13: Commit**

```bash
git init
git add -A
git commit -m "feat: scaffold social-assistant project with config and dependencies"
```

---

## Task 2: Database & Models

**Files:**
- Create: `src/models/schemas.py`
- Create: `src/db/database.py`
- Create: `tests/test_models.py`
- Create: `tests/test_database.py`

**Step 1: Write test for Pydantic models**

`tests/test_models.py`:
```python
from datetime import datetime, timezone

from src.models.schemas import (
    ActionSuggestion,
    ActionType,
    ContentItem,
    ContentSource,
    DraftContent,
    DraftStatus,
    DraftType,
)


def test_content_item_creation():
    item = ContentItem(
        source=ContentSource.NEWS,
        url="https://artouch.com/exhibit/1",
        title="台北當代藝術館新展開幕",
        summary="探討數位藝術與身體的關係",
        author="ARTouch",
        published_at=datetime(2026, 2, 10, tzinfo=timezone.utc),
        likes=42,
        replies=5,
        tags=["展覽", "當代藝術"],
    )
    assert item.id is not None
    assert item.source == ContentSource.NEWS
    assert item.collected_at is not None


def test_action_suggestion_creation():
    suggestion = ActionSuggestion(
        content_item_id="abc-123",
        action_type=ActionType.POST,
        relevance_score=8.5,
        heat_score=7.0,
        suggested_angle="分享這個展覽的互動裝置體驗",
        priority=1,
        risk_flags=[],
    )
    assert suggestion.action_type == ActionType.POST
    assert suggestion.relevance_score == 8.5


def test_draft_content_creation():
    draft = DraftContent(
        content_item_id="abc-123",
        draft_type=DraftType.POST,
        body="最近去看了台北當代藝術館的新展...",
        target_thread_id=None,
    )
    assert draft.status == DraftStatus.PENDING
    assert len(draft.body) <= 500
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_models.py -v
```
Expected: FAIL — cannot import `src.models.schemas`

**Step 3: Implement Pydantic models**

`src/models/schemas.py`:
```python
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


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


class ContentItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: ContentSource
    url: str
    title: str
    summary: str = ""
    author: str = ""
    published_at: datetime | None = None
    likes: int = 0
    replies: int = 0
    tags: list[str] = Field(default_factory=list)
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActionSuggestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content_item_id: str
    action_type: ActionType
    relevance_score: float = Field(ge=0, le=10)
    heat_score: float = Field(ge=0, le=10)
    suggested_angle: str = ""
    priority: int = 0
    risk_flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DraftContent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content_item_id: str
    draft_type: DraftType
    body: str
    target_thread_id: str | None = None
    status: DraftStatus = DraftStatus.PENDING
    rejection_reason: str = ""
    published_thread_id: str | None = None
    published_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

**Step 4: Run model tests**

```bash
uv run pytest tests/test_models.py -v
```
Expected: 3 passed

**Step 5: Write database tests**

`tests/test_database.py`:
```python
import pytest
from datetime import datetime, timezone

from src.db.database import Database
from src.models.schemas import ContentItem, ContentSource, DraftContent, DraftStatus, DraftType


@pytest.fixture
async def db(tmp_db_path):
    database = Database(str(tmp_db_path))
    await database.init()
    yield database
    await database.close()


async def test_init_creates_tables(db):
    # Tables exist if no error on init
    assert db is not None


async def test_save_and_get_content_item(db):
    item = ContentItem(
        source=ContentSource.NEWS,
        url="https://example.com/exhibit",
        title="測試展覽",
        summary="這是測試",
        author="Test",
        published_at=datetime(2026, 2, 10, tzinfo=timezone.utc),
        tags=["展覽"],
    )
    await db.save_content_item(item)
    result = await db.get_content_item(item.id)
    assert result is not None
    assert result.title == "測試展覽"
    assert result.source == ContentSource.NEWS


async def test_save_and_get_draft(db):
    draft = DraftContent(
        content_item_id="item-1",
        draft_type=DraftType.POST,
        body="測試貼文內容",
    )
    await db.save_draft(draft)
    result = await db.get_draft(draft.id)
    assert result is not None
    assert result.body == "測試貼文內容"
    assert result.status == DraftStatus.PENDING


async def test_get_pending_drafts(db):
    for i in range(3):
        draft = DraftContent(
            content_item_id=f"item-{i}",
            draft_type=DraftType.POST,
            body=f"貼文 {i}",
        )
        await db.save_draft(draft)
    pending = await db.get_drafts_by_status(DraftStatus.PENDING)
    assert len(pending) == 3


async def test_update_draft_status(db):
    draft = DraftContent(
        content_item_id="item-1",
        draft_type=DraftType.POST,
        body="待審核貼文",
    )
    await db.save_draft(draft)
    await db.update_draft_status(draft.id, DraftStatus.APPROVED)
    result = await db.get_draft(draft.id)
    assert result.status == DraftStatus.APPROVED


async def test_url_exists(db):
    item = ContentItem(
        source=ContentSource.NEWS,
        url="https://example.com/unique",
        title="唯一展覽",
    )
    await db.save_content_item(item)
    assert await db.url_exists("https://example.com/unique") is True
    assert await db.url_exists("https://example.com/other") is False
```

**Step 6: Run database tests to verify they fail**

```bash
uv run pytest tests/test_database.py -v
```
Expected: FAIL — cannot import `src.db.database`

**Step 7: Implement Database class**

`src/db/database.py`:
```python
import json
from datetime import datetime, timezone

import aiosqlite

from src.models.schemas import ContentItem, ContentSource, DraftContent, DraftStatus, DraftType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS content_items (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT DEFAULT '',
    author TEXT DEFAULT '',
    published_at TEXT,
    likes INTEGER DEFAULT 0,
    replies INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    collected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,
    content_item_id TEXT NOT NULL,
    draft_type TEXT NOT NULL,
    body TEXT NOT NULL,
    target_thread_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    rejection_reason TEXT DEFAULT '',
    published_thread_id TEXT,
    published_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (content_item_id) REFERENCES content_items(id)
);

CREATE INDEX IF NOT EXISTS idx_drafts_status ON drafts(status);
CREATE INDEX IF NOT EXISTS idx_content_url ON content_items(url);
"""


class Database:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    # -- ContentItem --

    async def save_content_item(self, item: ContentItem) -> None:
        await self._conn.execute(
            """INSERT OR IGNORE INTO content_items
               (id, source, url, title, summary, author, published_at, likes, replies, tags, collected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id,
                item.source.value,
                item.url,
                item.title,
                item.summary,
                item.author,
                item.published_at.isoformat() if item.published_at else None,
                item.likes,
                item.replies,
                json.dumps(item.tags, ensure_ascii=False),
                item.collected_at.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_content_item(self, item_id: str) -> ContentItem | None:
        cursor = await self._conn.execute(
            "SELECT * FROM content_items WHERE id = ?", (item_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_content_item(row)

    async def url_exists(self, url: str) -> bool:
        cursor = await self._conn.execute(
            "SELECT 1 FROM content_items WHERE url = ?", (url,)
        )
        return await cursor.fetchone() is not None

    async def get_recent_content_items(self, limit: int = 50) -> list[ContentItem]:
        cursor = await self._conn.execute(
            "SELECT * FROM content_items ORDER BY collected_at DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_content_item(r) for r in rows]

    # -- DraftContent --

    async def save_draft(self, draft: DraftContent) -> None:
        await self._conn.execute(
            """INSERT INTO drafts
               (id, content_item_id, draft_type, body, target_thread_id, status,
                rejection_reason, published_thread_id, published_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                draft.id,
                draft.content_item_id,
                draft.draft_type.value,
                draft.body,
                draft.target_thread_id,
                draft.status.value,
                draft.rejection_reason,
                draft.published_thread_id,
                draft.published_at.isoformat() if draft.published_at else None,
                draft.created_at.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_draft(self, draft_id: str) -> DraftContent | None:
        cursor = await self._conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (draft_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_draft(row)

    async def get_drafts_by_status(self, status: DraftStatus) -> list[DraftContent]:
        cursor = await self._conn.execute(
            "SELECT * FROM drafts WHERE status = ? ORDER BY created_at DESC",
            (status.value,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_draft(r) for r in rows]

    async def update_draft_status(
        self, draft_id: str, status: DraftStatus, **kwargs: str
    ) -> None:
        sets = ["status = ?"]
        vals: list[str] = [status.value]
        for key, val in kwargs.items():
            sets.append(f"{key} = ?")
            vals.append(val)
        vals.append(draft_id)
        await self._conn.execute(
            f"UPDATE drafts SET {', '.join(sets)} WHERE id = ?", vals
        )
        await self._conn.commit()

    # -- helpers --

    @staticmethod
    def _row_to_content_item(row) -> ContentItem:
        return ContentItem(
            id=row["id"],
            source=ContentSource(row["source"]),
            url=row["url"],
            title=row["title"],
            summary=row["summary"],
            author=row["author"],
            published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
            likes=row["likes"],
            replies=row["replies"],
            tags=json.loads(row["tags"]),
            collected_at=datetime.fromisoformat(row["collected_at"]),
        )

    @staticmethod
    def _row_to_draft(row) -> DraftContent:
        return DraftContent(
            id=row["id"],
            content_item_id=row["content_item_id"],
            draft_type=DraftType(row["draft_type"]),
            body=row["body"],
            target_thread_id=row["target_thread_id"],
            status=DraftStatus(row["status"]),
            rejection_reason=row["rejection_reason"],
            published_thread_id=row["published_thread_id"],
            published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
```

**Step 8: Run all tests**

```bash
uv run pytest tests/test_models.py tests/test_database.py -v
```
Expected: all pass

**Step 9: Commit**

```bash
git add src/models/ src/db/ tests/test_models.py tests/test_database.py
git commit -m "feat: add pydantic models and async SQLite database layer"
```

---

## Task 3: LLM Provider Abstraction

**Files:**
- Create: `src/generator/llm_provider.py`
- Create: `tests/test_llm_provider.py`

**Step 1: Write tests for LLM providers**

`tests/test_llm_provider.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.generator.llm_provider import ClaudeProvider, OpenAIProvider, create_provider


async def test_claude_provider_generate():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="這是一篇關於展覽的貼文")]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = ClaudeProvider(client=mock_client, model="claude-sonnet-4-5-20250929")
    result = await provider.generate(
        prompt="寫一篇關於展覽的貼文",
        context={"topic": "當代藝術展"},
        persona={"tone": "輕鬆友善"},
    )
    assert "展覽" in result
    mock_client.messages.create.assert_called_once()


async def test_openai_provider_generate():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="展覽推薦貼文"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = OpenAIProvider(client=mock_client, model="gpt-4o")
    result = await provider.generate(
        prompt="寫一篇關於展覽的貼文",
        context={"topic": "當代藝術展"},
        persona={"tone": "輕鬆友善"},
    )
    assert "展覽" in result


def test_create_provider_claude():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        provider = create_provider("claude")
        assert isinstance(provider, ClaudeProvider)


def test_create_provider_openai():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = create_provider("openai")
        assert isinstance(provider, OpenAIProvider)


def test_create_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("unknown")
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_provider.py -v
```

**Step 3: Implement LLM providers**

`src/generator/llm_provider.py`:
```python
import json
import os
from typing import Any, Protocol

import anthropic
import openai


class LLMProvider(Protocol):
    async def generate(self, prompt: str, context: dict[str, Any], persona: dict[str, Any]) -> str: ...


def _build_system_prompt(persona: dict[str, Any]) -> str:
    return (
        f"你是 {persona.get('name', 'ARTOGO')} 的社群小編。\n"
        f"語氣：{persona.get('tone', '輕鬆友善')}\n"
        f"語言：{persona.get('language', 'zh-TW')}\n"
        f"風格指南：\n"
        + "\n".join(f"- {g}" for g in persona.get("style_guidelines", []))
        + "\n禁用詞彙："
        + "、".join(persona.get("forbidden_words", []))
    )


class ClaudeProvider:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None, model: str = "claude-sonnet-4-5-20250929"):
        self._client = client or anthropic.AsyncAnthropic()
        self._model = model

    async def generate(self, prompt: str, context: dict[str, Any], persona: dict[str, Any]) -> str:
        system = _build_system_prompt(persona)
        user_msg = f"{prompt}\n\n參考資料：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text


class OpenAIProvider:
    def __init__(self, client: openai.AsyncOpenAI | None = None, model: str = "gpt-4o"):
        self._client = client or openai.AsyncOpenAI()
        self._model = model

    async def generate(self, prompt: str, context: dict[str, Any], persona: dict[str, Any]) -> str:
        system = _build_system_prompt(persona)
        user_msg = f"{prompt}\n\n參考資料：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content


def create_provider(name: str) -> LLMProvider:
    if name == "claude":
        return ClaudeProvider()
    if name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider: {name}")
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_llm_provider.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/generator/llm_provider.py tests/test_llm_provider.py
git commit -m "feat: add swappable LLM provider abstraction (Claude + OpenAI)"
```

---

## Task 4: News Collector

**Files:**
- Create: `src/collector/base.py`
- Create: `src/collector/news.py`
- Create: `tests/test_collector_news.py`

**Step 1: Write tests**

`tests/test_collector_news.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch

from src.collector.base import BaseCollector
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


async def test_news_collector_parse():
    collector = NewsCollector(
        source_name="TestNews",
        base_url="https://example.com",
        article_selector="article.post-item",
        title_selector="h2 a",
        excerpt_selector="p.excerpt",
        author_selector="span.author",
        date_selector="time",
    )
    items = collector.parse_html(SAMPLE_HTML, "https://example.com")
    assert len(items) == 2
    assert items[0].title == "台北當代藝術館新展開幕"
    assert items[0].source == ContentSource.NEWS
    assert "exhibit/123" in items[0].url


async def test_news_collector_collect(monkeypatch):
    collector = NewsCollector(
        source_name="TestNews",
        base_url="https://example.com",
        article_selector="article.post-item",
        title_selector="h2 a",
        excerpt_selector="p.excerpt",
        author_selector="span.author",
        date_selector="time",
    )

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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_collector_news.py -v
```

**Step 3: Implement BaseCollector and NewsCollector**

`src/collector/base.py`:
```python
from abc import ABC, abstractmethod

from src.models.schemas import ContentItem


class BaseCollector(ABC):
    @abstractmethod
    async def collect(self) -> list[ContentItem]:
        ...
```

`src/collector/news.py`:
```python
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
                        published_at = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
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
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_collector_news.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/collector/ tests/test_collector_news.py
git commit -m "feat: add news collector with HTML parsing"
```

---

## Task 5: Threads Collector

**Files:**
- Create: `src/collector/threads.py`
- Create: `tests/test_collector_threads.py`

**Step 1: Write tests**

`tests/test_collector_threads.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_collector_threads.py -v
```

**Step 3: Implement ThreadsCollector**

`src/collector/threads.py`:
```python
from datetime import datetime, timezone

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
            # Use first 50 chars as title
            title = text[:50] + ("..." if len(text) > 50 else "")
            published_at = None
            if ts := post.get("timestamp"):
                try:
                    published_at = datetime.fromisoformat(ts.replace("+0000", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Extract hashtags as tags
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
        all_items: list[ContentItem] = []
        async with httpx.AsyncClient(timeout=30) as client:
            # Fetch own recent threads
            response = await client.get(
                f"{THREADS_API_BASE}/{self._user_id}/threads",
                params={
                    "fields": "id,text,username,timestamp,like_count,reply_count,permalink",
                    "access_token": self._access_token,
                    "limit": 25,
                },
            )
            response.raise_for_status()
            all_items.extend(self.parse_response(response.json()))
        return all_items
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_collector_threads.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/collector/threads.py tests/test_collector_threads.py
git commit -m "feat: add Threads API collector"
```

---

## Task 6: Analyzer - Strategy Engine

**Files:**
- Create: `src/analyzer/strategy.py`
- Create: `tests/test_analyzer.py`

**Step 1: Write tests**

`tests/test_analyzer.py`:
```python
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from src.analyzer.strategy import StrategyAnalyzer
from src.models.schemas import ActionType, ContentItem, ContentSource


@pytest.fixture
def analyzer(sample_settings, sample_persona):
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="8.0")
    return StrategyAnalyzer(
        settings=sample_settings["analyzer"],
        persona=sample_persona,
        llm=mock_llm,
    )


def _make_item(**kwargs) -> ContentItem:
    defaults = {
        "source": ContentSource.NEWS,
        "url": "https://example.com/1",
        "title": "台北當代藝術館新展",
        "summary": "當代藝術展覽開幕",
        "likes": 50,
        "replies": 10,
        "published_at": datetime.now(timezone.utc) - timedelta(hours=2),
    }
    defaults.update(kwargs)
    return ContentItem(**defaults)


def test_heat_score_high_engagement(analyzer):
    item = _make_item(likes=100, replies=30)
    score = analyzer.compute_heat_score(item)
    assert score > 5.0


def test_heat_score_low_engagement(analyzer):
    item = _make_item(likes=1, replies=0)
    score = analyzer.compute_heat_score(item)
    assert score < 3.0


def test_heat_score_decays_with_time(analyzer):
    recent = _make_item(
        likes=50,
        replies=10,
        published_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    old = _make_item(
        likes=50,
        replies=10,
        published_at=datetime.now(timezone.utc) - timedelta(days=7),
    )
    assert analyzer.compute_heat_score(recent) > analyzer.compute_heat_score(old)


def test_risk_filter_flags_political(analyzer):
    item = _make_item(title="這場展覽牽涉政治爭議")
    flags = analyzer.check_risk(item)
    assert len(flags) > 0
    assert "政治" in flags[0]


def test_risk_filter_clean(analyzer):
    item = _make_item(title="故宮新展覽好美")
    flags = analyzer.check_risk(item)
    assert len(flags) == 0


async def test_analyze_returns_suggestions(analyzer):
    items = [
        _make_item(likes=100, replies=20),
        _make_item(url="https://example.com/2", likes=2, replies=0, title="冷門展"),
    ]
    suggestions = await analyzer.analyze(items)
    # High engagement item should get post/reply, low one may be skipped
    assert len(suggestions) >= 1
    assert suggestions[0].action_type in (ActionType.POST, ActionType.REPLY)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_analyzer.py -v
```

**Step 3: Implement StrategyAnalyzer**

`src/analyzer/strategy.py`:
```python
import math
from datetime import datetime, timezone
from typing import Any

from src.generator.llm_provider import LLMProvider
from src.models.schemas import ActionSuggestion, ActionType, ContentItem


class StrategyAnalyzer:
    def __init__(
        self,
        settings: dict[str, Any],
        persona: dict[str, Any],
        llm: LLMProvider,
    ):
        self._settings = settings
        self._persona = persona
        self._llm = llm

    def compute_heat_score(self, item: ContentItem) -> float:
        """Score 0-10 based on engagement and recency."""
        engagement = item.likes + item.replies * 3
        # Log scale for engagement (0-7 points)
        eng_score = min(7.0, math.log1p(engagement) * 1.2)

        # Time decay (0-3 points, full score if < 6h, decays over 7 days)
        if item.published_at:
            age_hours = (datetime.now(timezone.utc) - item.published_at).total_seconds() / 3600
        else:
            age_hours = 48  # default assume 2 days old
        time_score = max(0.0, 3.0 * math.exp(-age_hours / 48))

        return min(10.0, eng_score + time_score)

    def check_risk(self, item: ContentItem) -> list[str]:
        """Return list of risk flag descriptions."""
        flags: list[str] = []
        risk_keywords = self._settings.get("risk_keywords", [])
        text = f"{item.title} {item.summary}".lower()
        for kw in risk_keywords:
            if kw in text:
                flags.append(f"包含風險關鍵字：{kw}")
        return flags

    async def _score_relevance(self, item: ContentItem) -> float:
        """Use LLM to score relevance 0-10."""
        prompt = (
            f"請評估以下內容與「藝術展覽平台」的相關程度，只回覆一個 0 到 10 的數字。\n"
            f"標題：{item.title}\n摘要：{item.summary}"
        )
        raw = await self._llm.generate(prompt, {}, self._persona)
        try:
            return float(raw.strip().split()[0])
        except (ValueError, IndexError):
            return 5.0

    async def analyze(self, items: list[ContentItem]) -> list[ActionSuggestion]:
        min_rel = self._settings.get("min_relevance_score", 5.0)
        min_heat = self._settings.get("min_heat_score", 3.0)
        suggestions: list[ActionSuggestion] = []

        for item in items:
            risk_flags = self.check_risk(item)
            if risk_flags:
                continue

            heat = self.compute_heat_score(item)
            if heat < min_heat:
                continue

            relevance = await self._score_relevance(item)
            if relevance < min_rel:
                continue

            # Decide action type
            if heat >= 7.0 and relevance >= 7.0:
                action = ActionType.POST
            elif heat >= 4.0 or relevance >= 6.0:
                action = ActionType.REPLY
            else:
                continue

            suggestions.append(
                ActionSuggestion(
                    content_item_id=item.id,
                    action_type=action,
                    relevance_score=relevance,
                    heat_score=heat,
                    suggested_angle=f"針對「{item.title[:30]}」分享觀點",
                    priority=int(heat + relevance),
                    risk_flags=risk_flags,
                )
            )

        suggestions.sort(key=lambda s: s.priority, reverse=True)
        return suggestions
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_analyzer.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/analyzer/ tests/test_analyzer.py
git commit -m "feat: add strategy analyzer with heat scoring and risk filtering"
```

---

## Task 7: Content Generator

**Files:**
- Create: `src/generator/content.py`
- Create: `tests/test_generator.py`

**Step 1: Write tests**

`tests/test_generator.py`:
```python
import pytest
from unittest.mock import AsyncMock

from src.generator.content import ContentGenerator
from src.models.schemas import (
    ActionSuggestion,
    ActionType,
    ContentItem,
    ContentSource,
    DraftType,
)


@pytest.fixture
def generator(sample_settings, sample_persona):
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(return_value="最近去看了這個展覽，真的很推薦大家去看看！")
    return ContentGenerator(
        settings=sample_settings["generator"],
        persona=sample_persona,
        llm=mock_llm,
    )


def _make_item() -> ContentItem:
    return ContentItem(
        source=ContentSource.NEWS,
        url="https://example.com/1",
        title="台北當代藝術館新展",
        summary="探討數位藝術的沉浸式展覽",
    )


def _make_suggestion(action: ActionType = ActionType.POST) -> ActionSuggestion:
    return ActionSuggestion(
        content_item_id="item-1",
        action_type=action,
        relevance_score=8.0,
        heat_score=7.0,
        suggested_angle="分享沉浸式體驗心得",
    )


async def test_generate_post(generator):
    item = _make_item()
    suggestion = _make_suggestion(ActionType.POST)
    draft = await generator.generate(item, suggestion)
    assert draft.draft_type == DraftType.POST
    assert len(draft.body) > 0
    assert len(draft.body) <= 500


async def test_generate_reply(generator):
    item = _make_item()
    suggestion = _make_suggestion(ActionType.REPLY)
    draft = await generator.generate(item, suggestion, target_thread_id="thread-123")
    assert draft.draft_type == DraftType.REPLY
    assert draft.target_thread_id == "thread-123"


async def test_forbidden_word_triggers_retry(sample_settings, sample_persona):
    mock_llm = AsyncMock()
    mock_llm.generate = AsyncMock(
        side_effect=[
            "這個展覽是業配推薦",  # contains forbidden word
            "這個展覽真的值得一看！",  # clean
        ]
    )
    generator = ContentGenerator(
        settings=sample_settings["generator"],
        persona=sample_persona,
        llm=mock_llm,
    )
    item = _make_item()
    suggestion = _make_suggestion()
    draft = await generator.generate(item, suggestion)
    assert "業配" not in draft.body
    assert mock_llm.generate.call_count == 2
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_generator.py -v
```

**Step 3: Implement ContentGenerator**

`src/generator/content.py`:
```python
from typing import Any

from src.generator.llm_provider import LLMProvider
from src.models.schemas import (
    ActionSuggestion,
    ActionType,
    ContentItem,
    DraftContent,
    DraftType,
)

_POST_PROMPT = (
    "根據以下話題，以品牌人設撰寫一篇 Threads 貼文。\n"
    "要求：\n"
    "- 繁體中文\n"
    "- 口語化、像在跟朋友聊天\n"
    "- 不超過 {max_length} 字\n"
    "- 不要使用 hashtag\n"
    "- 只輸出貼文內容，不要加任何前綴說明\n\n"
    "話題：{title}\n"
    "摘要：{summary}\n"
    "切入角度：{angle}"
)

_REPLY_PROMPT = (
    "根據以下討論串，以品牌人設撰寫一則自然的回覆。\n"
    "要求：\n"
    "- 繁體中文\n"
    "- 像朋友之間的對話\n"
    "- 不超過 {max_length} 字\n"
    "- 視情況提供有價值的資訊或自然互動\n"
    "- 只輸出回覆內容\n\n"
    "原文：{summary}\n"
    "切入角度：{angle}"
)


class ContentGenerator:
    def __init__(
        self,
        settings: dict[str, Any],
        persona: dict[str, Any],
        llm: LLMProvider,
    ):
        self._settings = settings
        self._persona = persona
        self._llm = llm
        self._forbidden = set(persona.get("forbidden_words", []))
        self._max_retries = settings.get("max_retries", 3)

    def _contains_forbidden(self, text: str) -> bool:
        return any(w in text for w in self._forbidden)

    async def generate(
        self,
        item: ContentItem,
        suggestion: ActionSuggestion,
        target_thread_id: str | None = None,
    ) -> DraftContent:
        is_post = suggestion.action_type == ActionType.POST
        max_length = self._settings.get(
            "max_post_length" if is_post else "max_reply_length", 500
        )
        template = _POST_PROMPT if is_post else _REPLY_PROMPT
        prompt = template.format(
            max_length=max_length,
            title=item.title,
            summary=item.summary,
            angle=suggestion.suggested_angle,
        )
        context = {
            "source": item.source.value,
            "url": item.url,
            "tags": item.tags,
        }

        body = ""
        for _ in range(self._max_retries):
            body = await self._llm.generate(prompt, context, self._persona)
            body = body.strip()
            if len(body) <= max_length and not self._contains_forbidden(body):
                break

        return DraftContent(
            content_item_id=item.id,
            draft_type=DraftType.POST if is_post else DraftType.REPLY,
            body=body[:max_length],
            target_thread_id=target_thread_id,
        )
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_generator.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/generator/content.py tests/test_generator.py
git commit -m "feat: add content generator with persona enforcement and retry logic"
```

---

## Task 8: Slack Reviewer

**Files:**
- Create: `src/reviewer/slack_bot.py`
- Create: `tests/test_reviewer.py`

**Step 1: Write tests**

`tests/test_reviewer.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

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
    # Should have header, content preview, source info, and action buttons
    assert len(blocks) >= 3
    # Find actions block
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
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_reviewer.py -v
```

**Step 3: Implement SlackReviewer**

`src/reviewer/slack_bot.py`:
```python
from typing import Any

from src.models.schemas import ContentItem, DraftContent, DraftType


def build_review_blocks(draft: DraftContent, item: ContentItem) -> list[dict[str, Any]]:
    type_label = "📝 新貼文" if draft.draft_type == DraftType.POST else "💬 回覆留言"
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{type_label} 待審核", "emoji": True},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*話題來源：* <{item.url}|{item.title}>\n*來源平台：* {item.source.value}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*內容預覽：*\n>>> {draft.body}",
            },
        },
        {"type": "divider"},
        {
            "type": "actions",
            "block_id": f"review_{draft.id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ 核准", "emoji": True},
                    "style": "primary",
                    "action_id": "approve_draft",
                    "value": draft.id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✏️ 編輯", "emoji": True},
                    "action_id": "edit_draft",
                    "value": draft.id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ 拒絕", "emoji": True},
                    "style": "danger",
                    "action_id": "reject_draft",
                    "value": draft.id,
                },
            ],
        },
    ]


def build_edit_modal(draft: DraftContent) -> dict[str, Any]:
    return {
        "type": "modal",
        "callback_id": "edit_draft_modal",
        "private_metadata": draft.id,
        "title": {"type": "plain_text", "text": "編輯內容"},
        "submit": {"type": "plain_text", "text": "核准並發布"},
        "close": {"type": "plain_text", "text": "取消"},
        "blocks": [
            {
                "type": "input",
                "block_id": "content_block",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "content_input",
                    "multiline": True,
                    "initial_value": draft.body,
                    "max_length": 500,
                },
                "label": {"type": "plain_text", "text": "貼文內容"},
            }
        ],
    }


class SlackReviewer:
    def __init__(self, client: Any, channel: str):
        self._client = client
        self._channel = channel

    async def send_for_review(self, draft: DraftContent, item: ContentItem) -> str:
        blocks = build_review_blocks(draft, item)
        result = self._client.chat_postMessage(
            channel=self._channel,
            text=f"待審核：{item.title}",
            blocks=blocks,
        )
        return result["ts"]
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_reviewer.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/reviewer/ tests/test_reviewer.py
git commit -m "feat: add Slack reviewer with Block Kit UI and edit modal"
```

---

## Task 9: Threads Publisher

**Files:**
- Create: `src/publisher/threads.py`
- Create: `tests/test_publisher.py`

**Step 1: Write tests**

`tests/test_publisher.py`:
```python
import pytest
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

    # Mock: create container -> publish container
    mock_create_resp = AsyncMock()
    mock_create_resp.json = lambda: {"id": "container-1"}
    mock_create_resp.raise_for_status = lambda: None

    mock_publish_resp = AsyncMock()
    mock_publish_resp.json = lambda: {"id": "thread-post-1"}
    mock_publish_resp.raise_for_status = lambda: None

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[mock_create_resp, mock_publish_resp])
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
    mock_client.post = AsyncMock(side_effect=[mock_create_resp, mock_publish_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    draft = _make_draft(DraftType.REPLY, target="parent-thread-1")
    with patch("httpx.AsyncClient", return_value=mock_client):
        thread_id = await publisher.publish(draft)

    assert thread_id == "reply-1"
    # Verify reply_to_id was included in the create call
    create_call_kwargs = mock_client.post.call_args_list[0]
    assert "reply_to_id" in str(create_call_kwargs)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_publisher.py -v
```

**Step 3: Implement ThreadsPublisher**

`src/publisher/threads.py`:
```python
import httpx

from src.models.schemas import DraftContent, DraftType

THREADS_API_BASE = "https://graph.threads.net/v1.0"


class ThreadsPublisher:
    def __init__(self, access_token: str, user_id: str):
        self._access_token = access_token
        self._user_id = user_id

    async def publish(self, draft: DraftContent) -> str:
        """Publish a draft to Threads. Returns the published thread ID."""
        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Create media container
            create_params: dict = {
                "media_type": "TEXT",
                "text": draft.body,
                "access_token": self._access_token,
            }
            if draft.draft_type == DraftType.REPLY and draft.target_thread_id:
                create_params["reply_to_id"] = draft.target_thread_id

            create_resp = await client.post(
                f"{THREADS_API_BASE}/{self._user_id}/threads",
                params=create_params,
            )
            create_resp.raise_for_status()
            container_id = create_resp.json()["id"]

            # Step 2: Publish the container
            publish_resp = await client.post(
                f"{THREADS_API_BASE}/{self._user_id}/threads_publish",
                params={
                    "creation_id": container_id,
                    "access_token": self._access_token,
                },
            )
            publish_resp.raise_for_status()
            return publish_resp.json()["id"]
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_publisher.py -v
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/publisher/ tests/test_publisher.py
git commit -m "feat: add Threads publisher with two-step publish flow"
```

---

## Task 10: Pipeline Orchestrator & Scheduler

**Files:**
- Create: `src/pipeline.py`
- Create: `scheduler.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write pipeline tests**

`tests/test_pipeline.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.pipeline import Pipeline
from src.models.schemas import (
    ActionSuggestion,
    ActionType,
    ContentItem,
    ContentSource,
    DraftContent,
    DraftType,
)


@pytest.fixture
def mock_deps():
    db = AsyncMock()
    db.url_exists = AsyncMock(return_value=False)
    db.save_content_item = AsyncMock()
    db.save_draft = AsyncMock()

    collector = AsyncMock()
    collector.collect = AsyncMock(
        return_value=[
            ContentItem(
                source=ContentSource.NEWS,
                url="https://example.com/1",
                title="新展覽",
                summary="好看的展覽",
                likes=50,
                replies=10,
                published_at=datetime.now(timezone.utc),
            )
        ]
    )

    analyzer = AsyncMock()
    analyzer.analyze = AsyncMock(
        return_value=[
            ActionSuggestion(
                content_item_id="item-1",
                action_type=ActionType.POST,
                relevance_score=8.0,
                heat_score=7.0,
                suggested_angle="分享心得",
            )
        ]
    )

    generator = AsyncMock()
    generator.generate = AsyncMock(
        return_value=DraftContent(
            content_item_id="item-1",
            draft_type=DraftType.POST,
            body="推薦這個展覽！",
        )
    )

    reviewer = AsyncMock()
    reviewer.send_for_review = AsyncMock(return_value="msg-ts-1")

    return db, [collector], analyzer, generator, reviewer


async def test_pipeline_run(mock_deps):
    db, collectors, analyzer, generator, reviewer = mock_deps
    pipeline = Pipeline(
        db=db,
        collectors=collectors,
        analyzer=analyzer,
        generator=generator,
        reviewer=reviewer,
    )
    await pipeline.run()

    # Verify full flow executed
    collectors[0].collect.assert_called_once()
    db.save_content_item.assert_called_once()
    analyzer.analyze.assert_called_once()
    generator.generate.assert_called_once()
    db.save_draft.assert_called_once()
    reviewer.send_for_review.assert_called_once()


async def test_pipeline_skips_existing_urls(mock_deps):
    db, collectors, analyzer, generator, reviewer = mock_deps
    db.url_exists = AsyncMock(return_value=True)  # URL already exists

    pipeline = Pipeline(
        db=db,
        collectors=collectors,
        analyzer=analyzer,
        generator=generator,
        reviewer=reviewer,
    )
    await pipeline.run()

    # Should not save or analyze since URL exists
    db.save_content_item.assert_not_called()
    analyzer.analyze.assert_called_once()  # called with empty list
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_pipeline.py -v
```

**Step 3: Implement Pipeline**

`src/pipeline.py`:
```python
import logging

from src.analyzer.strategy import StrategyAnalyzer
from src.collector.base import BaseCollector
from src.db.database import Database
from src.generator.content import ContentGenerator
from src.models.schemas import ContentItem
from src.reviewer.slack_bot import SlackReviewer

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        db: Database,
        collectors: list[BaseCollector],
        analyzer: StrategyAnalyzer,
        generator: ContentGenerator,
        reviewer: SlackReviewer,
    ):
        self._db = db
        self._collectors = collectors
        self._analyzer = analyzer
        self._generator = generator
        self._reviewer = reviewer

    async def run(self) -> None:
        # 1. Collect
        all_items: list[ContentItem] = []
        for collector in self._collectors:
            try:
                items = await collector.collect()
                logger.info("Collected %d items from %s", len(items), type(collector).__name__)
                all_items.extend(items)
            except Exception:
                logger.exception("Collector %s failed", type(collector).__name__)

        # 2. Deduplicate and save
        new_items: list[ContentItem] = []
        for item in all_items:
            if not await self._db.url_exists(item.url):
                await self._db.save_content_item(item)
                new_items.append(item)

        logger.info("New items after dedup: %d", len(new_items))

        # 3. Analyze
        suggestions = await self._analyzer.analyze(new_items)
        logger.info("Action suggestions: %d", len(suggestions))

        # 4. Generate and send for review
        # Build lookup for content items
        item_map = {item.id: item for item in new_items}
        for suggestion in suggestions:
            item = item_map.get(suggestion.content_item_id)
            if not item:
                continue
            try:
                draft = await self._generator.generate(item, suggestion)
                await self._db.save_draft(draft)
                await self._reviewer.send_for_review(draft, item)
                logger.info("Sent draft %s for review", draft.id)
            except Exception:
                logger.exception("Failed to generate/review for item %s", item.id)
```

**Step 4: Implement scheduler.py (entry point)**

`scheduler.py`:
```python
import asyncio
import logging
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config import init_env, load_persona, load_settings
from src.db.database import Database
from src.analyzer.strategy import StrategyAnalyzer
from src.collector.news import NewsCollector
from src.collector.threads import ThreadsCollector
from src.generator.content import ContentGenerator
from src.generator.llm_provider import create_provider
from src.pipeline import Pipeline
from src.publisher.threads import ThreadsPublisher
from src.reviewer.slack_bot import SlackReviewer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    init_env()
    settings = load_settings()
    persona = load_persona()

    # Database
    db_path = os.getenv("DATABASE_PATH", "data/social_assistant.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = Database(db_path)
    await db.init()

    # LLM
    llm = create_provider(os.getenv("LLM_PROVIDER", "claude"))

    # Collectors
    collectors = []
    threads_token = os.getenv("THREADS_ACCESS_TOKEN")
    threads_user = os.getenv("THREADS_USER_ID")
    if threads_token and threads_user:
        collectors.append(ThreadsCollector(threads_token, threads_user))

    for src in settings["collector"].get("news_sources", []):
        if src.get("type") == "scrape":
            collectors.append(
                NewsCollector(
                    source_name=src["name"],
                    base_url=src["url"],
                    article_selector="article",
                    title_selector="h2 a, h3 a",
                    excerpt_selector="p",
                )
            )

    # Analyzer, Generator
    analyzer = StrategyAnalyzer(settings["analyzer"], persona, llm)
    generator = ContentGenerator(settings["generator"], persona, llm)

    # Slack
    from slack_sdk import WebClient
    slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
    reviewer = SlackReviewer(slack_client, os.getenv("SLACK_REVIEW_CHANNEL", "#social-review"))

    # Publisher (for approved items — handled by Slack bot event loop)
    _publisher = ThreadsPublisher(threads_token or "", threads_user or "")

    # Pipeline
    pipeline = Pipeline(db, collectors, analyzer, generator, reviewer)

    # Scheduler
    scheduler = AsyncIOScheduler()
    interval = settings["collector"].get("schedule_interval_hours", 4)
    scheduler.add_job(pipeline.run, "interval", hours=interval, id="pipeline")
    scheduler.start()

    logger.info("Social Assistant started. Pipeline runs every %d hours.", interval)

    # Run once immediately
    await pipeline.run()

    # Keep alive
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        await db.close()
        logger.info("Social Assistant stopped.")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 5: Run pipeline tests**

```bash
uv run pytest tests/test_pipeline.py -v
```
Expected: all pass

**Step 6: Run full test suite**

```bash
uv run pytest -v
```
Expected: all pass

**Step 7: Commit**

```bash
git add src/pipeline.py scheduler.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator and scheduler entry point"
```

---

## Task 11: Slack Event Handlers (Approve / Edit / Reject)

**Files:**
- Create: `src/reviewer/handlers.py`
- Create: `tests/test_handlers.py`

**Step 1: Write tests**

`tests/test_handlers.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

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
    # After publish, status should be updated to PUBLISHED
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

    # Should update body, approve, then publish
    publisher.publish.assert_called_once()
    published_draft = publisher.publish.call_args[0][0]
    assert published_draft.body == "修改後的內容"
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_handlers.py -v
```

**Step 3: Implement handlers**

`src/reviewer/handlers.py`:
```python
import logging

from src.db.database import Database
from src.models.schemas import DraftStatus
from src.publisher.threads import ThreadsPublisher

logger = logging.getLogger(__name__)


async def handle_approve(draft_id: str, db: Database, publisher: ThreadsPublisher) -> str:
    """Approve and publish a draft. Returns the published thread ID."""
    draft = await db.get_draft(draft_id)
    if not draft:
        raise ValueError(f"Draft {draft_id} not found")

    await db.update_draft_status(draft_id, DraftStatus.APPROVED)

    try:
        thread_id = await publisher.publish(draft)
        await db.update_draft_status(
            draft_id, DraftStatus.PUBLISHED, published_thread_id=thread_id
        )
        logger.info("Published draft %s as thread %s", draft_id, thread_id)
        return thread_id
    except Exception:
        await db.update_draft_status(draft_id, DraftStatus.FAILED)
        logger.exception("Failed to publish draft %s", draft_id)
        raise


async def handle_reject(draft_id: str, db: Database, reason: str = "") -> None:
    """Reject a draft with optional reason."""
    await db.update_draft_status(
        draft_id, DraftStatus.REJECTED, rejection_reason=reason
    )
    logger.info("Rejected draft %s: %s", draft_id, reason)


async def handle_edit_submit(
    draft_id: str, new_body: str, db: Database, publisher: ThreadsPublisher
) -> str:
    """Update draft body, approve, and publish."""
    draft = await db.get_draft(draft_id)
    if not draft:
        raise ValueError(f"Draft {draft_id} not found")

    draft.body = new_body
    await db.update_draft_status(draft_id, DraftStatus.EDITED)

    try:
        thread_id = await publisher.publish(draft)
        await db.update_draft_status(
            draft_id, DraftStatus.PUBLISHED, published_thread_id=thread_id
        )
        logger.info("Published edited draft %s as thread %s", draft_id, thread_id)
        return thread_id
    except Exception:
        await db.update_draft_status(draft_id, DraftStatus.FAILED)
        logger.exception("Failed to publish edited draft %s", draft_id)
        raise
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_handlers.py -v
```
Expected: all pass

**Step 5: Run full test suite**

```bash
uv run pytest -v
```
Expected: all pass

**Step 6: Commit**

```bash
git add src/reviewer/handlers.py tests/test_handlers.py
git commit -m "feat: add Slack action handlers for approve, reject, and edit flows"
```

---

## Task 12: Final Integration & Verification

**Step 1: Run full test suite with coverage**

```bash
uv run pytest -v --tb=short
```
Expected: all tests pass

**Step 2: Run ruff linter**

```bash
uv run ruff check src/ tests/
```
Expected: no errors (or fix any that appear)

**Step 3: Verify project structure**

```bash
find /Users/walt/Desktop/Ai/ARTOGO/social-assistant -type f -name "*.py" | sort
```

Expected files:
```
scheduler.py
src/__init__.py
src/analyzer/__init__.py
src/analyzer/strategy.py
src/collector/__init__.py
src/collector/base.py
src/collector/news.py
src/collector/threads.py
src/config.py
src/db/__init__.py
src/db/database.py
src/generator/__init__.py
src/generator/content.py
src/generator/llm_provider.py
src/models/__init__.py
src/models/schemas.py
src/pipeline.py
src/publisher/__init__.py
src/publisher/threads.py
src/reviewer/__init__.py
src/reviewer/handlers.py
src/reviewer/slack_bot.py
tests/__init__.py
tests/conftest.py
tests/test_analyzer.py
tests/test_collector_news.py
tests/test_collector_threads.py
tests/test_database.py
tests/test_generator.py
tests/test_handlers.py
tests/test_llm_provider.py
tests/test_models.py
tests/test_pipeline.py
tests/test_publisher.py
tests/test_reviewer.py
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete social-assistant v0.1 with full test coverage"
```

---

## Summary

| Task | Description | Key Files | Tests |
|------|-------------|-----------|-------|
| 1 | Project scaffolding | pyproject.toml, config/, src/config.py | setup verification |
| 2 | Database & Models | src/models/schemas.py, src/db/database.py | test_models, test_database |
| 3 | LLM Provider | src/generator/llm_provider.py | test_llm_provider |
| 4 | News Collector | src/collector/news.py, base.py | test_collector_news |
| 5 | Threads Collector | src/collector/threads.py | test_collector_threads |
| 6 | Strategy Analyzer | src/analyzer/strategy.py | test_analyzer |
| 7 | Content Generator | src/generator/content.py | test_generator |
| 8 | Slack Reviewer | src/reviewer/slack_bot.py | test_reviewer |
| 9 | Threads Publisher | src/publisher/threads.py | test_publisher |
| 10 | Pipeline & Scheduler | src/pipeline.py, scheduler.py | test_pipeline |
| 11 | Slack Handlers | src/reviewer/handlers.py | test_handlers |
| 12 | Integration | — | full suite |
