"""Tests for the pipeline orchestrator."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

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

    item = ContentItem(
        id="item-1",
        source=ContentSource.NEWS,
        url="https://example.com/1",
        title="新展覽",
        summary="好看的展覽",
        likes=50,
        replies=10,
        published_at=datetime.now(timezone.utc),
    )

    collector = AsyncMock()
    collector.collect = AsyncMock(return_value=[item])

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

    collectors[0].collect.assert_called_once()
    db.save_content_item.assert_called_once()
    analyzer.analyze.assert_called_once()
    generator.generate.assert_called_once()
    db.save_draft.assert_called_once()
    reviewer.send_for_review.assert_called_once()


async def test_pipeline_skips_existing_urls(mock_deps):
    db, collectors, analyzer, generator, reviewer = mock_deps
    db.url_exists = AsyncMock(return_value=True)

    pipeline = Pipeline(
        db=db,
        collectors=collectors,
        analyzer=analyzer,
        generator=generator,
        reviewer=reviewer,
    )
    await pipeline.run()

    db.save_content_item.assert_not_called()
    analyzer.analyze.assert_called_once()
