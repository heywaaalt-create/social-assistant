"""Tests for the content generator."""

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
    mock_llm.generate = AsyncMock(
        return_value="最近去看了這個展覽，真的很推薦大家去看看！"
    )
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
