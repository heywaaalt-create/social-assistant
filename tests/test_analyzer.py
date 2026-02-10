"""Tests for the strategy analyzer."""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import pytest

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
    assert score < 5.0


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
        _make_item(
            url="https://example.com/2",
            likes=2,
            replies=0,
            title="冷門展",
        ),
    ]
    suggestions = await analyzer.analyze(items)
    assert len(suggestions) >= 1
    assert suggestions[0].action_type in (ActionType.POST, ActionType.REPLY)
