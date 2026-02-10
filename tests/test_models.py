"""Tests for Pydantic models."""

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


class TestContentItem:
    def test_content_item_creation(self):
        """Verify id auto-generated, source correct, collected_at auto-set."""
        item = ContentItem(
            source=ContentSource.THREADS,
            url="https://threads.net/t/123",
            title="Test Thread",
        )
        assert item.id is not None and len(item.id) > 0
        assert item.source == ContentSource.THREADS
        assert item.collected_at is not None
        assert isinstance(item.collected_at, datetime)
        assert item.summary == ""
        assert item.author == ""
        assert item.likes == 0
        assert item.replies == 0
        assert item.tags == []
        assert item.published_at is None

    def test_content_item_with_all_fields(self):
        """Verify all fields can be set explicitly."""
        now = datetime.now(timezone.utc)
        item = ContentItem(
            source=ContentSource.NEWS,
            url="https://example.com/article",
            title="Art Exhibition",
            summary="A great exhibition",
            author="Reporter",
            published_at=now,
            likes=42,
            replies=7,
            tags=["art", "exhibition"],
        )
        assert item.source == ContentSource.NEWS
        assert item.summary == "A great exhibition"
        assert item.author == "Reporter"
        assert item.published_at == now
        assert item.likes == 42
        assert item.replies == 7
        assert item.tags == ["art", "exhibition"]

    def test_content_item_unique_ids(self):
        """Each ContentItem should get a unique id."""
        item1 = ContentItem(
            source=ContentSource.OFFICIAL,
            url="https://example.com/1",
            title="Item 1",
        )
        item2 = ContentItem(
            source=ContentSource.ARTOGO,
            url="https://example.com/2",
            title="Item 2",
        )
        assert item1.id != item2.id


class TestActionSuggestion:
    def test_action_suggestion_creation(self):
        """Verify action_type and relevance_score."""
        suggestion = ActionSuggestion(
            content_item_id="test-content-id",
            action_type=ActionType.POST,
            relevance_score=8,
            heat_score=6,
        )
        assert suggestion.id is not None
        assert suggestion.action_type == ActionType.POST
        assert suggestion.relevance_score == 8
        assert suggestion.heat_score == 6
        assert suggestion.suggested_angle == ""
        assert suggestion.priority == 0
        assert suggestion.risk_flags == []
        assert suggestion.created_at is not None

    def test_action_suggestion_with_details(self):
        """Verify all optional fields work."""
        suggestion = ActionSuggestion(
            content_item_id="test-content-id",
            action_type=ActionType.REPLY,
            relevance_score=7,
            heat_score=9,
            suggested_angle="Respond with exhibition info",
            priority=5,
            risk_flags=["controversial"],
        )
        assert suggestion.action_type == ActionType.REPLY
        assert suggestion.suggested_angle == "Respond with exhibition info"
        assert suggestion.priority == 5
        assert suggestion.risk_flags == ["controversial"]

    def test_action_suggestion_skip(self):
        """Verify skip action type works."""
        suggestion = ActionSuggestion(
            content_item_id="test-content-id",
            action_type=ActionType.SKIP,
            relevance_score=2,
            heat_score=1,
        )
        assert suggestion.action_type == ActionType.SKIP


class TestDraftContent:
    def test_draft_content_creation(self):
        """Verify status defaults to PENDING, body length check."""
        draft = DraftContent(
            content_item_id="test-content-id",
            draft_type=DraftType.POST,
            body="This is a test post body",
        )
        assert draft.id is not None
        assert draft.status == DraftStatus.PENDING
        assert draft.body == "This is a test post body"
        assert len(draft.body) > 0
        assert draft.target_thread_id is None
        assert draft.rejection_reason == ""
        assert draft.published_thread_id is None
        assert draft.published_at is None
        assert draft.created_at is not None

    def test_draft_content_reply(self):
        """Verify reply draft with target_thread_id."""
        draft = DraftContent(
            content_item_id="test-content-id",
            draft_type=DraftType.REPLY,
            body="This is a reply",
            target_thread_id="thread-456",
        )
        assert draft.draft_type == DraftType.REPLY
        assert draft.target_thread_id == "thread-456"

    def test_draft_content_published(self):
        """Verify published draft has all metadata."""
        now = datetime.now(timezone.utc)
        draft = DraftContent(
            content_item_id="test-content-id",
            draft_type=DraftType.POST,
            body="Published post",
            status=DraftStatus.PUBLISHED,
            published_thread_id="published-123",
            published_at=now,
        )
        assert draft.status == DraftStatus.PUBLISHED
        assert draft.published_thread_id == "published-123"
        assert draft.published_at == now


class TestEnums:
    def test_content_source_values(self):
        assert ContentSource.THREADS == "threads"
        assert ContentSource.NEWS == "news"
        assert ContentSource.OFFICIAL == "official"
        assert ContentSource.ARTOGO == "artogo"

    def test_action_type_values(self):
        assert ActionType.POST == "post"
        assert ActionType.REPLY == "reply"
        assert ActionType.SKIP == "skip"

    def test_draft_type_values(self):
        assert DraftType.POST == "post"
        assert DraftType.REPLY == "reply"

    def test_draft_status_values(self):
        assert DraftStatus.PENDING == "pending"
        assert DraftStatus.APPROVED == "approved"
        assert DraftStatus.EDITED == "edited"
        assert DraftStatus.REJECTED == "rejected"
        assert DraftStatus.PUBLISHED == "published"
        assert DraftStatus.FAILED == "failed"
