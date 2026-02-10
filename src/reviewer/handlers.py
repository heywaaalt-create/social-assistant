"""Slack action handlers for approve, reject, and edit flows."""

import logging

from src.db.database import Database
from src.models.schemas import DraftStatus
from src.publisher.threads import ThreadsPublisher

logger = logging.getLogger(__name__)


async def handle_approve(
    draft_id: str, db: Database, publisher: ThreadsPublisher
) -> str:
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


async def handle_reject(
    draft_id: str, db: Database, reason: str = ""
) -> None:
    """Reject a draft with optional reason."""
    await db.update_draft_status(
        draft_id, DraftStatus.REJECTED, rejection_reason=reason
    )
    logger.info("Rejected draft %s: %s", draft_id, reason)


async def handle_edit_submit(
    draft_id: str,
    new_body: str,
    db: Database,
    publisher: ThreadsPublisher,
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
        logger.info(
            "Published edited draft %s as thread %s", draft_id, thread_id
        )
        return thread_id
    except Exception:
        await db.update_draft_status(draft_id, DraftStatus.FAILED)
        logger.exception("Failed to publish edited draft %s", draft_id)
        raise
