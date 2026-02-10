"""Threads publisher: posts content via Meta Threads API."""

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
