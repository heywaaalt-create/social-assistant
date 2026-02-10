"""Slack reviewer: sends drafts for human approval via Block Kit UI."""

from typing import Any

from src.models.schemas import ContentItem, DraftContent, DraftType


def build_review_blocks(
    draft: DraftContent, item: ContentItem
) -> list[dict[str, Any]]:
    type_label = (
        "\ud83d\udcdd 新貼文" if draft.draft_type == DraftType.POST else "\ud83d\udcac 回覆留言"
    )
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{type_label} 待審核",
                "emoji": True,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*話題來源：* <{item.url}|{item.title}>\n"
                    f"*來源平台：* {item.source.value}"
                ),
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
                    "text": {
                        "type": "plain_text",
                        "text": "\u2705 核准",
                        "emoji": True,
                    },
                    "style": "primary",
                    "action_id": "approve_draft",
                    "value": draft.id,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "\u270f\ufe0f 編輯",
                        "emoji": True,
                    },
                    "action_id": "edit_draft",
                    "value": draft.id,
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "\u274c 拒絕",
                        "emoji": True,
                    },
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

    async def send_for_review(
        self, draft: DraftContent, item: ContentItem
    ) -> str:
        blocks = build_review_blocks(draft, item)
        result = self._client.chat_postMessage(
            channel=self._channel,
            text=f"待審核：{item.title}",
            blocks=blocks,
        )
        return result["ts"]
