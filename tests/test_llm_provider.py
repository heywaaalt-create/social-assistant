"""Tests for LLM provider abstraction layer."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.generator.llm_provider import (
    ClaudeProvider,
    OpenAIProvider,
    _build_system_prompt,
    create_provider,
)


@pytest.fixture
def persona():
    return {
        "name": "ARTOGO",
        "tone": "輕鬆友善",
        "language": "zh-TW",
        "style_guidelines": ["用口語化的方式分享藝術觀點", "偶爾帶點幽默感"],
        "forbidden_words": ["業配", "工商"],
    }


@pytest.fixture
def context():
    return {
        "title": "北美館新展覽",
        "summary": "一場關於當代藝術的大型展覽",
        "url": "https://example.com/exhibition",
    }


class TestBuildSystemPrompt:
    def test_build_system_prompt_contains_all_fields(self, persona):
        prompt = _build_system_prompt(persona)
        assert "ARTOGO" in prompt
        assert "輕鬆友善" in prompt
        assert "zh-TW" in prompt
        assert "用口語化的方式分享藝術觀點" in prompt
        assert "偶爾帶點幽默感" in prompt
        assert "業配" in prompt
        assert "工商" in prompt

    def test_build_system_prompt_format(self, persona):
        prompt = _build_system_prompt(persona)
        assert prompt.startswith("你是 ARTOGO 的社群小編。")
        assert "語氣：輕鬆友善" in prompt
        assert "語言：zh-TW" in prompt
        assert "風格指南：" in prompt
        assert "禁用詞彙：" in prompt


class TestClaudeProvider:
    async def test_claude_provider_default_model(self):
        mock_client = MagicMock()
        provider = ClaudeProvider(client=mock_client)
        assert provider.model == "claude-sonnet-4-5-20250929"

    async def test_claude_provider_custom_model(self):
        mock_client = MagicMock()
        provider = ClaudeProvider(client=mock_client, model="claude-haiku-35")
        assert provider.model == "claude-haiku-35"

    async def test_claude_provider_generate(self, persona, context):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="這是生成的貼文內容")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = ClaudeProvider(client=mock_client)
        result = await provider.generate(
            prompt="請寫一篇展覽介紹",
            context=context,
            persona=persona,
        )

        assert result == "這是生成的貼文內容"
        mock_client.messages.create.assert_awaited_once()

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert "ARTOGO" in call_kwargs["system"]
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        # user message should contain the prompt and context
        user_content = call_kwargs["messages"][0]["content"]
        assert "請寫一篇展覽介紹" in user_content
        assert "北美館新展覽" in user_content

    @patch("src.generator.llm_provider.anthropic")
    async def test_claude_provider_creates_client_if_none(self, mock_anthropic):
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()
        provider = ClaudeProvider()
        assert provider.client is not None
        mock_anthropic.AsyncAnthropic.assert_called_once()


class TestOpenAIProvider:
    async def test_openai_provider_default_model(self):
        mock_client = MagicMock()
        provider = OpenAIProvider(client=mock_client)
        assert provider.model == "gpt-4o"

    async def test_openai_provider_custom_model(self):
        mock_client = MagicMock()
        provider = OpenAIProvider(client=mock_client, model="gpt-4o-mini")
        assert provider.model == "gpt-4o-mini"

    async def test_openai_provider_generate(self, persona, context):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI 生成的內容"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(client=mock_client)
        result = await provider.generate(
            prompt="請寫一篇展覽介紹",
            context=context,
            persona=persona,
        )

        assert result == "OpenAI 生成的內容"
        mock_client.chat.completions.create.assert_awaited_once()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "ARTOGO" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "請寫一篇展覽介紹" in messages[1]["content"]
        assert "北美館新展覽" in messages[1]["content"]

    @patch("src.generator.llm_provider.openai")
    async def test_openai_provider_creates_client_if_none(self, mock_openai):
        mock_openai.AsyncOpenAI.return_value = MagicMock()
        provider = OpenAIProvider()
        assert provider.client is not None
        mock_openai.AsyncOpenAI.assert_called_once()


class TestCreateProvider:
    @patch("src.generator.llm_provider.anthropic")
    def test_create_provider_claude(self, mock_anthropic):
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()
        provider = create_provider("claude")
        assert isinstance(provider, ClaudeProvider)

    @patch("src.generator.llm_provider.openai")
    def test_create_provider_openai(self, mock_openai):
        mock_openai.AsyncOpenAI.return_value = MagicMock()
        provider = create_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_create_provider_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("gemini")
