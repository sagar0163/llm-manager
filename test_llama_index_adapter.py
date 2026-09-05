"""Unit tests for LlamaIndex Integration Adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index_adapter import (
    LlamaIndexLLM,
    create_llama_index_llm,
    LlamaIndexQueryEngineWrapper,
    generate_with_llama_index_failover,
)
from llm_manager import LLMResponse


class TestLlamaIndexLLMInit:
    def test_default_init(self):
        llm = LlamaIndexLLM()
        assert llm.primary_provider == "gemini"
        assert llm.secondary_provider == "groq"
        assert llm.model == "gemini-1.5-flash"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048

    def test_custom_init(self):
        llm = LlamaIndexLLM(
            primary_provider="openai",
            secondary_provider="anthropic",
            model="gpt-4o",
            temperature=0.5,
            max_tokens=1024,
        )
        assert llm.primary_provider == "openai"
        assert llm.secondary_provider == "anthropic"
        assert llm.model == "gpt-4o"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024


class TestLlamaIndexLLMCreate:
    def test_create_llama_index_llm(self):
        llm = create_llama_index_llm()
        assert isinstance(llm, LlamaIndexLLM)
        assert llm.primary_provider == "gemini"

    def test_create_llama_index_llm_custom(self):
        llm = create_llama_index_llm(
            primary_provider="groq",
            model="llama-3.1-70b",
        )
        assert llm.primary_provider == "groq"
        assert llm.model == "llama-3.1-70b"


class TestLlamaIndexLLMMetadata:
    def test_metadata_returns_llmmetadata(self):
        llm = LlamaIndexLLM(temperature=0.3, max_tokens=1000)
        metadata = llm.metadata
        assert hasattr(metadata, 'model_name')
        assert hasattr(metadata, 'max_tokens')
        assert hasattr(metadata, 'context_window')
        assert metadata.model_name == "gemini-1.5-flash"
        assert metadata.temperature == 0.3
        assert metadata.max_tokens == 1000
        assert metadata.context_window == 100000

    def test_metadata_defaults(self):
        llm = LlamaIndexLLM()
        metadata = llm.metadata
        assert metadata.model_name == "gemini-1.5-flash"
        assert metadata.max_tokens == 2048
        assert metadata.context_window == 100000


class TestLlamaIndexLLMComplete:
    @pytest.mark.asyncio
    async def test_complete_with_mock_manager(self):
        llm = LlamaIndexLLM()
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="Completed text", provider="gemini", model="flash", success=True)
        )
        llm._initialized = True

        result = await llm.complete("Test prompt")
        assert result == "Completed text"

    @pytest.mark.asyncio
    async def test_complete_with_temperature(self):
        llm = LlamaIndexLLM(temperature=0.5)
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="Response with temp", provider="gemini", model="flash", success=True)
        )
        llm._initialized = True

        result = await llm.complete("Test prompt", temperature=0.8)
        assert result == "Response with temp"

    @pytest.mark.asyncio
    async def test_complete_all_fail_returns_fallback(self):
        llm = LlamaIndexLLM(primary_provider="gemini", secondary_provider="groq")
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="", provider="gemini", model="flash", success=False, error="all failed")
        )
        llm._initialized = True

        result = await llm.complete("Test prompt")
        assert "I apologize" in result

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens(self):
        llm = LlamaIndexLLM(max_tokens=512)
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="Response with tokens", provider="gemini", model="flash", success=True)
        )
        llm._initialized = True

        result = await llm.complete("Test prompt", max_tokens=100)
        assert result == "Response with tokens"


class TestLlamaIndexLLMStreamComplete:
    @pytest.mark.asyncio
    @patch("llama_index_adapter.LlamaIndexLLM.complete", new_callable=AsyncMock)
    async def test_stream_complete_calls_complete(self, mock_complete):
        llm = LlamaIndexLLM()
        mock_complete.return_value = MagicMock(text="Streamed response")
        
        result = await llm.stream_complete("Test prompt")
        mock_complete.assert_awaited_once()
        assert result == "Streamed response"


class TestLlamaIndexLLMEmbed:
    @pytest.mark.asyncio
    async def test_embedding_returns_empty_list(self):
        llm = LlamaIndexLLM()
        result = await llm.embed("test text")
        assert result == []


class TestLlamaIndexQueryEngineWrapper:
    def test_wrapper_init(self):
        llm = LlamaIndexLLM()
        wrapper = LlamaIndexQueryEngineWrapper(llm)
        assert wrapper.llm is llm

    @pytest.mark.asyncio
    async def test_wrapper_query(self):
        llm = LlamaIndexLLM()
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="Query response", provider="gemini", model="flash", success=True)
        )
        llm._initialized = True

        wrapper = LlamaIndexQueryEngineWrapper(llm)
        result = await wrapper.query("What is this?")
        assert result == "Query response"


class TestGenerateWithLlamaIndexFailover:
    @pytest.mark.asyncio
    @patch("llama_index_adapter.LlamaIndexLLM.complete", new_callable=AsyncMock)
    async def test_generate_with_llama_index_failover_defaults(self, mock_complete):
        mock_complete.return_value = "LlamaIndex failover test"
        result = await generate_with_llama_index_failover("test prompt")
        assert result == "LlamaIndex failover test"

    @pytest.mark.asyncio
    @patch("llama_index_adapter.LlamaIndexLLM.complete", new_callable=AsyncMock)
    async def test_generate_with_llama_index_failover_custom(self, mock_complete):
        mock_complete.return_value = "Custom llama index"
        result = await generate_with_llama_index_failover(
            "test prompt",
            primary_provider="openai",
            secondary_provider="anthropic",
            model="gpt-4",
        )
        assert result == "Custom llama index"
