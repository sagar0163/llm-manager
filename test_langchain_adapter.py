"""Unit tests for LangChain Integration Adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_adapter import LangChainLLM, create_langchain_llm, generate_with_failover
from llm_manager import LLMResponse


class TestLangChainLLMInit:
    def test_default_init(self):
        llm = LangChainLLM()
        assert llm.primary_provider == "gemini"
        assert llm.secondary_provider == "groq"
        assert llm.model == "gemini-1.5-flash"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 2048

    def test_custom_init(self):
        llm = LangChainLLM(
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


class TestLangChainLLMCreate:
    def test_create_langchain_llm(self):
        llm = create_langchain_llm()
        assert llm.primary_provider == "gemini"
        assert llm.secondary_provider == "groq"

    def test_create_langchain_llm_custom(self):
        llm = create_langchain_llm(
            primary_provider="groq",
            model="llama-3.1-70b",
        )
        assert llm.primary_provider == "groq"
        assert llm.model == "llama-3.1-70b"


class TestLangChainLLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_with_mock_manager(self):
        llm = LangChainLLM()
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="Hello!", provider="gemini", model="flash", success=True)
        )
        llm._initialized = True

        response = await llm._generate(["Hello, how are you?"])
        assert response == "Hello!"

    @pytest.mark.asyncio
    async def test_generate_failover_primary_success(self):
        """Test that when primary succeeds, we return its response."""
        llm = LangChainLLM(primary_provider="gemini", secondary_provider="groq")
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="Primary response", provider="gemini", model="flash", success=True)
        )
        llm._initialized = True

        response = await llm._generate(["Test prompt"])
        assert response == "Primary response"

    @pytest.mark.asyncio
    async def test_generate_all_fail_returns_fallback(self):
        """Test that when all providers fail, fallback is returned."""
        llm = LangChainLLM(primary_provider="gemini", secondary_provider="groq")
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="", provider="gemini", model="flash", success=False, error="all failed")
        )
        llm._initialized = True

        response = await llm._generate(["Test prompt"])
        assert response == "I apologize, but I'm unable to process your request right now."

    @pytest.mark.asyncio
    async def test_generate_with_temperature_and_tokens(self):
        llm = LangChainLLM(temperature=0.5, max_tokens=512)
        llm._manager = MagicMock()
        llm._manager.generate_text = AsyncMock(
            return_value=LLMResponse(text="Response", provider="gemini", model="flash", success=True)
        )
        llm._initialized = True

        response = await llm._generate(["Prompt"], temperature=0.8, max_tokens=100)
        assert response == "Response"


class TestLangChainLLMProperties:
    def test_llm_type(self):
        llm = LangChainLLM()
        assert llm._llm_type == "langchain-llm-manager"

    def test_user_id(self):
        llm = LangChainLLM()
        assert llm._user_id is None

    def test_metadata(self):
        llm = LangChainLLM(temperature=0.3, max_tokens=1000)
        metadata = llm.metadata
        assert metadata["temperature"] == 0.3
        assert metadata["max_tokens"] == 1000
        assert metadata["primary_provider"] == "gemini"
        assert metadata["secondary_provider"] == "groq"
        assert metadata["model"] == "gemini-1.5-flash"


class TestLangChainLLMInitialize:
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        llm = LangChainLLM()
        # Call initialize twice
        await llm._initialize()
        await llm._initialize()  # Should not re-initialize
        assert llm._initialized is True


class TestGenerateWithFailover:
    @pytest.mark.asyncio
    @patch("langchain_adapter.LangChainLLM._generate")
    async def test_generate_with_failover_defaults(self, mock_generate):
        """Test the convenience function."""
        mock_generate.return_value = "Failover test"
        result = await generate_with_failover("test prompt")
        assert result == "Failover test"

    @pytest.mark.asyncio
    @patch("langchain_adapter.LangChainLLM._generate")
    async def test_generate_with_failover_custom(self, mock_generate):
        """Test the convenience function with custom providers."""
        mock_generate.return_value = "Custom failover"
        result = await generate_with_failover(
            "test prompt",
            primary_provider="openai",
            secondary_provider="groq",
            model="gpt-4",
        )
        assert result == "Custom failover"
