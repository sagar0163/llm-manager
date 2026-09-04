"""Unit tests for the real async llm_manager implementation.

These tests exercise the actual async multi-provider LLM wrapper (provider
dataclasses, factory, caching, failover/resilience, retries) without hitting
real network or provider SDKs.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_manager import (
    BaseLLMProvider,
    FallbackStrategy,
    GeminiProvider,
    GroqProvider,
    LLMManager,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    ProviderConfig,
    ProviderFactory,
    SafeFallbackCache,
    SAFE_FALLBACKS,
)


# =============================================================================
# Data classes
# =============================================================================


class TestLLMResponse:
    def test_construction_with_defaults(self):
        resp = LLMResponse(
            text="hello",
            provider="gemini",
            model="gemini-1.5-flash",
            success=True,
        )
        assert resp.text == "hello"
        assert resp.provider == "gemini"
        assert resp.model == "gemini-1.5-flash"
        assert resp.success is True
        assert resp.error is None
        assert resp.latency_ms is None
        assert resp.tokens_used is None
        assert isinstance(resp.timestamp, datetime)

    def test_full_construction(self):
        resp = LLMResponse(
            text="out",
            provider="openai",
            model="gpt-4o-mini",
            success=False,
            error="boom",
            latency_ms=12.5,
            tokens_used=42,
        )
        assert resp.error == "boom"
        assert resp.latency_ms == 12.5
        assert resp.tokens_used == 42


class TestProviderConfig:
    def test_defaults(self):
        cfg = ProviderConfig(name="gemini", provider_type=Provider.GEMINI)
        assert cfg.name == "gemini"
        assert cfg.provider_type == Provider.GEMINI
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.model == "default"
        assert cfg.max_retries == 2
        assert cfg.timeout == 30
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048

    def test_full_config(self):
        cfg = ProviderConfig(
            name="openai",
            provider_type=Provider.OPENAI,
            api_key="sk-test",
            base_url="https://api.openai.com",
            model="gpt-4",
            max_retries=5,
            timeout=60,
            temperature=0.1,
            max_tokens=512,
        )
        assert cfg.api_key == "sk-test"
        assert cfg.base_url == "https://api.openai.com"
        assert cfg.model == "gpt-4"
        assert cfg.max_retries == 5
        assert cfg.timeout == 60
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 512


# =============================================================================
# ProviderFactory
# =============================================================================


class TestProviderFactory:
    def test_create_gemini(self):
        cfg = ProviderConfig(name="gemini", provider_type=Provider.GEMINI)
        provider = ProviderFactory.create("gemini", cfg)
        assert isinstance(provider, GeminiProvider)
        assert isinstance(provider, BaseLLMProvider)

    def test_create_openai(self):
        cfg = ProviderConfig(name="openai", provider_type=Provider.OPENAI)
        provider = ProviderFactory.create("openai", cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_create_local_maps_to_ollama(self):
        cfg = ProviderConfig(name="ollama", provider_type=Provider.LOCAL)
        provider = ProviderFactory.create("local", cfg)
        assert isinstance(provider, OllamaProvider)

    def test_create_groq(self):
        cfg = ProviderConfig(name="groq", provider_type=Provider.GROQ)
        provider = ProviderFactory.create("groq", cfg)
        assert isinstance(provider, GroqProvider)

    def test_create_unknown_raises(self):
        cfg = ProviderConfig(name="nope", provider_type=Provider.GEMINI)
        with pytest.raises(ValueError):
            ProviderFactory.create("nope", cfg)

    def test_create_unknown_enum_raises(self):
        cfg = ProviderConfig(name="gemini", provider_type=Provider.GEMINI)
        with pytest.raises(ValueError):
            ProviderFactory.create("nope", cfg)

    def test_get_available_providers(self):
        providers = ProviderFactory.get_available_providers()
        expected = [p.value for p in Provider]
        assert providers == expected
        assert "gemini" in providers
        assert "groq" in providers
        assert "ollama" in providers


# =============================================================================
# SafeFallbackCache
# =============================================================================


class TestSafeFallbackCache:
    def test_hardcoded_fallback_returned_on_miss(self, tmp_path):
        cache = SafeFallbackCache(cache_dir=str(tmp_path))
        assert cache.get_safe_response("default") == SAFE_FALLBACKS["default"]
        assert cache.get_safe_response("summarize") == SAFE_FALLBACKS["summarize"]
        assert cache.get_safe_response("does-not-exist") == SAFE_FALLBACKS["default"]

    def test_set_get_roundtrip(self, tmp_path):
        cache = SafeFallbackCache(cache_dir=str(tmp_path))
        cache.set("analyze", "cached-analyze")
        assert cache.get("analyze") == "cached-analyze"
        assert cache.get_safe_response("analyze") == "cached-analyze"

    def test_cache_persists_to_disk(self, tmp_path):
        cache_dir = str(tmp_path / "sub")
        cache = SafeFallbackCache(cache_dir=cache_dir)
        cache.set("translate", "cached-translate")

        reloaded = SafeFallbackCache(cache_dir=cache_dir)
        assert reloaded.get("translate") == "cached-translate"

    def test_get_expired_entry_returns_none(self, tmp_path):
        cache = SafeFallbackCache(cache_dir=str(tmp_path))
        expired_ts = (datetime.now() - timedelta(hours=25)).isoformat()
        cache._cache["stale-key"] = {
            "response": "stale",
            "timestamp": expired_ts,
        }
        assert cache.get("stale-key") is None

    def test_get_fresh_entry_within_24h(self, tmp_path):
        cache = SafeFallbackCache(cache_dir=str(tmp_path))
        fresh_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        cache._cache["fresh-key"] = {
            "response": "fresh",
            "timestamp": fresh_ts,
        }
        assert cache.get("fresh-key") == "fresh"

    def test_returns_hardcoded_when_cache_expired(self, tmp_path):
        cache = SafeFallbackCache(cache_dir=str(tmp_path))
        expired_ts = (datetime.now() - timedelta(hours=30)).isoformat()
        cache._cache["generate"] = {
            "response": "stale",
            "timestamp": expired_ts,
        }
        assert cache.get_safe_response("generate") == SAFE_FALLBACKS["generate"]


# =============================================================================
# LLMManager config / init
# =============================================================================


class TestLLMConfig:
    def test_load_env_config_defaults(self, monkeypatch):
        manager = LLMManager()
        cfg = manager.config
        assert cfg["primary_provider"] == "gemini"
        assert cfg["secondary_provider"] == "groq"
        assert cfg["local_provider_url"] == "http://localhost:11434"
        assert cfg["fallback_strategy"] == "ordered"
        assert cfg["gemini_model"] == "gemini-1.5-flash"
        assert cfg["openai_model"] == "gpt-4o-mini"
        assert cfg["ollama_model"] == "llama3"
        assert cfg["groq_model"] == "llama-3.1-70b-versatile"

    def test_load_env_config_from_env(self, monkeypatch):
        monkeypatch.setenv("PRIMARY_PROVIDER", "openai")
        monkeypatch.setenv("SECONDARY_PROVIDER", "ollama")
        monkeypatch.setenv("GEMINI_MODEL", "custom-model")
        manager = LLMManager()
        assert manager.config["primary_provider"] == "openai"
        assert manager.config["secondary_provider"] == "ollama"
        assert manager.config["gemini_model"] == "custom-model"

    def test_custom_config_overrides_env(self):
        manager = LLMManager({"primary_provider": "openai", "secondary_provider": "ollama"})
        assert manager.config["primary_provider"] == "openai"
        assert manager.config["secondary_provider"] == "ollama"

    def test_initial_attributes(self, tmp_path):
        manager = LLMManager()
        assert manager.primary is None
        assert manager.secondary is None
        assert manager.local is None
        assert manager.current_provider is None


# =============================================================================
# generate_text resilience / failover
# =============================================================================


def _manager_with_providers(primary_Mock=True, secondary_Mock=True, local_Mock=True):
    manager = LLMManager({"primary_provider": "gemini", "secondary_provider": "groq"})
    if primary_Mock:
        p = MagicMock()
        p.config.name = "gemini"
        p.config.model = "m"
        manager.primary = p
    if secondary_Mock:
        s = MagicMock()
        s.config.name = "groq"
        s.config.model = "m"
        manager.secondary = s
    if local_Mock:
        l = MagicMock()
        l.config.name = "ollama"
        l.config.model = "m"
        manager.local = l
    return manager


def _success_response(text="ok", provider="x"):
    return LLMResponse(
        text=text,
        provider=provider,
        model="m",
        success=True,
    )


def _error_response(err="500 Internal Server Error"):
    return LLMResponse(
        text="",
        provider="x",
        model="m",
        success=False,
        error=err,
    )


@pytest.mark.asyncio
async def test_generate_success_on_primary():
    manager = _manager_with_providers()
    manager.primary.generate = AsyncMock(return_value=_success_response("primary-ok"))
    resp = await manager.generate_text("hello", prompt_type="default")
    assert resp.success
    assert resp.text == "primary-ok"
    assert manager.current_provider == "gemini"
    manager.secondary.generate.assert_not_called()
    manager.local.generate.assert_not_called()


@pytest.mark.asyncio
async def test_failover_primary_fails_to_secondary():
    manager = _manager_with_providers()
    manager.primary.generate = AsyncMock(return_value=_error_response())
    manager.secondary.generate = AsyncMock(return_value=_success_response("secondary-ok"))
    resp = await manager.generate_text("hello", prompt_type="default")
    assert resp.success
    assert resp.text == "secondary-ok"
    assert manager.current_provider == "groq"


@pytest.mark.asyncio
async def test_failover_primary_and_secondary_fail_to_local():
    manager = _manager_with_providers()
    manager.primary.generate = AsyncMock(return_value=_error_response())
    manager.secondary.generate = AsyncMock(return_value=_error_response())
    manager.local.generate = AsyncMock(return_value=_success_response("local-ok"))
    resp = await manager.generate_text("hello", prompt_type="default")
    assert resp.success
    assert resp.text == "local-ok"
    assert manager.current_provider == "ollama"


@pytest.mark.asyncio
async def test_all_providers_fail_returns_safe_fallback(tmp_path):
    manager = _manager_with_providers()
    manager.fallback_cache = SafeFallbackCache(cache_dir=str(tmp_path))
    manager.primary.generate = AsyncMock(return_value=_error_response())
    manager.secondary.generate = AsyncMock(return_value=_error_response())
    manager.local.generate = AsyncMock(return_value=_error_response())
    resp = await manager.generate_text("hello", prompt_type="generate")
    assert resp.provider == "fallback"
    assert resp.model == "safe-mode"
    assert resp.success
    assert resp.text == SAFE_FALLBACKS["generate"]


@pytest.mark.asyncio
async def test_skips_missing_secondary_and_local():
    manager = _manager_with_providers(secondary_Mock=False, local_Mock=False)
    manager.primary.generate = AsyncMock(return_value=_success_response("primary"))
    resp = await manager.generate_text("hello")
    assert resp.text == "primary"


@pytest.mark.asyncio
async def test_returns_fallback_when_no_providers_at_all(tmp_path):
    manager = _manager_with_providers(
        primary_Mock=False, secondary_Mock=False, local_Mock=False
    )
    manager.fallback_cache = SafeFallbackCache(cache_dir=str(tmp_path))
    resp = await manager.generate_text("hello", prompt_type="analyze")
    assert resp.provider == "fallback"
    assert resp.text == SAFE_FALLBACKS["analyze"]


# =============================================================================
# _try_generate retry / backoff
# =============================================================================


@pytest.mark.asyncio
async def test_try_generate_returns_success_immediately():
    manager = _manager_with_providers(secondary_Mock=False, local_Mock=False)
    provider = manager.primary
    provider.generate = AsyncMock(return_value=_success_response("ok"))
    resp = await manager._try_generate(provider, "hello", retries=2)
    assert resp.success
    assert provider.generate.await_count == 1


@pytest.mark.asyncio
async def test_try_generate_retries_on_retryable_error_with_backoff():
    manager = _manager_with_providers(secondary_Mock=False, local_Mock=False)
    provider = manager.primary
    provider.generate = AsyncMock(
        side_effect=[
            _error_response("500 Internal Server Error"),
            _error_response("502 Bad Gateway"),
            _success_response("recovered"),
        ]
    )
    with patch(
        "llm_manager.asyncio.sleep", new=AsyncMock()
    ) as mock_sleep:
        resp = await manager._try_generate(provider, "hello", retries=2)
    assert resp.success
    assert resp.text == "recovered"
    assert provider.generate.await_count == 3
    # backoff sleeps of 1s then 2s
    assert mock_sleep.await_args_list[0].args[0] == 1
    assert mock_sleep.await_args_list[1].args[0] == 2


@pytest.mark.asyncio
async def test_try_generate_non_retryable_error_no_retry():
    manager = _manager_with_providers(secondary_Mock=False, local_Mock=False)
    provider = manager.primary
    provider.generate = AsyncMock(
        return_value=_error_response("401 Unauthorized")
    )
    with patch("llm_manager.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        resp = await manager._try_generate(provider, "hello", retries=2)
    assert not resp.success
    assert provider.generate.await_count == 1
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_try_generate_exhausts_retries_returns_error():
    manager = _manager_with_providers(secondary_Mock=False, local_Mock=False)
    provider = manager.primary
    provider.generate = AsyncMock(
        return_value=_error_response("500 Internal Server Error")
    )
    with patch("llm_manager.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        resp = await manager._try_generate(provider, "hello", retries=2)
    assert not resp.success
    assert provider.generate.await_count == 3
    assert mock_sleep.await_count == 3


@pytest.mark.asyncio
async def test_try_generate_handles_raised_exception():
    manager = _manager_with_providers(secondary_Mock=False, local_Mock=False)
    provider = manager.primary
    provider.generate = AsyncMock(
        side_effect=[RuntimeError("network down"), _success_response("ok")]
    )
    with patch("llm_manager.asyncio.sleep", new=AsyncMock()):
        resp = await manager._try_generate(provider, "hello", retries=1)
    assert resp.success
    assert resp.text == "ok"


# =============================================================================
# health_check aggregation
# =============================================================================


@pytest.mark.asyncio
async def test_health_check_aggregates_all_providers():
    manager = _manager_with_providers()
    manager.primary.health_check = AsyncMock(return_value=True)
    manager.secondary.health_check = AsyncMock(return_value=False)
    manager.local.health_check = AsyncMock(return_value=True)
    status = await manager.health_check()
    assert status == {"gemini": True, "groq": False, "ollama": True}


@pytest.mark.asyncio
async def test_health_check_with_missing_providers():
    manager = _manager_with_providers(secondary_Mock=False, local_Mock=False)
    manager.primary.health_check = AsyncMock(return_value=True)
    status = await manager.health_check()
    assert status == {"gemini": True}
