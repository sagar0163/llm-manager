"""
LangChain Integration Adapter for LLM Manager

This module provides a LangChain LLM wrapper that uses llm-manager under the hood,
enabling seamless use of llm-manager's failover/cost-tracking features within
existing LangChain workflows.

Usage:
    from langchain_adapter import LangChainLLM
    
    llm = LangChainLLM(
        primary_provider="gemini",
        secondary_provider="groq",
        model="gemini-1.5-flash"
    )
    
    response = llm.invoke("Explain quantum computing")
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List

from langchain.language_models import BaseLLM
from langchain.schema import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.utils import get_from_env

from llm_manager import LLMManager, LLMResponse, ProviderConfig, Provider

logger = logging.getLogger(__name__)


class LangChainLLM(BaseLLM):
    """
    A LangChain-compatible LLM wrapper that uses llm-manager under the hood.
    
    This wrapper provides:
    - Automatic failover (Primary → Secondary → Local/Ollama)
    - Cost tracking integration
    - Safe fallback responses
    - Async support for generate method
    
    Example:
        >>> llm = LangChainLLM(
        ...     primary_provider="gemini",
        ...     secondary_provider="groq",
        ...     model="gemini-1.5-flash"
        ... )
        >>> response = llm.invoke("Hello, how are you?")
    """
    
    def __init__(
        self,
        primary_provider: str = "gemini",
        secondary_provider: str = "groq",
        model: str = "gemini-1.5-flash",
        local_provider_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        """
        Initialize the LangChainLLM wrapper.
        
        Args:
            primary_provider: Primary LLM provider (e.g., "gemini", "openai", "groq")
            secondary_provider: Secondary LLM provider for failover
            model: Default model name
            local_provider_url: URL for local Ollama provider
            temperature: Default temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to LLMManager
        """
        super().__init__(**kwargs)
        
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.model = model
        self.local_provider_url = local_provider_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the LLMManager
        self._manager: Optional[LLMManager] = None
        self._initialized = False
    
    async def _initialize(self) -> None:
        """Initialize the underlying LLMManager."""
        if self._initialized:
            return
        
        self._manager = LLMManager({
            "primary_provider": self.primary_provider,
            "secondary_provider": self.secondary_provider,
            "local_provider_url": self.local_provider_url,
            "fallback_strategy": "ordered",
            "gemini_model": self.model,
            "groq_model": self.model,
            "ollama_model": self.model,
        })
        
        await self._manager.initialize()
        self._initialized = True
        logger.info(f"LangChainLLM initialized with {self.primary_provider} as primary")
    
    async def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using llm-manager with failover support.
        
        Args:
            prompts: List of prompts (LangChain passes a list, we use the first)
            stop: Optional stop sequences
            callbacks: Optional callbacks
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        await self._initialize()
        
        # Use the first prompt
        prompt = prompts[0] if prompts else ""
        
        # Merge temperature and max_tokens from kwargs with defaults
        temp = kwargs.get("temperature", self.temperature)
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        
        # Generate text using llm-manager
        response = await self._manager.generate_text(
            prompt,
            temperature=temp,
            max_tokens=max_tok,
            prompt_type="generate",
        )
        
        if response.success:
            return response.text
        else:
            # Return fallback response
            logger.warning(f"LLM generation failed: {response.error}")
            return get_from_env("FALLBACK_RESPONSE", "I apologize, but I'm unable to process your request right now.")
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "langchain-llm-manager"
    
    @property
    def _user_id(self) -> Optional[str]:
        """Optional user ID for tracing."""
        return None
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata about the LLM."""
        return {
            "primary_provider": self.primary_provider,
            "secondary_provider": self.secondary_provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# Convenience function for quick instantiation
def create_langchain_llm(
    primary_provider: str = "gemini",
    secondary_provider: str = "groq",
    model: str = "gemini-1.5-flash",
    **kwargs: Any,
) -> LangChainLLM:
    """
    Create a LangChainLLM instance with default settings.
    
    Args:
        primary_provider: Primary LLM provider
        secondary_provider: Secondary LLM provider for failover
        model: Default model name
        **kwargs: Additional arguments for LangChainLLM
        
    Returns:
        Configured LangChainLLM instance
    """
    return LangChainLLM(
        primary_provider=primary_provider,
        secondary_provider=secondary_provider,
        model=model,
        **kwargs,
)


# Async generator for streaming (optional extension)
async def generate_with_failover(
    prompt: str,
    primary_provider: str = "gemini",
    secondary_provider: str = "groq",
    model: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    Quick async function to generate text with failover.
    
    Args:
        prompt: The prompt to send
        primary_provider: Primary provider
        secondary_provider: Secondary provider for failover
        model: Model name
        temperature: Generation temperature
        max_tokens: Maximum tokens
        
    Returns:
        Generated text
    """
    llm = LangChainLLM(
        primary_provider=primary_provider,
        secondary_provider=secondary_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    await llm._initialize()
    return await llm._generate([prompt])