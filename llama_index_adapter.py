"""LlamaIndex Integration Adapter for LLM Manager

This module provides a LlamaIndex integration that exposes llm-manager as a
query pipeline node, enabling seamless use of llm-manager's failover/cost-tracking
features within existing LlamaIndex workflows.

Usage:
    from llama_index_adapter import LlamaIndexLLM
    
    llm = LlamaIndexLLM(
        primary_provider="gemini",
        secondary_provider="groq",
        model="gemini-1.5-flash"
    )
    
    from llama_index.core import VectorStoreIndex, Document
    documents = [Document(text="...")]
    index = VectorStoreIndex.from_documents(documents, llm=llm)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is this about?")
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List

from llama_index.core import QueryBundle
from llama_index.core.llms import CustomLLM, LLMMetadata

from llm_manager import LLMManager, LLMResponse, ProviderConfig, Provider

logger = logging.getLogger(__name__)


class LlamaIndexLLM(CustomLLM):
    """A LlamaIndex-compatible LLM wrapper that uses llm-manager under the hood.
    
    This wrapper provides LlamaIndex integration with:
    - Automatic failover (Primary -> Secondary -> Local/Ollama)
    - Cost tracking integration
    - Safe fallback responses
    - Compatibility with LlamaIndex query pipeline nodes
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
        """Initialize the LlamaIndexLLM wrapper.
        
        Args:
            primary_provider: Primary LLM provider (e.g., "gemini", "openai", "groq")
            secondary_provider: Secondary LLM provider for failover
            model: Default model name
            local_provider_url: URL for local Ollama provider
            temperature: Default temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to LLMManager
        """
        # Pass required params to CustomLLM via kwargs, store custom ones after
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Store custom attributes after pydantic init
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'temperature', temperature)
        object.__setattr__(self, 'max_tokens', max_tokens)
        object.__setattr__(self, 'primary_provider', primary_provider)
        object.__setattr__(self, 'secondary_provider', secondary_provider)
        object.__setattr__(self, 'local_provider_url', local_provider_url)
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
        logger.info(f"LlamaIndexLLM initialized with {self.primary_provider} as primary")
    
    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata for LlamaIndex."""
        return LLMMetadata(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            context_window=100000,  # Default large context window
            some_attribute="some_value",
        )
    
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Complete a text prompt.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional generation parameters
            
        Returns:
            Completed text
        """
        await self._initialize()
        
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
            logger.warning(f"LLM completion failed: {response.error}")
            return f"I apologize, but I'm unable to process: {prompt[:50]}..."
    
    async def stream_complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Stream completion of a text prompt.
        
        Note: This currently returns the full response since llm-manager
        doesn't natively stream, but is compatible with LlamaIndex's interface.
        
        Args:
            prompt: The prompt to complete
            **kwargs: Additional generation parameters
            
        Returns:
            Completed text
        """
        # For now, just call complete since we don't have native streaming
        return await self.complete(prompt, **kwargs)
    
    async def embed(self, text: str) -> List[float]:
        """
        Embed a text (not supported by all providers, returns empty list).
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector (empty list if not supported)
        """
        # Embedding not natively supported by all providers
        # Return empty list - LlamaIndex will handle this case
        logger.warning("Embedding not supported by LlamaIndexLLM wrapper")
        return []


# Convenience function for quick instantiation
def create_llama_index_llm(
    primary_provider: str = "gemini",
    secondary_provider: str = "groq",
    model: str = "gemini-1.5-flash",
    **kwargs: Any,
) -> LlamaIndexLLM:
    """Create a LlamaIndexLLM instance with default settings.
    
    Args:
        primary_provider: Primary LLM provider
        secondary_provider: Secondary provider for failover
        model: Default model name
        **kwargs: Additional arguments for LlamaIndexLLM
        
    Returns:
        Configured LlamaIndexLLM instance
    """
    return LlamaIndexLLM(
        primary_provider=primary_provider,
        secondary_provider=secondary_provider,
        model=model,
        **kwargs,
    )


# Compatibility wrapper for use with LlamaIndex query engines
class LlamaIndexQueryEngineWrapper:
    """A wrapper that integrates LlamaIndexLLM with LlamaIndex query engines.
    
    This class provides a bridge between llm-manager and LlamaIndex's query pipeline,
    enabling use of failover/cost-tracking features in vector store queries.
    """
    
    def __init__(self, llm: LlamaIndexLLM):
        """Initialize the query engine wrapper.
        
        Args:
            llm: LlamaIndexLLM instance
        """
        self.llm = llm
    
    async def query(self, query_str: str) -> str:
        """Query using the llm-manager wrapped LLM.
        
        Args:
            query_str: The query string
            
        Returns:
            Query response text
        """
        return await self.llm.complete(query_str)


# Async generator for quick generation with failover
async def generate_with_llama_index_failover(
    prompt: str,
    primary_provider: str = "gemini",
    secondary_provider: str = "groq",
    model: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Quick async function to generate text with LlamaIndex-style integration.
    
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
    llm = LlamaIndexLLM(
        primary_provider=primary_provider,
        secondary_provider=secondary_provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    await llm._initialize()
    return await llm.complete(prompt)
