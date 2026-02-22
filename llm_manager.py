"""
LLMManager - Resilient Multi-Provider LLM Wrapper
=================================================

A production-ready wrapper for multiple LLM providers with:
- Environment-driven configuration
- Automatic failover (Primary → Secondary → Local)
- Async support
- Comprehensive logging
- Safe fallback responses
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("LLMManager")


# =============================================================================
# DATA CLASSES
# =============================================================================

class Provider(Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    LOCAL = "local"


class FallbackStrategy(Enum):
    """Fallback strategies"""
    ORDERED = "ordered"      # Try providers in order
    LATENCY = "latency"    # Try by fastest first (not implemented)


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    text: str
    provider: str
    model: str
    success: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProviderConfig:
    """Provider configuration"""
    name: str
    provider_type: Provider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "default"
    max_retries: int = 2
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 2048


# =============================================================================
# SAFE FALLBACKS
# =============================================================================

SAFE_FALLBACKS = {
    "default": "I apologize, but I'm temporarily unable to process your request. Please try again later.",
    "summarize": "I cannot provide a summary at this time. Please try again later.",
    "generate": "I cannot generate content at this moment. Please try again shortly.",
    "analyze": "I apologize, but I'm unable to analyze this input right now.",
    "translate": "Translation service is temporarily unavailable. Please try again later."
}


class SafeFallbackCache:
    """Local JSON cache for fallback responses"""
    
    def __init__(self, cache_dir: str = ".llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "fallback_cache.json"
        self._cache: Dict[str, Any] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load fallback cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save fallback cache: {e}")
    
    def get(self, key: str) -> Optional[str]:
        """Get cached fallback"""
        entry = self._cache.get(key)
        if entry:
            # Check if expired (24 hours)
            cached_time = datetime.fromisoformat(entry.get('timestamp', '2000-01-01'))
            if (datetime.now() - cached_time).hours < 24:
                return entry.get('response')
        return None
    
    def set(self, key: str, response: str):
        """Cache a fallback response"""
        self._cache[key] = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache()
    
    def get_safe_response(self, prompt_type: str = "default") -> str:
        """Get safe fallback response"""
        # Try cache first
        cached = self.get(prompt_type)
        if cached:
            return cached
        
        # Return hardcoded fallback
        return SAFE_FALLBACKS.get(prompt_type, SAFE_FALLBACKS["default"])


# =============================================================================
# ABSTRACT PROVIDER
# =============================================================================

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider client"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available"""
        pass
    
    def _create_response(self, text: str, **kwargs) -> LLMResponse:
        """Create standardized response"""
        return LLMResponse(
            text=text,
            provider=self.config.name,
            model=self.config.model,
            success=True,
            **kwargs
        )
    
    def _create_error_response(self, error: Exception) -> LLMResponse:
        """Create error response"""
        return LLMResponse(
            text="",
            provider=self.config.name,
            model=self.config.model,
            success=False,
            error=str(error)
        )


# =============================================================================
# PROVIDER IMPLEMENTATIONS
# =============================================================================

class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider"""
    
    async def initialize(self) -> bool:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key)
            self.client = genai
            logger.info(f"✅ Gemini provider initialized with model: {self.config.model}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            model = self.client.GenerativeModel(self.config.model)
            
            generation_config = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_output_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }
            
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=generation_config
            )
            
            return self._create_response(response.text)
            
        except Exception as e:
            logger.error(f"❌ Gemini generation failed: {e}")
            return self._create_error_response(e)
    
    async def health_check(self) -> bool:
        try:
            # Simple health check - try to list models
            models = self.client.list_models()
            return len(models) > 0
        except:
            return False


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider"""
    
    async def initialize(self) -> bool:
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info(f"✅ OpenAI provider initialized with model: {self.config.model}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get('model', self.config.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
            )
            
            text = response.choices[0].message.content
            return self._create_response(
                text,
                tokens_used=response.usage.total_tokens if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            return self._create_error_response(e)
    
    async def health_check(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except:
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama Local provider"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
    
    async def initialize(self) -> bool:
        try:
            import aiohttp
            self.client = aiohttp
            logger.info(f"✅ Ollama provider initialized at: {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ollama: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": kwargs.get('model', self.config.model),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "num_predict": kwargs.get('max_tokens', self.config.max_tokens)
                }
            }
            
            async with self.client.ClientSession() as session:
                async with session.post(url, json=payload, timeout=self.config.timeout) as resp:
                    if resp.status != 200:
                        raise Exception(f"Ollama returned {resp.status}")
                    
                    data = await resp.json()
                    text = data.get('response', '')
            
            return self._create_response(text)
            
        except Exception as e:
            logger.error(f"❌ Ollama generation failed: {e}")
            return self._create_error_response(e)
    
    async def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            async with self.client.ClientSession() as session:
                async with session.get(url, timeout=5) as resp:
                    return resp.status == 200
        except:
            return False


class GroqProvider(BaseLLMProvider):
    """Groq provider (fast inference)"""
    
    async def initialize(self) -> bool:
        try:
            from groq import AsyncGroq
            self.client = AsyncGroq(api_key=self.config.api_key)
            logger.info(f"✅ Groq provider initialized with model: {self.config.model}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        try:
            response = await self.client.chat.completions.create(
                model=kwargs.get('model', self.config.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
            )
            
            text = response.choices[0].message.content
            return self._create_response(text)
            
        except Exception as e:
            logger.error(f"❌ Groq generation failed: {e}")
            return self._create_error_response(e)
    
    async def health_check(self) -> bool:
        return self.client is not None


# =============================================================================
# PROVIDER FACTORY
# =============================================================================

class ProviderFactory:
    """Factory for creating LLM providers"""
    
    _providers = {
        Provider.GEMINI: GeminiProvider,
        Provider.OPENAI: OpenAIProviderOLLAMA: Oll,
        Provider.amaProvider,
        Provider.LOCAL: OllamaProvider,
        Provider.GROQ: GroqProvider,
    }
    
    @classmethod
    def create(cls, provider_type: str, config: ProviderConfig) -> BaseLLMProvider:
        """Create a provider instance"""
        provider_enum = Provider(provider_type.lower())
        
        if provider_enum not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_type}")
        
        return cls._providers[provider_enum](config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider names"""
        return [p.value for p in Provider]


# =============================================================================
# MAIN LLM MANAGER
# =============================================================================

class LLMManager:
    """
    Resilient LLM Manager with automatic failover
    =============================================
    
    Features:
    - Environment-driven configuration
    - Multi-provider support (Gemini, OpenAI, Ollama, Groq)
    - Automatic failover (Primary → Secondary → Local)
    - Retry logic with exponential backoff
    - Comprehensive logging
    - Safe fallback responses
    - Full async/await support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM Manager
        
        Args:
            config: Optional config dict. If None, loads from environment.
        """
        # Load configuration from environment or use provided config
        self.config = config or self._load_env_config()
        
        # Initialize providers
        self.primary = None
        self.secondary = None
        self.local = None
        
        # Fallback cache
        self.fallback_cache = SafeFallbackCache()
        
        # Track current provider for logging
        self.current_provider = None
        
        logger.info("🚀 LLMManager initialized")
        logger.info(f"   Primary: {self.config['primary_provider']}")
        logger.info(f"   Secondary: {self.config['secondary_provider']}")
        logger.info(f"   Local: {self.config.get('local_provider_url', 'N/A')}")
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'primary_provider': os.getenv('PRIMARY_PROVIDER', 'gemini').lower(),
            'secondary_provider': os.getenv('SECONDARY_PROVIDER', 'groq').lower(),
            'local_provider_url': os.getenv('LOCAL_PROVIDER_URL', 'http://localhost:11434'),
            'fallback_strategy': os.getenv('FALLBACK_STRATEGY', 'ordered'),
            
            # API Keys
            'gemini_api_key': os.getenv('GEMINI_API_KEY'),
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            
            # Default models
            'gemini_model': os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),
            'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            'ollama_model': os.getenv('OLLAMA_MODEL', 'llama3'),
            'groq_model': os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile'),
        }
    
    async def initialize(self) -> bool:
        """Initialize all configured providers"""
        logger.info("📦 Initializing providers...")
        
        # Initialize primary
        self.primary = await self._create_provider(
            self.config['primary_provider'],
            self.config.get(f"{self.config['primary_provider']}_api_key"),
            self.config.get(f"{self.config['primary_provider']}_model", "default")
        )
        
        # Initialize secondary
        self.secondary = await self._create_provider(
            self.config['secondary_provider'],
            self.config.get(f"{self.config['secondary_provider']}_api_key"),
            self.config.get(f"{self.config['secondary_provider']}_model", "default")
        )
        
        # Initialize local (Ollama)
        self.local = await self._create_provider(
            'ollama',
            None,
            self.config.get('ollama_model', 'llama3'),
            self.config['local_provider_url']
        )
        
        return True
    
    async def _create_provider(self, provider_type: str, api_key: Optional[str] = None, 
                               model: str = "default", base_url: Optional[str] = None) -> Optional[BaseLLMProvider]:
        """Create and initialize a provider"""
        try:
            config = ProviderConfig(
                name=provider_type,
                provider_type=Provider(provider_type),
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            
            provider = ProviderFactory.create(provider_type, config)
            success = await provider.initialize()
            
            if success:
                logger.info(f"   ✅ {provider_type} ready")
                return provider
            else:
                logger.warning(f"   ⚠️ {provider_type} initialization failed")
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Failed to create {provider_type}: {e}")
            return None
    
    async def generate_text(self, prompt: str, prompt_type: str = "default", 
                           **kwargs) -> LLMResponse:
        """
        Generate text with resilience loop
        ==================================
        
        Step 1: Try Primary (2 retries on 5xx errors)
        Step 2: If fails, try Secondary
        Step 3: If fails, try Local (Ollama)
        Step 4: If all fail, return Safe Fallback
        """
        
        # Step 1: Try Primary
        if self.primary:
            logger.info(f"🎯 Trying PRIMARY ({self.config['primary_provider']})...")
            self.current_provider = self.config['primary_provider']
            
            response = await self._try_generate(self.primary, prompt, retries=2, **kwargs)
            if response.success:
                logger.info(f"✅ Primary succeeded!")
                return response
            
            logger.warning(f"❌ Primary failed: {response.error}")
        
        # Step 2: Try Secondary
        if self.secondary:
            logger.info(f"🎯 Trying SECONDARY ({self.config['secondary_provider']})...")
            self.current_provider = self.config['secondary_provider']
            
            response = await self._try_generate(self.secondary, prompt, retries=1, **kwargs)
            if response.success:
                logger.info(f"✅ Secondary succeeded!")
                return response
            
            logger.warning(f"❌ Secondary failed: {response.error}")
        
        # Step 3: Try Local
        if self.local:
            logger.info(f"🎯 Trying LOCAL (Ollama)...")
            self.current_provider = "ollama"
            
            response = await self._try_generate(self.local, prompt, retries=0, **kwargs)
            if response.success:
                logger.info(f"✅ Local succeeded!")
                return response
            
            logger.warning(f"❌ Local failed: {response.error}")
        
        # Step 4: Safe Fallback
        logger.error("� ALL PROVIDERS FAILED! Returning safe fallback...")
        return self._get_fallback(prompt_type)
    
    async def _try_generate(self, provider: BaseLLMProvider, prompt: str, 
                           retries: int = 2, **kwargs) -> LLMResponse:
        """Try to generate with retries"""
        
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                response = await provider.generate(prompt, **kwargs)
                
                if response.success:
                    return response
                
                # Check if it's a 5xx error (retryable)
                if response.error and any(err in str(response.error) for err in ['500', '502', '503', '504', 'rate limit']):
                    last_error = response.error
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"   ⚠️ Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Non-retryable error
                return response
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"   ❌ Exception: {e}")
                
                if attempt < retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
        
        return LLMResponse(
            text="",
            provider=provider.config.name,
            model=provider.config.model,
            success=False,
            error=last_error or "Max retries exceeded"
        )
    
    def _get_fallback(self, prompt_type: str) -> LLMResponse:
        """Get safe fallback response"""
        fallback_text = self.fallback_cache.get_safe_response(prompt_type)
        
        logger.warning(f"📦 Returning fallback: {fallback_text[:50]}...")
        
        return LLMResponse(
            text=fallback_text,
            provider="fallback",
            model="safe-mode",
            success=True  # Mark as success since we have a fallback
        )
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        status = {}
        
        if self.primary:
            status[self.config['primary_provider']] = await self.primary.health_check()
        
        if self.secondary:
            status[self.config['secondary_provider']] = await self.secondary.health_check()
        
        if self.local:
            status['ollama'] = await self.local.health_check()
        
        return status


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def main():
    """Example usage"""
    
    # Initialize manager
    manager = LLMManager()
    await manager.initialize()
    
    # Check health
    health = await manager.health_check()
    print(f"\n🏥 Provider Health: {health}\n")
    
    # Generate text
    print("🤖 Generating response...")
    response = await manager.generate_text(
        "Explain quantum computing in simple terms.",
        prompt_type="generate"
    )
    
    print(f"\n📝 Response:")
    print(f"   Provider: {response.provider}")
    print(f"   Model: {response.model}")
    print(f"   Success: {response.success}")
    print(f"   Text: {response.text[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
