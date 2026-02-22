"""
Caching Layer for LLM Manager
==============================
Smart response caching to reduce API calls
"""

import hashlib
import json
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path


@dataclass
class CacheEntry:
    """Cached response"""
    key: str
    response_text: str
    provider: str
    model: str
    created_at: datetime
    expires_at: datetime
    hits: int = 0


class LLMCache:
    """
    Smart caching for LLM responses
    
    Features:
    - Prompt-based caching
    - TTL (time to live) support
    - Hit tracking
    - Automatic cleanup
    
    Usage:
        cache = LLMCache()
        
        # Check cache before API call
        cached = cache.get(prompt)
        if cached:
            return cached
        
        # After getting response
        cache.set(prompt, response.text)
    """
    
    def __init__(
        self,
        cache_dir: str = ".llm_cache",
        default_ttl: int = 3600,  # 1 hour
        max_entries: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        
        # Load from disk
        self._load_index()
    
    def _generate_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and kwargs"""
        # Include model and other params in key
        key_data = {
            'prompt': prompt,
            'model': kwargs.get('model', ''),
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 2048)
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Use hash for shorter keys
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _load_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "index.json"
        
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    
                    for key, entry in data.items():
                        entry['created_at'] = datetime.fromisoformat(entry['created_at'])
                        entry['expires_at'] = datetime.fromisoformat(entry['expires_at'])
                        self._cache[key] = CacheEntry(**entry)
                    
            except Exception:
                pass  # Start fresh if error
    
    def _save_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "index.json"
        
        data = {
            key: {
                'key': entry.key,
                'response_text': entry.response_text,
                'provider': entry.provider,
                'model': entry.model,
                'created_at': entry.created_at.isoformat(),
                'expires_at': entry.expires_at.isoformat(),
                'hits': entry.hits
            }
            for key, entry in self._cache.items()
        }
        
        with open(index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def get(self, prompt: str, **kwargs) -> Optional[str]:
        """Get cached response if available"""
        key = self._generate_key(prompt, **kwargs)
        
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if datetime.now() > entry.expires_at:
                del self._cache[key]
                self._save_index()
                return None
            
            # Update hit count
            entry.hits += 1
            self._save_index()
            
            return entry.response_text
    
    async def set(
        self, 
        prompt: str, 
        response: str,
        provider: str = "unknown",
        model: str = "unknown",
        ttl: Optional[int] = None,
        **kwargs
    ):
        """Cache a response"""
        key = self._generate_key(prompt, **kwargs)
        
        async with self._lock:
            # Cleanup if at max
            if len(self._cache) >= self.max_entries:
                await self._cleanup()
            
            ttl = ttl or self.default_ttl
            now = datetime.now()
            
            entry = CacheEntry(
                key=key,
                response_text=response,
                provider=provider,
                model=model,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                hits=0
            )
            
            self._cache[key] = entry
            self._save_index()
    
    async def _cleanup(self):
        """Clean up expired and least-used entries"""
        now = datetime.now()
        
        # Remove expired
        expired = [
            key for key, entry in self._cache.items()
            if now > entry.expires_at
        ]
        
        for key in expired:
            del self._cache[key]
        
        # If still over limit, remove least used
        if len(self._cache) >= self.max_entries:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].hits
            )
            
            # Remove 10% least used
            remove_count = max(1, len(sorted_entries) // 10)
            for key, _ in sorted_entries[:remove_count]:
                del self._cache[key]
    
    async def invalidate(self, prompt: str, **kwargs):
        """Invalidate a specific cache entry"""
        key = self._generate_key(prompt,        async with self **kwargs)
        
._lock:
            if key in self._cache:
                del self._cache[key]
                self._save_index()
    
    async def clear(self):
        """Clear all cache"""
        async with self._lock:
            self._cache.clear()
            self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(entry.hits for entry in self._cache.values())
        expired = sum(
            1 for entry in self._cache.values()
            if datetime.now() > entry.expires_at
        )
        
        return {
            "total_entries": len(self._cache),
            "total_hits": total_hits,
            "expired_entries": expired,
            "cache_dir": str(self.cache_dir)
        }


# Semantic cache using embeddings (simple version)
class SemanticCache:
    """
    Semantic similarity-based caching
    
    Uses simple keyword matching for now.
    For production, would use embeddings.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.cache = LLMCache()
        self.similarity_threshold = similarity_threshold
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text"""
        # Simple word extraction
        words = text.lower().split()
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these', 'those', 'am', 'it', 'its'}
        return set(w for w in words if w not in stopwords and len(w) > 2)
    
    async def get_or_compute(
        self,
        prompt: str,
        compute_func,
        **kwargs
    ):
        """Get from cache or compute"""
        # Try exact match first
        cached = await self.cache.get(prompt, **kwargs)
        if cached:
            return cached
        
        # Extract keywords
        keywords = self._extract_keywords(prompt)
        
        # Would search semantic cache here
        # For now, compute and cache
        response = await compute_func(prompt)
        
        await self.cache.set(prompt, response, **kwargs)
        
        return response


# Example usage
async def example_caching():
    """Example of using caching"""
    
    cache = LLMCache()
    
    prompt = "What is Python?"
    
    # First call - cache miss
    print("First call...")
    cached = await cache.get(prompt)
    if cached:
        print(f"Cache hit: {cached}")
    else:
        print("Cache miss - would call API")
        await cache.set(prompt, "Python is a programming language.")
    
    # Second call - cache hit
    print("\nSecond call...")
    cached = await cache.get(prompt)
    if cached:
        print(f"Cache hit: {cached}")
    
    # Stats
    print(f"\n📊 Cache stats: {cache.get_stats()}")


if __name__ == "__main__":
    asyncio.run(example_caching())
