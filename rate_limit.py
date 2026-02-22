"""
Rate Limiting for LLM Manager
==============================
Token bucket and request limiting
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import defaultdict


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    burst_size: int = 10


class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_token(self, tokens: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)


class RateLimiter:
    """
    Multi-provider rate limiter
    
    Usage:
        limiter = RateLimiter()
        
        # Check before request
        await limiter.check("gemini")
        
        # After request
        limiter.record("gemini", tokens_used=1000)
    """
    
    def __init__(self):
        self.buckets: Dict[str, TokenBucket] = {}
        self.request_counts: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
        
        # Default configurations per provider
        self.configs = {
            "gemini": RateLimitConfig(requests_per_minute=60, tokens_per_minute=100000),
            "openai": RateLimitConfig(requests_per_minute=500, tokens_per_minute=150000),
            "groq": RateLimitConfig(requests_per_minute=30, tokens_per_minute=6000),
            "anthropic": RateLimitConfig(requests_per_minute=50, tokens_per_minute=100000),
            "ollama": RateLimitConfig(requests_per_minute=1000, tokens_per_minute=1000000),
        }
    
    async def initialize(self):
        """Initialize buckets for each provider"""
        for provider, config in self.configs.items():
            # Convert requests per minute to rate per second
            rate = config.requests_per_minute / 60
            
            self.buckets[provider] = TokenBucket(
                rate=rate,
                capacity=config.burst_size
            )
    
    async def check(self, provider: str) -> bool:
        """
        Check if request is allowed
        
        Returns True if allowed, False if rate limited
        """
        if provider not in self.buckets:
            return True  # No limit for unknown providers
        
        # Check request limit
        bucket = self.buckets[provider]
        
        if not await bucket.acquire(1):
            return False
        
        # Check request count per minute
        async with self._lock:
            now = time.time()
            self.request_counts[provider] = [
                t for t in self.request_counts[provider]
                if now - t < 60  # Keep last minute
            ]
            
            config = self.configs.get(provider, RateLimitConfig())
            
            if len(self.request_counts[provider]) >= config.requests_per_minute:
                return False
            
            self.request_counts[provider].append(now)
        
        return True
    
    async def wait_until_available(self, provider: str):
        """Wait until request is allowed"""
        while not await self.check(provider):
            await asyncio.sleep(1)
    
    def record(self, provider: str, tokens_used: int = 0):
        """Record token usage"""
        # Could be used for token-based limiting
        pass
    
    def get_status(self, provider: str) -> Dict:
        """Get rate limit status for a provider"""
        if provider not in self.buckets:
            return {"available": True, "requests_remaining": "unlimited"}
        
        config = self.configs.get(provider, RateLimitConfig())
        
        async with self._lock:
            now = time.time()
            requests_last_minute = len([
                t for t in self.request_counts[provider]
                if now - t < 60
            ])
        
        return {
            "provider": provider,
            "requests_per_minute": config.requests_per_minute,
            "requests_remaining": config.requests_per_minute - requests_last_minute,
            "burst_size": config.burst_size,
        }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all providers"""
        return {
            provider: self.get_status(provider)
            for provider in self.buckets.keys()
        }


class CircuitBreaker:
    """
    Circuit breaker pattern for provider failover
    
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if execution is allowed"""
        async with self._lock:
            if self.state == "CLOSED":
                return True
            
            if self.state == "OPEN":
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    return True
                return False
            
            if self.state == "HALF_OPEN":
                return self.half_open_calls < self.half_open_max_calls
            
            return False
    
    async def record_success(self):
        """Record successful execution"""
        async with self._lock:
            if self.state == "HALF_OPEN":
                self.half_open_calls += 1
                
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = "CLOSED"
                    self.failure_count = 0
            
            elif self.state == "CLOSED":
                self.failure_count = 0
    
    async def record_failure(self):
        """Record failed execution"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state


# Multi-provider circuit breaker manager
class CircuitBreakerManager:
    """Manage circuit breakers for multiple providers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider"""
        if provider not in self.breakers:
            self.breakers[provider] = CircuitBreaker()
        return self.breakers[provider]
    
    async def can_execute(self, provider: str) -> bool:
        """Check if provider can execute"""
        return await self.get_breaker(provider).can_execute()
    
    async def record_success(self, provider: str):
        """Record success for provider"""
        await self.get_breaker(provider).record_success()
    
    async def record_failure(self, provider: str):
        """Record failure for provider"""
        await self.get_breaker(provider).record_failure()
    
    def get_all_states(self) -> Dict[str, str]:
        """Get all circuit breaker states"""
        return {
            provider: breaker.get_state()
            for provider, breaker in self.breakers.items()
        }


# Example usage
async def example_rate_limiting():
    """Example of using rate limiting"""
    
    limiter = RateLimiter()
    await limiter.initialize()
    
    # Check if we can make a request
    if await limiter.check("gemini"):
        print("✅ Request allowed!")
    else:
        print("⏳ Rate limited, waiting...")
        await limiter.wait_until_available("gemini")
    
    # Get status
    status = limiter.get_status("gemini")
    print(f"📊 Status: {status}")
    
    # Circuit breaker
    cb_manager = CircuitBreakerManager()
    
    for i in range(10):
        can_exec = await cb_manager.can_execute("gemini")
        print(f"Request {i+1}: {'✅' if can_exec else '❌'}")
        
        if can_exec:
            # Simulate success/failure
            if i % 3 == 0:
                await cb_manager.record_failure("gemini")
            else:
                await cb_manager.record_success("gemini")
        
        print(f"  Circuit state: {cb_manager.get_all_states()}")


if __name__ == "__main__":
    asyncio.run(example_rate_limiting())
