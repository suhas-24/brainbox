"""
Advanced Rate Limiting System for AI Forge

Provides multiple rate limiting strategies including token bucket, sliding window,
fixed window, and user-based limits with Redis and memory backends.
"""

import asyncio
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from enum import Enum
import threading
import math

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .logging import get_logger


logger = get_logger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    max_requests: int  # Maximum requests allowed
    window_size: int   # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    burst_size: Optional[int] = None  # Allow burst above normal rate


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None
    current_usage: int = 0


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    async def check_limit(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check if request is within rate limits."""
        pass
    
    @abstractmethod
    async def reset_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        pass
    
    @abstractmethod
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage statistics for a key."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter implementation."""
    
    def __init__(self, backend: str = "memory"):
        self.backend = backend
        self.logger = get_logger(__name__)
        
        if backend == "memory":
            self._buckets: Dict[str, Dict[str, Any]] = {}
            self._lock = threading.RLock()
        elif backend == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis not available for rate limiting")
            self._redis: Optional[aioredis.Redis] = None
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = aioredis.from_url("redis://localhost:6379")
        return self._redis
    
    async def _get_bucket_memory(self, key: str, config: RateLimitConfig) -> Dict[str, Any]:
        """Get bucket state from memory."""
        with self._lock:
            now = time.time()
            
            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": config.max_requests,
                    "last_refill": now,
                    "max_tokens": config.max_requests
                }
            
            bucket = self._buckets[key]
            
            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_refill"]
            refill_rate = config.max_requests / config.window_size
            tokens_to_add = elapsed * refill_rate
            
            bucket["tokens"] = min(
                bucket["max_tokens"],
                bucket["tokens"] + tokens_to_add
            )
            bucket["last_refill"] = now
            
            return bucket
    
    async def _get_bucket_redis(self, key: str, config: RateLimitConfig) -> Dict[str, Any]:
        """Get bucket state from Redis."""
        redis = await self._get_redis()
        bucket_key = f"rate_limit:bucket:{key}"
        
        # Use Redis pipeline for atomic operations
        async with redis.pipeline() as pipe:
            await pipe.hgetall(bucket_key)
            await pipe.expire(bucket_key, config.window_size * 2)
            results = await pipe.execute()
        
        bucket_data = results[0]
        now = time.time()
        
        if not bucket_data:
            # Initialize new bucket
            bucket = {
                "tokens": config.max_requests,
                "last_refill": now,
                "max_tokens": config.max_requests
            }
        else:
            bucket = {
                "tokens": float(bucket_data.get(b"tokens", config.max_requests)),
                "last_refill": float(bucket_data.get(b"last_refill", now)),
                "max_tokens": int(bucket_data.get(b"max_tokens", config.max_requests))
            }
        
        # Refill tokens
        elapsed = now - bucket["last_refill"]
        refill_rate = config.max_requests / config.window_size
        tokens_to_add = elapsed * refill_rate
        
        bucket["tokens"] = min(
            bucket["max_tokens"],
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = now
        
        # Save back to Redis
        await redis.hmset(bucket_key, {
            "tokens": bucket["tokens"],
            "last_refill": bucket["last_refill"],
            "max_tokens": bucket["max_tokens"]
        })
        
        return bucket
    
    async def check_limit(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        try:
            if self.backend == "memory":
                bucket = await self._get_bucket_memory(key, config)
            else:
                bucket = await self._get_bucket_redis(key, config)
            
            now = time.time()
            
            if bucket["tokens"] >= 1:
                # Allow request and consume token
                bucket["tokens"] -= 1
                
                if self.backend == "memory":
                    with self._lock:
                        self._buckets[key] = bucket
                elif self.backend == "redis":
                    redis = await self._get_redis()
                    bucket_key = f"rate_limit:bucket:{key}"
                    await redis.hmset(bucket_key, {"tokens": bucket["tokens"]})
                
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_time=int(now + config.window_size),
                    current_usage=config.max_requests - int(bucket["tokens"])
                )
            else:
                # Rate limit exceeded
                retry_after = int(math.ceil(1.0 / (config.max_requests / config.window_size)))
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=int(now + config.window_size),
                    retry_after=retry_after,
                    current_usage=config.max_requests
                )
                
        except Exception as e:
            self.logger.error(f"Rate limit check failed for {key}: {e}")
            # Fail open - allow request on error
            return RateLimitResult(
                allowed=True,
                remaining=config.max_requests,
                reset_time=int(time.time() + config.window_size)
            )
    
    async def reset_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        try:
            if self.backend == "memory":
                with self._lock:
                    self._buckets.pop(key, None)
            else:
                redis = await self._get_redis()
                bucket_key = f"rate_limit:bucket:{key}"
                await redis.delete(bucket_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit reset failed for {key}: {e}")
            return False
    
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage statistics."""
        try:
            if self.backend == "memory":
                with self._lock:
                    bucket = self._buckets.get(key, {})
            else:
                redis = await self._get_redis()
                bucket_key = f"rate_limit:bucket:{key}"
                bucket_data = await redis.hgetall(bucket_key)
                bucket = {
                    "tokens": float(bucket_data.get(b"tokens", 0)),
                    "max_tokens": int(bucket_data.get(b"max_tokens", 0))
                } if bucket_data else {}
            
            return {
                "tokens_remaining": bucket.get("tokens", 0),
                "max_tokens": bucket.get("max_tokens", 0),
                "usage_percentage": (
                    (bucket.get("max_tokens", 0) - bucket.get("tokens", 0)) / 
                    bucket.get("max_tokens", 1) * 100
                ) if bucket else 0
            }
            
        except Exception as e:
            self.logger.error(f"Get usage failed for {key}: {e}")
            return {}


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter implementation."""
    
    def __init__(self, backend: str = "memory"):
        self.backend = backend
        self.logger = get_logger(__name__)
        
        if backend == "memory":
            self._windows: Dict[str, List[float]] = {}
            self._lock = threading.RLock()
        elif backend == "redis":
            if not REDIS_AVAILABLE:
                raise ImportError("Redis not available for rate limiting")
            self._redis: Optional[aioredis.Redis] = None
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = aioredis.from_url("redis://localhost:6379")
        return self._redis
    
    async def _cleanup_window_memory(self, key: str, window_start: float):
        """Clean up old entries from memory window."""
        with self._lock:
            if key in self._windows:
                self._windows[key] = [
                    timestamp for timestamp in self._windows[key]
                    if timestamp >= window_start
                ]
    
    async def check_limit(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        try:
            now = time.time()
            window_start = now - config.window_size
            
            if self.backend == "memory":
                await self._cleanup_window_memory(key, window_start)
                
                with self._lock:
                    if key not in self._windows:
                        self._windows[key] = []
                    
                    current_requests = len(self._windows[key])
                    
                    if current_requests < config.max_requests:
                        # Allow request
                        self._windows[key].append(now)
                        
                        return RateLimitResult(
                            allowed=True,
                            remaining=config.max_requests - current_requests - 1,
                            reset_time=int(now + config.window_size),
                            current_usage=current_requests + 1
                        )
                    else:
                        # Rate limit exceeded
                        oldest_request = min(self._windows[key]) if self._windows[key] else now
                        retry_after = int(oldest_request + config.window_size - now)
                        
                        return RateLimitResult(
                            allowed=False,
                            remaining=0,
                            reset_time=int(oldest_request + config.window_size),
                            retry_after=max(1, retry_after),
                            current_usage=current_requests
                        )
            
            else:  # Redis backend
                redis = await self._get_redis()
                window_key = f"rate_limit:window:{key}"
                
                # Remove expired entries and count current requests
                async with redis.pipeline() as pipe:
                    await pipe.zremrangebyscore(window_key, 0, window_start)
                    await pipe.zcard(window_key)
                    await pipe.expire(window_key, config.window_size)
                    results = await pipe.execute()
                
                current_requests = results[1]
                
                if current_requests < config.max_requests:
                    # Allow request
                    await redis.zadd(window_key, {str(now): now})
                    
                    return RateLimitResult(
                        allowed=True,
                        remaining=config.max_requests - current_requests - 1,
                        reset_time=int(now + config.window_size),
                        current_usage=current_requests + 1
                    )
                else:
                    # Rate limit exceeded
                    oldest_timestamp = await redis.zrange(window_key, 0, 0, withscores=True)
                    oldest_time = oldest_timestamp[0][1] if oldest_timestamp else now
                    retry_after = int(oldest_time + config.window_size - now)
                    
                    return RateLimitResult(
                        allowed=False,
                        remaining=0,
                        reset_time=int(oldest_time + config.window_size),
                        retry_after=max(1, retry_after),
                        current_usage=current_requests
                    )
                    
        except Exception as e:
            self.logger.error(f"Sliding window check failed for {key}: {e}")
            # Fail open
            return RateLimitResult(
                allowed=True,
                remaining=config.max_requests,
                reset_time=int(time.time() + config.window_size)
            )
    
    async def reset_limit(self, key: str) -> bool:
        """Reset rate limit for a key."""
        try:
            if self.backend == "memory":
                with self._lock:
                    self._windows.pop(key, None)
            else:
                redis = await self._get_redis()
                window_key = f"rate_limit:window:{key}"
                await redis.delete(window_key)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit reset failed for {key}: {e}")
            return False
    
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get current usage statistics."""
        try:
            now = time.time()
            window_start = now - 3600  # 1 hour window for stats
            
            if self.backend == "memory":
                with self._lock:
                    requests = self._windows.get(key, [])
                    recent_requests = [r for r in requests if r >= window_start]
            else:
                redis = await self._get_redis()
                window_key = f"rate_limit:window:{key}"
                recent_requests = await redis.zrangebyscore(window_key, window_start, now)
            
            return {
                "recent_requests": len(recent_requests),
                "window_start": window_start,
                "current_time": now
            }
            
        except Exception as e:
            self.logger.error(f"Get usage failed for {key}: {e}")
            return {}


class RateLimitManager:
    """Unified rate limit manager with multiple strategies."""
    
    def __init__(self, backend: str = "memory"):
        self.backend = backend
        self.logger = get_logger(__name__)
        
        # Initialize limiters
        self._limiters = {
            RateLimitStrategy.TOKEN_BUCKET: TokenBucketLimiter(backend),
            RateLimitStrategy.SLIDING_WINDOW: SlidingWindowLimiter(backend),
            # Can add more strategies here
        }
        
        # Default configurations for different use cases
        self.default_configs = {
            "api": RateLimitConfig(
                max_requests=100,
                window_size=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET
            ),
            "llm": RateLimitConfig(
                max_requests=10,
                window_size=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            ),
            "strict": RateLimitConfig(
                max_requests=5,
                window_size=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            )
        }
    
    async def check_limit(
        self,
        key: str,
        config: Optional[RateLimitConfig] = None,
        config_name: Optional[str] = None
    ) -> RateLimitResult:
        """Check rate limit for a key."""
        try:
            # Get configuration
            if config is None:
                if config_name and config_name in self.default_configs:
                    config = self.default_configs[config_name]
                else:
                    config = self.default_configs["api"]
            
            # Get appropriate limiter
            limiter = self._limiters.get(config.strategy)
            if not limiter:
                raise ValueError(f"Unknown rate limit strategy: {config.strategy}")
            
            return await limiter.check_limit(key, config)
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            # Fail open
            return RateLimitResult(
                allowed=True,
                remaining=100,
                reset_time=int(time.time() + 60)
            )
    
    async def reset_limit(self, key: str, strategy: RateLimitStrategy = None) -> bool:
        """Reset rate limit for a key."""
        try:
            if strategy:
                limiter = self._limiters.get(strategy)
                if limiter:
                    return await limiter.reset_limit(key)
            else:
                # Reset for all strategies
                results = []
                for limiter in self._limiters.values():
                    results.append(await limiter.reset_limit(key))
                return any(results)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Rate limit reset failed: {e}")
            return False
    
    async def get_usage(self, key: str) -> Dict[str, Any]:
        """Get usage statistics for a key across all strategies."""
        usage = {}
        
        for strategy, limiter in self._limiters.items():
            try:
                usage[strategy.value] = await limiter.get_usage(key)
            except Exception as e:
                self.logger.error(f"Failed to get usage for {strategy}: {e}")
                usage[strategy.value] = {}
        
        return usage
    
    def add_config(self, name: str, config: RateLimitConfig):
        """Add a named rate limit configuration."""
        self.default_configs[name] = config
    
    def rate_limit(
        self,
        config: Optional[RateLimitConfig] = None,
        config_name: Optional[str] = None,
        key_func: Optional[callable] = None
    ):
        """Decorator for rate limiting functions."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate rate limit key
                if key_func:
                    limit_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    limit_key = f"{func.__name__}:default"
                
                # Check rate limit
                result = await self.check_limit(limit_key, config, config_name)
                
                if not result.allowed:
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={
                            "X-RateLimit-Remaining": str(result.remaining),
                            "X-RateLimit-Reset": str(result.reset_time),
                            "Retry-After": str(result.retry_after or 60)
                        }
                    )
                
                # Execute function
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator


# Global rate limiter instance
_rate_limiter: Optional[RateLimitManager] = None


def get_rate_limiter(backend: str = "memory") -> RateLimitManager:
    """Get global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimitManager(backend)
    
    return _rate_limiter


async def setup_rate_limiting(backend: str = "memory") -> RateLimitManager:
    """Setup global rate limiting."""
    global _rate_limiter
    _rate_limiter = RateLimitManager(backend)
    return _rate_limiter
