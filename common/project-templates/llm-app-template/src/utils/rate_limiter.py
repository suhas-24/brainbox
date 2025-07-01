"""
Advanced Rate Limiting System for AI Forge

Provides token bucket, sliding window, and adaptive rate limiting
for API calls, user requests, and resource management.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import threading
from collections import defaultdict, deque

from .logger import get_logger


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed."""
        pass
    
    @abstractmethod
    async def reset(self, key: str) -> bool:
        """Reset rate limit for key."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter with burst capacity."""
    
    def __init__(
        self,
        rate: float,
        burst: int,
        window: int = 60
    ):
        self.rate = rate  # tokens per second
        self.burst = burst  # maximum burst size
        self.window = window  # window size in seconds
        
        self._buckets: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"tokens": burst, "last_update": time.time()}
        )
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)
    
    def _refill_bucket(self, bucket: Dict[str, float]) -> None:
        """Refill bucket with tokens based on elapsed time."""
        now = time.time()
        elapsed = now - bucket["last_update"]
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.rate
        bucket["tokens"] = min(self.burst, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now
    
    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed under token bucket."""
        with self._lock:
            bucket = self._buckets[key]
            self._refill_bucket(bucket)
            
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_time=time.time() + (self.burst - bucket["tokens"]) / self.rate
                )
            else:
                retry_after = (1 - bucket["tokens"]) / self.rate
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=time.time() + retry_after,
                    retry_after=retry_after
                )
    
    async def reset(self, key: str) -> bool:
        """Reset bucket for key."""
        with self._lock:
            if key in self._buckets:
                self._buckets[key] = {
                    "tokens": self.burst,
                    "last_update": time.time()
                }
                return True
            return False


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter."""
    
    def __init__(
        self,
        limit: int,
        window: int = 60
    ):
        self.limit = limit
        self.window = window
        
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)
    
    def _clean_window(self, window: deque) -> None:
        """Remove expired entries from window."""
        now = time.time()
        cutoff = now - self.window
        
        while window and window[0] < cutoff:
            window.popleft()
    
    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed under sliding window."""
        with self._lock:
            window = self._windows[key]
            self._clean_window(window)
            
            current_count = len(window)
            
            if current_count < self.limit:
                window.append(time.time())
                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - current_count - 1,
                    reset_time=window[0] + self.window if window else time.time()
                )
            else:
                # Calculate when the oldest request expires
                oldest_request = window[0] if window else time.time()
                retry_after = oldest_request + self.window - time.time()
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=oldest_request + self.window,
                    retry_after=max(0, retry_after)
                )
    
    async def reset(self, key: str) -> bool:
        """Reset window for key."""
        with self._lock:
            if key in self._windows:
                self._windows[key].clear()
                return True
            return False


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts based on system load."""
    
    def __init__(
        self,
        base_limit: int,
        window: int = 60,
        load_threshold: float = 0.8,
        min_limit: int = 10
    ):
        self.base_limit = base_limit
        self.window = window
        self.load_threshold = load_threshold
        self.min_limit = min_limit
        
        self._current_limit = base_limit
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._system_load = 0.0
        self._last_adjustment = time.time()
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)
    
    def update_system_load(self, load: float) -> None:
        """Update system load metric (0.0 to 1.0)."""
        self._system_load = max(0.0, min(1.0, load))
        self._adjust_limit()
    
    def _adjust_limit(self) -> None:
        """Adjust rate limit based on system load."""
        now = time.time()
        if now - self._last_adjustment < 5:  # Adjust at most every 5 seconds
            return
        
        if self._system_load > self.load_threshold:
            # Reduce limit when system is under load
            new_limit = max(
                self.min_limit,
                int(self._current_limit * (2 - self._system_load))
            )
        else:
            # Increase limit when system has capacity
            new_limit = min(
                self.base_limit,
                int(self._current_limit * (1 + (1 - self._system_load) * 0.1))
            )
        
        if new_limit != self._current_limit:
            self.logger.info(
                f"Adjusted rate limit: {self._current_limit} -> {new_limit} "
                f"(load: {self._system_load:.2f})"
            )
            self._current_limit = new_limit
        
        self._last_adjustment = now
    
    async def is_allowed(self, key: str) -> RateLimitResult:
        """Check if request is allowed under adaptive limiting."""
        with self._lock:
            self._adjust_limit()
            
            window = self._windows[key]
            now = time.time()
            cutoff = now - self.window
            
            # Clean expired entries
            while window and window[0] < cutoff:
                window.popleft()
            
            current_count = len(window)
            
            if current_count < self._current_limit:
                window.append(now)
                return RateLimitResult(
                    allowed=True,
                    remaining=self._current_limit - current_count - 1,
                    reset_time=window[0] + self.window if window else now
                )
            else:
                oldest_request = window[0] if window else now
                retry_after = oldest_request + self.window - now
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_time=oldest_request + self.window,
                    retry_after=max(0, retry_after)
                )
    
    async def reset(self, key: str) -> bool:
        """Reset window for key."""
        with self._lock:
            if key in self._windows:
                self._windows[key].clear()
                return True
            return False


class HierarchicalRateLimiter:
    """Multi-level rate limiter (user, IP, global)."""
    
    def __init__(
        self,
        user_limiter: RateLimiter,
        ip_limiter: RateLimiter,
        global_limiter: Optional[RateLimiter] = None
    ):
        self.user_limiter = user_limiter
        self.ip_limiter = ip_limiter
        self.global_limiter = global_limiter
        self.logger = get_logger(__name__)
    
    async def is_allowed(
        self,
        user_id: str,
        ip_address: str,
        global_key: str = "global"
    ) -> RateLimitResult:
        """Check all rate limit levels."""
        
        # Check global limit first
        if self.global_limiter:
            global_result = await self.global_limiter.is_allowed(global_key)
            if not global_result.allowed:
                self.logger.warning(f"Global rate limit exceeded")
                return global_result
        
        # Check IP limit
        ip_result = await self.ip_limiter.is_allowed(ip_address)
        if not ip_result.allowed:
            self.logger.warning(f"IP rate limit exceeded: {ip_address}")
            return ip_result
        
        # Check user limit
        user_result = await self.user_limiter.is_allowed(user_id)
        if not user_result.allowed:
            self.logger.info(f"User rate limit exceeded: {user_id}")
            return user_result
        
        # All checks passed
        return user_result
    
    async def reset_user(self, user_id: str) -> bool:
        """Reset rate limit for specific user."""
        return await self.user_limiter.reset(user_id)
    
    async def reset_ip(self, ip_address: str) -> bool:
        """Reset rate limit for specific IP."""
        return await self.ip_limiter.reset(ip_address)


# Rate limiting decorators
def rate_limit(limiter: RateLimiter, key_func: callable = None):
    """Decorator to apply rate limiting to functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate key for rate limiting
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"
            
            # Check rate limit
            result = await limiter.is_allowed(key)
            
            if not result.allowed:
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {result.retry_after:.2f}s",
                    retry_after=result.retry_after
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class RateLimitError(Exception):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


# Utility functions
def create_user_rate_limiter(
    requests_per_minute: int = 60,
    burst_size: int = 10
) -> TokenBucketLimiter:
    """Create a rate limiter for user requests."""
    return TokenBucketLimiter(
        rate=requests_per_minute / 60.0,  # Convert to per-second
        burst=burst_size,
        window=60
    )


def create_ip_rate_limiter(
    requests_per_hour: int = 1000
) -> SlidingWindowLimiter:
    """Create a rate limiter for IP addresses."""
    return SlidingWindowLimiter(
        limit=requests_per_hour,
        window=3600  # 1 hour
    )


def create_adaptive_limiter(
    base_requests_per_minute: int = 100
) -> AdaptiveRateLimiter:
    """Create an adaptive rate limiter."""
    return AdaptiveRateLimiter(
        base_limit=base_requests_per_minute,
        window=60
    )
