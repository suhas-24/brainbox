"""
Advanced Caching System for AI Forge

Provides multiple caching backends (memory, Redis, file) with 
TTL, compression, and intelligent cache management.
"""

import asyncio
import hashlib
import json
import pickle
import time
import weakref
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import threading
from dataclasses import dataclass
from functools import wraps

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

from .logger import get_logger


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    hit_count: int = 0
    size_bytes: int = 0


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self.logger = get_logger(__name__)
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _get_size_bytes(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        if entry.expires_at is None:
            return False
        return time.time() > entry.expires_at
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and current_time > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        while len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._evictions += 1
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            self._evict_expired()
            
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            if self._is_expired(entry):
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._misses += 1
                return None
            
            # Update statistics and access order
            entry.hit_count += 1
            self._hits += 1
            self._update_access_order(key)
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        with self._lock:
            self._evict_expired()
            self._evict_lru()
            
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl if ttl > 0 else None
            
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
                size_bytes=self._get_size_bytes(value)
            )
            
            self._cache[key] = entry
            self._update_access_order(key)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return await self.get(key) is not None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                "backend": "memory",
                "entries": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "evictions": self._evictions,
                "total_size_bytes": total_size,
                "average_entry_size": total_size / len(self._cache) if self._cache else 0
            }


class RedisCache(CacheBackend):
    """Redis-based cache backend."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "aiforge:",
        default_ttl: int = 3600,
        compression: bool = True
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not available. Install with: pip install redis")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.compression = compression and LZ4_AVAILABLE
        self.logger = get_logger(__name__)
        
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get Redis connection."""
        if self._redis is None or not self._connected:
            try:
                self._redis = aioredis.from_url(self.redis_url)
                await self._redis.ping()
                self._connected = True
                self.logger.info("Connected to Redis cache")
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._redis
    
    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        data = pickle.dumps(value)
        
        if self.compression:
            try:
                data = lz4.frame.compress(data)
            except Exception as e:
                self.logger.warning(f"Compression failed: {e}")
        
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.compression:
            try:
                data = lz4.frame.decompress(data)
            except Exception as e:
                self.logger.warning(f"Decompression failed: {e}")
        
        return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            redis = await self._get_redis()
            data = await redis.get(self._make_key(key))
            
            if data is None:
                return None
            
            return self._deserialize(data)
        
        except Exception as e:
            self.logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            redis = await self._get_redis()
            data = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            if ttl > 0:
                await redis.setex(self._make_key(key), ttl, data)
            else:
                await redis.set(self._make_key(key), data)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis = await self._get_redis()
            result = await redis.delete(self._make_key(key))
            return result > 0
        
        except Exception as e:
            self.logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        try:
            redis = await self._get_redis()
            keys = await redis.keys(f"{self.key_prefix}*")
            
            if keys:
                await redis.delete(*keys)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            redis = await self._get_redis()
            result = await redis.exists(self._make_key(key))
            return result > 0
        
        except Exception as e:
            self.logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            redis = await self._get_redis()
            info = await redis.info()
            
            return {
                "backend": "redis",
                "connected": self._connected,
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_connections": info.get("total_connections_received", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "compression": self.compression
            }
        
        except Exception as e:
            self.logger.error(f"Redis stats error: {e}")
            return {"backend": "redis", "connected": False, "error": str(e)}


class FileCache(CacheBackend):
    """File-based cache backend."""
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        default_ttl: int = 3600,
        max_files: int = 10000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_files = max_files
        self.logger = get_logger(__name__)
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, file_path: Path, ttl: int) -> bool:
        """Check if cached file has expired."""
        if not file_path.exists():
            return True
        
        if ttl <= 0:
            return False
        
        age = time.time() - file_path.stat().st_mtime
        return age > ttl
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        try:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return None
            
            # Check if expired
            if self._is_expired(file_path, self.default_ttl):
                file_path.unlink(missing_ok=True)
                return None
            
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        
        except Exception as e:
            self.logger.error(f"File cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache."""
        try:
            file_path = self._get_file_path(key)
            
            # Clean up old files if approaching limit
            await self._cleanup_old_files()
            
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            return True
        
        except Exception as e:
            self.logger.error(f"File cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from file cache."""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        
        except Exception as e:
            self.logger.error(f"File cache delete error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache files."""
        try:
            for file_path in self.cache_dir.glob("*.cache"):
                file_path.unlink()
            return True
        
        except Exception as e:
            self.logger.error(f"File cache clear error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in file cache."""
        file_path = self._get_file_path(key)
        return file_path.exists() and not self._is_expired(file_path, self.default_ttl)
    
    async def _cleanup_old_files(self) -> None:
        """Clean up old cache files."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            
            if len(cache_files) >= self.max_files:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda p: p.stat().st_mtime)
                files_to_remove = cache_files[:len(cache_files) - self.max_files + 100]
                
                for file_path in files_to_remove:
                    file_path.unlink(missing_ok=True)
        
        except Exception as e:
            self.logger.warning(f"Cache cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get file cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "backend": "file",
                "cache_dir": str(self.cache_dir),
                "files": len(cache_files),
                "max_files": self.max_files,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2)
            }
        
        except Exception as e:
            self.logger.error(f"File cache stats error: {e}")
            return {"backend": "file", "error": str(e)}


class CacheManager:
    """Unified cache manager with multiple backends."""
    
    def __init__(
        self,
        primary_backend: CacheBackend,
        fallback_backend: Optional[CacheBackend] = None
    ):
        self.primary = primary_backend
        self.fallback = fallback_backend
        self.logger = get_logger(__name__)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback."""
        try:
            # Try primary backend first
            value = await self.primary.get(key)
            if value is not None:
                return value
            
            # Try fallback backend
            if self.fallback:
                value = await self.fallback.get(key)
                if value is not None:
                    # Store in primary for next time
                    await self.primary.set(key, value)
                    return value
            
            return None
        
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            # Set in primary backend
            primary_success = await self.primary.set(key, value, ttl)
            
            # Optionally set in fallback
            fallback_success = True
            if self.fallback:
                fallback_success = await self.fallback.set(key, value, ttl)
            
            return primary_success or fallback_success
        
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache backends."""
        primary_success = await self.primary.delete(key)
        fallback_success = await self.fallback.delete(key) if self.fallback else True
        return primary_success or fallback_success
    
    async def clear(self) -> bool:
        """Clear all cache backends."""
        primary_success = await self.primary.clear()
        fallback_success = await self.fallback.clear() if self.fallback else True
        return primary_success and fallback_success
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all backends."""
        stats = {
            "primary": await self.primary.get_stats()
        }
        
        if self.fallback:
            stats["fallback"] = await self.fallback.get_stats()
        
        return stats


def cache_result(ttl: int = 3600, key_func: Optional[callable] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            if hasattr(wrapper, '_cache'):
                cached_result = await wrapper._cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if hasattr(wrapper, '_cache'):
                await wrapper._cache.set(cache_key, result, ttl)
            
            return result
        
        # Allow setting cache backend
        wrapper.set_cache = lambda cache: setattr(wrapper, '_cache', cache)
        
        return wrapper
    
    return decorator


# Default cache instances
_default_memory_cache = None
_default_cache_manager = None


def get_default_cache() -> CacheManager:
    """Get default cache manager instance."""
    global _default_memory_cache, _default_cache_manager
    
    if _default_cache_manager is None:
        _default_memory_cache = MemoryCache()
        _default_cache_manager = CacheManager(_default_memory_cache)
    
    return _default_cache_manager
