"""Caching service for performance optimization"""
import json
import hashlib
import time
from typing import Any, Optional, Callable, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import OrderedDict
import threading


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: float
    hit_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache with TTL support"""

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600):
        """Initialize LRU cache

        Args:
            max_size: Maximum number of entries
            default_ttl_seconds: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def _compute_size(self, value: Any) -> int:
        """Estimate size of value in bytes

        Args:
            value: Value to measure

        Returns:
            Estimated size in bytes
        """
        try:
            # Serialize to JSON and measure
            return len(json.dumps(value, default=str).encode())
        except:
            # Fallback for non-serializable objects
            return len(str(value).encode())

    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            key, entry = self.cache.popitem(last=False)  # Remove oldest
            self.evictions += 1

    def _clean_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.cache.items():
            if current_time > entry.expires_at:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]
            self.expirations += 1

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            # Clean expired entries periodically
            if len(self.cache) > 0 and self.hits % 100 == 0:
                self._clean_expired()

            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check expiration
            if time.time() > entry.expires_at:
                del self.cache[key]
                self.expirations += 1
                self.misses += 1
                return None

            # Update access metadata
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self.hits += 1

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ):
        """Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
            current_time = time.time()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                expires_at=current_time + ttl,
                last_accessed=current_time,
                size_bytes=self._compute_size(value)
            )

            # Evict if at capacity and key doesn't exist
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_lru()

            # Add to cache
            self.cache[key] = entry
            self.cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete key from cache

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            total_size = sum(entry.size_bytes for entry in self.cache.values())

            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024)
            }


class CacheService:
    """High-level caching service with multiple cache layers"""

    def __init__(
        self,
        embedding_cache_size: int = 10000,
        prompt_cache_size: int = 1000,
        query_cache_size: int = 5000,
        default_ttl_seconds: int = 3600
    ):
        """Initialize cache service

        Args:
            embedding_cache_size: Size of embedding cache
            prompt_cache_size: Size of prompt cache
            query_cache_size: Size of query result cache
            default_ttl_seconds: Default TTL for all caches
        """
        # Separate caches for different purposes
        self.embedding_cache = LRUCache(
            max_size=embedding_cache_size,
            default_ttl_seconds=86400  # 24 hours for embeddings
        )

        self.prompt_cache = LRUCache(
            max_size=prompt_cache_size,
            default_ttl_seconds=3600  # 1 hour for prompts
        )

        self.query_cache = LRUCache(
            max_size=query_cache_size,
            default_ttl_seconds=1800  # 30 minutes for queries
        )

        self.general_cache = LRUCache(
            max_size=1000,
            default_ttl_seconds=default_ttl_seconds
        )

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments

        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key
        """
        # Create deterministic string from arguments
        key_parts = [prefix]

        for arg in args:
            key_parts.append(str(arg))

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_string = ":".join(key_parts)

        # Hash for consistent length
        return hashlib.md5(key_string.encode()).hexdigest()

    def cache_embedding(
        self,
        text: str,
        embedding: list,
        model: str = "text-embedding-ada-002"
    ):
        """Cache an embedding

        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model: Model used for embedding
        """
        key = self._generate_key("embedding", model, text)
        self.embedding_cache.set(key, embedding)

    def get_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> Optional[list]:
        """Get cached embedding

        Args:
            text: Text to get embedding for
            model: Model used for embedding

        Returns:
            Cached embedding or None
        """
        key = self._generate_key("embedding", model, text)
        return self.embedding_cache.get(key)

    def cache_prompt_result(
        self,
        prompt_key: str,
        variables: Dict[str, Any],
        result: str,
        ttl_seconds: int = 3600
    ):
        """Cache a prompt result

        Args:
            prompt_key: Prompt template key
            variables: Template variables
            result: LLM result
            ttl_seconds: Time-to-live
        """
        # Create deterministic key from variables
        var_str = json.dumps(variables, sort_keys=True)
        key = self._generate_key("prompt", prompt_key, var_str)
        self.prompt_cache.set(key, result, ttl_seconds)

    def get_prompt_result(
        self,
        prompt_key: str,
        variables: Dict[str, Any]
    ) -> Optional[str]:
        """Get cached prompt result

        Args:
            prompt_key: Prompt template key
            variables: Template variables

        Returns:
            Cached result or None
        """
        var_str = json.dumps(variables, sort_keys=True)
        key = self._generate_key("prompt", prompt_key, var_str)
        return self.prompt_cache.get(key)

    def cache_query_result(
        self,
        query: str,
        user_id: Optional[str],
        result: Dict[str, Any],
        ttl_seconds: int = 1800
    ):
        """Cache a query result

        Args:
            query: User query
            user_id: User identifier
            result: Query result
            ttl_seconds: Time-to-live
        """
        key = self._generate_key("query", query, user_id or "anonymous")
        self.query_cache.set(key, result, ttl_seconds)

    def get_query_result(
        self,
        query: str,
        user_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result

        Args:
            query: User query
            user_id: User identifier

        Returns:
            Cached result or None
        """
        key = self._generate_key("query", query, user_id or "anonymous")
        return self.query_cache.get(key)

    def cache_value(
        self,
        category: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ):
        """Cache a general value

        Args:
            category: Value category
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live
        """
        cache_key = self._generate_key(category, key)
        self.general_cache.set(cache_key, value, ttl_seconds)

    def get_value(
        self,
        category: str,
        key: str
    ) -> Optional[Any]:
        """Get cached general value

        Args:
            category: Value category
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_key = self._generate_key(category, key)
        return self.general_cache.get(cache_key)

    def invalidate_category(self, category: str):
        """Invalidate all entries in a category

        Args:
            category: Category to invalidate
        """
        # This is a simplified implementation
        # In production, you might want to track categories more efficiently
        prefix = hashlib.md5(category.encode()).hexdigest()[:8]

        # Clear matching entries from all caches
        for cache in [self.embedding_cache, self.prompt_cache, self.query_cache, self.general_cache]:
            keys_to_delete = [
                key for key in cache.cache.keys()
                if key.startswith(prefix)
            ]
            for key in keys_to_delete:
                cache.delete(key)

    def clear_all(self):
        """Clear all caches"""
        self.embedding_cache.clear()
        self.prompt_cache.clear()
        self.query_cache.clear()
        self.general_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches

        Returns:
            Dictionary with cache statistics
        """
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "prompt_cache": self.prompt_cache.get_stats(),
            "query_cache": self.query_cache.get_stats(),
            "general_cache": self.general_cache.get_stats()
        }

    def cached(
        self,
        category: str,
        ttl_seconds: Optional[int] = None
    ) -> Callable:
        """Decorator for caching function results

        Args:
            category: Cache category
            ttl_seconds: Time-to-live

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key = self._generate_key(
                    category,
                    func.__name__,
                    *args,
                    **kwargs
                )

                # Try to get from cache
                cached_result = self.general_cache.get(key)
                if cached_result is not None:
                    return cached_result

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                self.general_cache.set(key, result, ttl_seconds)

                return result

            return wrapper
        return decorator


# Global cache service instance
cache_service = CacheService()
