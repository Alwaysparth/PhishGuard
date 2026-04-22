"""
╔══════════════════════════════════════════════════════════════╗
║         PhishGuard · app/core/cache.py                       ║
║         In-memory URL result cache (TTL + LRU eviction)      ║
╚══════════════════════════════════════════════════════════════╝
"""

import time
import threading
from collections import OrderedDict
from typing import Any, Optional


class URLCache:
    """
    Thread-safe, TTL-aware LRU cache for storing URL scan results.

    Why not Redis?
    ──────────────
    For a hackathon / prototype deployment a lightweight in-process
    cache avoids an extra service dependency while still giving us
    meaningful performance gains.  The design is intentionally drop-in
    replaceable with a Redis-backed cache later.

    Eviction policy
    ───────────────
    • Entries expire after `ttl_seconds` of inactivity.
    • When `max_size` is reached the oldest entry (LRU) is evicted first.

    Usage
    ─────
        cache = URLCache(ttl_seconds=600, max_size=1000)
        cache.set("https://example.com", result_dict)
        hit = cache.get("https://example.com")   # None on miss / expiry
    """

    def __init__(self, ttl_seconds: int = 600, max_size: int = 1000) -> None:
        self._ttl = ttl_seconds
        self._max = max_size
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.RLock()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return cached value or None if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._store[key]
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """Insert or refresh a cache entry."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = {"value": value, "ts": time.monotonic()}
            self._evict_if_needed()

    def delete(self, key: str) -> None:
        """Explicitly remove a key (e.g. after a domain is blacklisted)."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Wipe the entire cache."""
        with self._lock:
            self._store.clear()

    # ── Stats (exposed via /health endpoint) ──────────────────────────────────

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def max_size(self) -> int:
        return self._max

    @property
    def ttl(self) -> int:
        return self._ttl

    def stats(self) -> dict:
        with self._lock:
            valid = sum(1 for e in self._store.values() if not self._is_expired(e))
            return {
                "entries_total": len(self._store),
                "entries_valid": valid,
                "entries_expired": len(self._store) - valid,
                "max_size": self._max,
                "ttl_seconds": self._ttl,
            }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_expired(self, entry: dict) -> bool:
        return (time.monotonic() - entry["ts"]) > self._ttl

    def _evict_if_needed(self) -> None:
        """Remove the LRU entry when the store exceeds max_size."""
        while len(self._store) > self._max:
            self._store.popitem(last=False)
