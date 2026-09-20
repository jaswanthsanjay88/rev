"""
Thread-safe LRU state prefix KV-cache manager for rev.
Caches precomputed key/value tensors of document prefixes to enable sub-5ms decision queries.
"""

import collections
import hashlib
import threading
import time
from typing import Any, Optional


class CachedStateEntry:
    def __init__(self, key: str, past_key_values: Any, state_len: int, token_ids: list[int] = None):
        self.key = key
        self.past_key_values = past_key_values
        self.state_len = state_len
        self.token_ids = token_ids or []
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.hits = 0

    def touch(self):
        self.last_accessed = time.time()
        self.hits += 1


class StateKVCacheManager:
    """
    LRU cache for prefix key-value states.
    Thread-safe and eviction-ready.
    """
    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._cache: collections.OrderedDict[str, CachedStateEntry] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    @staticmethod
    def hash_state(state_text: str) -> str:
        return hashlib.sha256(state_text.strip().encode("utf-8")).hexdigest()

    def get(self, state_text: str) -> Optional[CachedStateEntry]:
        key = self.hash_state(state_text)
        return self.get_by_hash(key)

    def get_by_hash(self, key: str) -> Optional[CachedStateEntry]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.touch()
                self._cache.move_to_end(key)
                self._total_hits += 1
                return entry
            else:
                self._total_misses += 1
                return None

    def put(self, state_text: str, past_key_values: Any, state_len: int, token_ids: list[int] = None) -> str:
        key = self.hash_state(state_text)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.past_key_values = past_key_values
                entry.state_len = state_len
                entry.token_ids = token_ids or []
                entry.touch()
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_entries:
                    self._cache.popitem(last=False)
                entry = CachedStateEntry(key, past_key_values, state_len, token_ids or [])
                self._cache[key] = entry
        return key

    def evict(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._total_hits = 0
            self._total_misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self._total_hits + self._total_misses
            hit_rate = round(self._total_hits / total, 4) if total > 0 else 0.0
            entries_info = [
                {
                    "hash": k,
                    "state_len": entry.state_len,
                    "hits": entry.hits,
                    "age_seconds": round(time.time() - entry.created_at, 1),
                }
                for k, entry in self._cache.items()
            ]
            return {
                "total_entries": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._total_hits,
                "misses": self._total_misses,
                "hit_rate": hit_rate,
                "entries": entries_info,
            }
