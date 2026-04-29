# Paged disk KV cache for mlx-engine.
# Per-block slice design adapted from jundot/omlx (Apache 2.0).

import hashlib
import json
import logging
import math
import os
import struct
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    QuantizedKVCache,
    load_prompt_cache,
    save_prompt_cache,
)

logger = logging.getLogger(__name__)

BLOCK_SIZE = 2048
CACHE_DIR = Path.home() / ".cache" / "mlx-engine" / "kv_cache"
_MAX_CACHE_BYTES = int(os.environ.get("MLX_DISK_KV_CACHE_MAX_GB", "5")) * 1024**3
_INITIAL_HIT_COUNT = 1


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[kv-disk] failed to load {path.name}: {e}")
        return default


def _save_json(path: Path, data) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"[kv-disk] failed to write {path.name}: {e}")


def _block_hash(block_tokens: list, prev_hash: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(prev_hash)
    h.update(struct.pack(f"{len(block_tokens)}I", *block_tokens))
    return h.digest()


def compute_block_hashes(token_list: list, block_size: int = BLOCK_SIZE) -> list[bytes]:
    """Chained SHA-256 hashes for every full block in token_list."""
    hashes: list[bytes] = []
    prev = b""
    for i in range(len(token_list) // block_size):
        prev = _block_hash(token_list[i * block_size : (i + 1) * block_size], prev)
        hashes.append(prev)
    return hashes


def _state_kind(state) -> str:
    """Classify a cache layer's state without relying on class identity.

    Returns 'quantized', 'plain', 'arrays', or 'unknown'.
    Structure rules:
      - list                   -> ArraysCache (GDN state)
      - tuple[tuple, tuple]    -> QuantizedKVCache (packed key/value components)
      - tuple[mx.array, ...]   -> KVCache (raw key/value tensors)
    """
    if isinstance(state, list):
        return "arrays"
    if isinstance(state[0], (list, tuple)):
        return "quantized"
    if isinstance(state[0], mx.array):
        return "plain"
    return "unknown"


def _quant_format_mismatch(cache: list, kv_bits) -> bool:
    """Return True if the cache's quantization format doesn't match kv_bits."""
    if kv_bits is not None:
        return any(hasattr(c, "to_quantized") for c in cache)
    return any(_state_kind(c.state) == "quantized" for c in cache)


def _slice_cache(cache: list, start: int, end: int) -> list:
    """Return new cache objects with KV sliced to [start:end] and GDN full state."""
    result = []
    for layer in cache:
        state = layer.state
        kind = _state_kind(state)
        if kind == "quantized":
            ks, vs = state
            new_layer = QuantizedKVCache.__new__(QuantizedKVCache)
            new_layer.state = (
                tuple(t[..., start:end, :] for t in ks),
                tuple(t[..., start:end, :] for t in vs),
            )
            # Preserve all meta fields (group_size, bits, etc.); update offset only.
            new_layer.meta_state = (str(end - start),) + tuple(layer.meta_state[1:])
        elif kind == "plain" and not hasattr(layer, "meta_state"):
            # RotatingKVCache also has plain state but carries meta_state, so it falls through.
            ks, vs = state
            new_layer = KVCache.__new__(KVCache)
            new_layer.state = (ks[..., start:end, :], vs[..., start:end, :])
        elif kind == "arrays":
            # ArraysCache (GDN): not sequence-indexed, copy full state as-is.
            new_layer = ArraysCache.__new__(ArraysCache)
            new_layer.state = list(state)
        else:
            result.append(layer)
            continue
        result.append(new_layer)
    return result


def _concatenate_block_caches(block_caches: list) -> list:
    """Concatenate N per-block cache lists along the sequence axis.

    KV layers are concatenated; GDN (ArraysCache) state comes from the last
    block, which is the correct cumulative snapshot.
    """
    n_layers = len(block_caches[0])
    result = []
    for i in range(n_layers):
        first_state = block_caches[0][i].state
        kind = _state_kind(first_state)
        if kind == "quantized":
            all_ks = [bc[i].state[0] for bc in block_caches]
            all_vs = [bc[i].state[1] for bc in block_caches]
            n_comp = len(all_ks[0])
            concat_k = tuple(
                mx.concatenate([k[j] for k in all_ks], axis=2) for j in range(n_comp)
            )
            concat_v = tuple(
                mx.concatenate([v[j] for v in all_vs], axis=2) for j in range(n_comp)
            )
            new_layer = QuantizedKVCache.__new__(QuantizedKVCache)
            new_layer.state = (concat_k, concat_v)
            new_layer.meta_state = (
                str(sum(bc[i].offset for bc in block_caches)),
            ) + tuple(block_caches[0][i].meta_state[1:])
            result.append(new_layer)
        elif kind == "plain" and not hasattr(block_caches[0][i], "meta_state"):
            # RotatingKVCache also has plain state but carries meta_state, so it falls through.
            all_ks = [bc[i].state[0] for bc in block_caches]
            all_vs = [bc[i].state[1] for bc in block_caches]
            new_layer = KVCache.__new__(KVCache)
            new_layer.state = (
                mx.concatenate(all_ks, axis=2),
                mx.concatenate(all_vs, axis=2),
            )
            result.append(new_layer)
        else:
            # ArraysCache (GDN) and pass-through types: last block holds the
            # correct cumulative state.
            result.append(block_caches[-1][i])
    return result


class PagedDiskKVCache:
    """Paged disk KV cache with per-block safetensors files and chained hashing.

    Each file stores the KV slice for its BLOCK_SIZE-token window plus the full
    GDN (ArraysCache) state at that boundary. On load, KV slices are concatenated
    and the last block's GDN state is used for reconstruction.

    Eviction: score = hit_count * exp(-age / tau), where tau is a running average
    of inter-session gaps, measured once per process lifetime on the first
    find_and_load call and persisted in session_tau.json. Capped at
    MLX_DISK_KV_CACHE_MAX_GB (default 5 GB).
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        block_size: int = BLOCK_SIZE,
    ) -> None:
        self._cache_dir = cache_dir
        self._block_size = block_size
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._cache_dir / "manifest.json"
        self._manifest: dict = _load_json(self._manifest_path, {})
        self._tau_path = self._cache_dir / "session_tau.json"
        tau_data = _load_json(self._tau_path, {})
        self._tau: Optional[float] = tau_data.get("tau")
        self._tau_weight: int = tau_data.get("weight", 0)
        self._tau_updated = False

    def should_save_block(self, block_hash: bytes, model_path: str) -> bool:
        """Return True if this block is not already on disk for this model."""
        key = block_hash.hex()
        entry = self._manifest.get(key)
        if not entry or entry.get("model_path") != model_path:
            return True
        return not (self._cache_dir / f"{key}.safetensors").exists()

    def save_block(
        self,
        block_hash: bytes,
        cache: list,
        start: int,
        end: int,
        model_path: str,
    ) -> None:
        """Persist a per-block KV slice + GDN snapshot to disk."""
        key = block_hash.hex()
        path = self._cache_dir / f"{key}.safetensors"

        if key not in self._manifest and self._is_eviction_candidate(key, model_path):
            logger.info(
                f"[kv-disk] skipping block [{start}:{end}]: would be immediately evicted"
            )
            return

        try:
            sliced = _slice_cache(cache, start, end)
            save_prompt_cache(str(path), sliced, metadata={"model_path": model_path})
        except Exception as e:
            logger.warning(f"[kv-disk] save failed block=[{start}:{end}]: {e}")
            return
        self._manifest[key] = {
            "model_path": model_path,
            "file_size": path.stat().st_size,
            "last_used": time.time(),
            "hit_count": self._manifest.get(key, {}).get(
                "hit_count", _INITIAL_HIT_COUNT
            ),
        }
        self._evict_if_needed()
        self._save_manifest()
        logger.info(f"[kv-disk] saved block [{start}:{end}] → {key[:8]}…")

    def find_and_load(
        self,
        token_list: list,
        model_path: str,
        kv_bits=None,
    ) -> Optional[tuple]:
        """Walk the block hash chain and restore the longest cached prefix.

        Returns (reconstructed_cache, cached_token_count) on hit, None on miss.
        """
        now = time.time()
        self._maybe_update_tau(now)
        hashes = compute_block_hashes(token_list, self._block_size)
        matches: list[tuple[str, Path]] = []
        for h in hashes:
            key = h.hex()
            entry = self._manifest.get(key)
            if not entry or entry.get("model_path") != model_path:
                break
            path = self._cache_dir / f"{key}.safetensors"
            if not path.exists():
                logger.warning(f"[kv-disk] manifest entry {key[:8]} missing on disk")
                del self._manifest[key]
                self._save_manifest()
                break
            matches.append((key, path))

        if not matches:
            return None

        try:
            block_caches = [load_prompt_cache(str(p)) for _, p in matches]
        except Exception as e:
            logger.warning(f"[kv-disk] load failed: {e}")
            return None

        # Validate that all blocks have the same type per layer before concatenating.
        # A mismatch indicates stale files saved under a different quantization config.
        n_layers = len(block_caches[0])
        for i in range(n_layers):
            kind = _state_kind(block_caches[0][i].state)
            if any(_state_kind(bc[i].state) != kind for bc in block_caches[1:]):
                logger.warning("[kv-disk] stale blocks invalidated (mixed types)")
                for key, path in matches:
                    self._manifest.pop(key, None)
                    path.unlink(missing_ok=True)
                self._save_manifest()
                return None

        cache = _concatenate_block_caches(block_caches)

        # Skip blocks whose quantization format doesn't match the current config.
        # Blocks are kept on disk in case the config is reverted, but marked with
        # last_used=0 so they are evicted first when the cache fills up.
        if _quant_format_mismatch(cache, kv_bits):
            logger.warning(
                "[kv-disk] quantization config changed, skipping stale blocks"
            )
            for key, _ in matches:
                self._manifest[key]["last_used"] = 0
            self._save_manifest()
            return None

        cached_count = len(matches) * self._block_size
        keys = [key for key, _ in matches]
        for key in keys:
            self._manifest[key]["last_used"] = now
        self._increment_hit_counts(keys)
        self._save_manifest()
        logger.info(
            f"[kv-disk] hit: {cached_count}/{len(token_list)} tokens"
            f" ({len(matches)} blocks)"
        )
        return cache, cached_count

    def _increment_hit_counts(self, keys: list[str]) -> bool:
        """Positional decay: weight(i) = (n-i) / (n*(n+1)/2), sums to 1.0 per call."""
        n = len(keys)
        if n == 0:
            return False
        total_weight = n * (n + 1) / 2
        changed = False
        for i, key in enumerate(keys):
            entry = self._manifest.get(key)
            if entry is not None:
                entry["hit_count"] = (
                    entry.get("hit_count", _INITIAL_HIT_COUNT) + (n - i) / total_weight
                )
                changed = True
        return changed

    def record_lru_hit(self, token_list: list, cached_token_count: int) -> None:
        """Increment hit_count for LRU-served blocks. Does not update last_used."""
        n = cached_token_count // self._block_size
        if n == 0:
            return
        hashes = compute_block_hashes(token_list, self._block_size)
        keys = [h.hex() for h in hashes[:n]]
        if self._increment_hit_counts(keys):
            self._save_manifest()

    def _maybe_update_tau(self, now: float) -> None:
        """Measure the inter-session gap and update the stored tau (once per process)."""
        if self._tau_updated or not self._manifest:
            return
        self._tau_updated = True
        max_last = max(e.get("last_used", 0) for e in self._manifest.values())
        if max_last <= 0:
            return
        # Floor at 1h: technical guard for restarts that happen seconds after last use.
        gap = max(now - max_last, 3600.0)
        if self._tau is None:
            self._tau = gap
            self._tau_weight = 1
        else:
            self._tau = (self._tau * self._tau_weight + gap) / (self._tau_weight + 1)
            self._tau_weight += 1
        self._save_tau()
        logger.debug(
            f"[kv-disk] tau updated: {self._tau / 3600:.1f}h (weight={self._tau_weight})"
        )

    def _save_tau(self) -> None:
        _save_json(self._tau_path, {"tau": self._tau, "weight": self._tau_weight})

    def _compute_tau(self, now: float) -> float:
        if self._tau is not None:
            return self._tau
        # Bootstrap: no stored tau yet (first ever session). Arithmetic mean of ages.
        ages = [
            max(now - e.get("last_used", now), 1.0) for e in self._manifest.values()
        ]
        return sum(ages) / len(ages) if ages else 1.0

    def _evict_score(self, entry: dict, now: float, tau: float) -> float:
        age = max(now - entry.get("last_used", now), 1.0)
        return entry.get("hit_count", _INITIAL_HIT_COUNT) * math.exp(-age / tau)

    def _is_eviction_candidate(self, key: str, model_path: str) -> bool:
        """Return True if a new block would be immediately evicted after writing."""
        total = same_model_total = same_model_count = 0
        for e in self._manifest.values():
            size = e.get("file_size", 0)
            total += size
            if e.get("model_path") == model_path:
                same_model_total += size
                same_model_count += 1
        if not same_model_count:
            return False
        avg_size = same_model_total // same_model_count
        if total + avg_size <= _MAX_CACHE_BYTES:
            return False
        now = time.time()
        tau = self._compute_tau(now)
        # Dry-run: add the candidate with hit=1 and find the worst-scoring entry.
        test = {
            **self._manifest,
            key: {
                "file_size": avg_size,
                "last_used": now,
                "hit_count": _INITIAL_HIT_COUNT,
            },
        }
        worst = min(test, key=lambda k: self._evict_score(test[k], now, tau))
        return worst == key

    def _evict_if_needed(self) -> None:
        total = sum(e.get("file_size", 0) for e in self._manifest.values())
        if total <= _MAX_CACHE_BYTES:
            return
        now = time.time()
        while total > _MAX_CACHE_BYTES and self._manifest:
            tau = self._compute_tau(now)
            worst = min(
                self._manifest,
                key=lambda k: self._evict_score(self._manifest[k], now, tau),
            )
            entry = self._manifest.pop(worst)
            (self._cache_dir / f"{worst}.safetensors").unlink(missing_ok=True)
            total -= entry.get("file_size", 0)
            logger.info(
                f"[kv-disk] evicted {worst[:8]} ({entry.get('file_size', 0) // 1024**2} MB)"
            )

    def _save_manifest(self) -> None:
        _save_json(self._manifest_path, self._manifest)
