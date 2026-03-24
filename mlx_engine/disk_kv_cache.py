import hashlib
import json
import logging
import time
from pathlib import Path

import mlx_lm
from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "mlx-engine" / "kv_cache"
MIN_TOKENS = 1000
MAX_CACHE_BYTES = 2 * 1024**3  # 2 GB


def _sha256_tokens(tokens: list) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


class DiskKVCacheStore:
    """
    Persists KV cache snapshots to disk so that long system prompt prefills
    can be skipped on subsequent sessions.

    Layout:
        ~/.cache/mlx-engine/kv_cache/<sha256>.safetensors  — cache arrays
        ~/.cache/mlx-engine/kv_cache/manifest.json         — prefix lookup index

    Only caches snapshots with at least MIN_TOKENS tokens (default 1000).
    Each entry in the manifest stores the full token list so that prefix
    matching can be performed without loading the arrays.

    Disk LRU eviction: total cache size is capped at MAX_CACHE_BYTES (default
    2 GB).  Each entry tracks ``last_used`` (updated on every hit) and
    ``file_size``.  When a new save would exceed the cap, the least-recently-
    used entries are deleted first.
    """

    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._manifest_path = CACHE_DIR / "manifest.json"
        self._manifest: dict = {}
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    self._manifest = json.load(f)
            except Exception as e:
                logger.warning(f"[kv-disk] failed to load manifest: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_save(self, tokens: list, model_path: str, cache: list) -> None:
        """Persist *cache* to disk keyed by *tokens*, if above MIN_TOKENS.

        No-op if an entry for (sha256, model_path) already exists.
        """
        if len(tokens) < MIN_TOKENS:
            return

        sha = _sha256_tokens(tokens)

        cache_file = CACHE_DIR / f"{sha}.safetensors"

        # Skip if already saved for this model
        if sha in self._manifest and self._manifest[sha]["model_path"] == model_path:
            if cache_file.exists():
                return
            # Orphaned manifest entry (file deleted manually): remove and re-save
            logger.warning(f"[kv-disk] orphaned manifest entry {sha[:8]}, re-saving")
            del self._manifest[sha]
        try:
            save_prompt_cache(
                str(cache_file),
                cache,
                metadata={
                    "model_path": model_path,
                    "mlx_lm_version": mlx_lm.__version__,
                    "token_count": str(len(tokens)),
                },
            )
        except Exception as e:
            logger.warning(f"[kv-disk] save failed: {e}")
            return

        self._manifest[sha] = {
            "tokens": tokens,
            "model_path": model_path,
            "mlx_lm_version": mlx_lm.__version__,
            "token_count": len(tokens),
            "file_size": cache_file.stat().st_size,
            "last_used": time.time(),
        }
        self._evict_if_needed()
        self._save_manifest()
        logger.info(f"[kv-disk] saved len={len(tokens)} → {sha[:8]}...")

    def find_and_load(self, prompt_tokens: list, model_path: str):
        """Find the longest cached prefix of *prompt_tokens* for *model_path*.

        Returns:
            (cache, cached_count) on hit, or None on miss.
            cached_count: number of tokens already in the returned cache.
        """
        best_sha = None
        best_len = 0

        for sha, entry in self._manifest.items():
            if entry["model_path"] != model_path:
                continue
            stored = entry["tokens"]
            n = len(stored)
            if n <= best_len:
                continue
            if prompt_tokens[:n] == stored:
                best_sha = sha
                best_len = n

        if best_sha is None:
            return None

        cache_file = CACHE_DIR / f"{best_sha}.safetensors"
        if not cache_file.exists():
            logger.warning(f"[kv-disk] manifest entry {best_sha[:8]} missing on disk")
            del self._manifest[best_sha]
            self._save_manifest()
            return None

        try:
            cache = load_prompt_cache(str(cache_file))
        except Exception as e:
            logger.warning(f"[kv-disk] load failed: {e}")
            return None

        self._manifest[best_sha]["last_used"] = time.time()
        self._save_manifest()
        logger.info(f"[kv-disk] hit: {best_len}/{len(prompt_tokens)} tokens")
        return cache, best_len

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Delete least-recently-used entries until total size is below MAX_CACHE_BYTES."""
        total = sum(e.get("file_size", 0) for e in self._manifest.values())
        if total <= MAX_CACHE_BYTES:
            return

        # Sort by last_used ascending (oldest first); entries without last_used evicted first.
        by_age = sorted(
            self._manifest.items(), key=lambda kv: kv[1].get("last_used", 0)
        )
        for sha, entry in by_age:
            if total <= MAX_CACHE_BYTES:
                break
            cache_file = CACHE_DIR / f"{sha}.safetensors"
            try:
                cache_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"[kv-disk] evict unlink failed {sha[:8]}: {e}")
            total -= entry.get("file_size", 0)
            del self._manifest[sha]
            logger.info(f"[kv-disk] evicted {sha[:8]} (LRU, freed {entry.get('file_size', 0) // 1024**2} MB)")

    def _save_manifest(self) -> None:
        try:
            with open(self._manifest_path, "w") as f:
                json.dump(self._manifest, f)
        except Exception as e:
            logger.warning(f"[kv-disk] failed to write manifest: {e}")
