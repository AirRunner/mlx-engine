import bisect
import copy
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    ArraysCache,
    LRUPromptCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_engine.disk_kv_cache import PagedDiskKVCache, compute_block_hashes
from mlx_engine.utils.prompt_progress_reporter import (
    PromptProgressReporter,
    StopPromptProcessing,
)


PROMPT_PROCESSING_CHUNK_SIZE = 2048

# Checkpoint N tokens before end of prompt
# This value is at parity with mlx-lm:
# https://github.com/ml-explore/mlx-lm/blob/d9c63f/mlx_lm/server.py#L587
DEFAULT_CHECKPOINT_TAIL_TOKENS = 11

logger = logging.getLogger(__name__)


@dataclass
class _PrefillCheckpoint:
    """State captured at the LRU checkpoint boundary during update_cache.

    Consumed by finalize_generation to restore the cache and insert into LRU.
    """

    gdn_snapshot: List  # deepcopy of ArraysCache layers at the checkpoint
    lru_key: list  # think-normalized token key for LRU insertion
    kv_len: int  # original-space token count at the checkpoint boundary


def validate_prefill_step_size(prefill_step_size: Optional[int] = None) -> int:
    if prefill_step_size is None:
        return PROMPT_PROCESSING_CHUNK_SIZE
    if (
        isinstance(prefill_step_size, bool)
        or not isinstance(prefill_step_size, int)
        or prefill_step_size < 1
    ):
        raise ValueError("prefill_step_size must be a positive integer")
    return prefill_step_size


def _vram_str(kv_cache: list | None = None) -> str:
    active = mx.get_active_memory() // 1024**2
    alloc_cache = mx.get_cache_memory() // 1024**2
    total = active + alloc_cache
    if kv_cache is not None:
        kv_mb = sum(getattr(c, "nbytes", 0) for c in kv_cache) // 1024**2
        return f"VRAM {total} MB (active={active} cache={alloc_cache} kv={kv_mb})"
    return f"VRAM {total} MB (active={active} cache={alloc_cache})"


class CacheWrapper:
    def __init__(
        self,
        model: nn.Module,
        max_kv_size: Optional[int],
        *,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        chunk_size: int,
        checkpoint_tail_tokens: int = DEFAULT_CHECKPOINT_TAIL_TOKENS,
        history_capacity: int = 10,
        tokenizer=None,
        model_path: str = "",
    ):
        self.model = model
        self._draft_model: Optional[nn.Module] = None
        self._max_kv_size = max_kv_size
        self._chunk_size = chunk_size
        self._checkpoint_tail_tokens = checkpoint_tail_tokens
        self._history_capacity = history_capacity
        self._kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )

        self._history = self._make_history()
        self._history_key = "session"
        self._live_tokens: Optional[mx.array] = None
        self._live_cache: List[Any] = self._make_cache()

        # Set by update_cache, consumed and cleared by finalize_generation.
        self._prefill_checkpoint: Optional[_PrefillCheckpoint] = None
        # Checkpoint from the last finalized turn, used as fallback when the
        # LRU misses: restore and prefill only the delta.
        self._prev_checkpoint: Optional[_PrefillCheckpoint] = None
        # Optional image checkpoint store, activated by ModelKit when a
        # VisionAddOn is present.
        self._image_store: Optional["ImageCheckpointStore"] = None
        self._tokenizer = tokenizer
        self._model_path: str = model_path
        self._disk_store: Optional[PagedDiskKVCache] = (
            PagedDiskKVCache() if model_path else None
        )
        self._disk_save_queue: list[tuple] = []

    @property
    def cache(self) -> List[Any]:
        return self._live_cache

    @cache.setter
    def cache(self, value: List[Any]) -> None:
        self._live_cache = value

    @property
    def tokens(self) -> Optional[mx.array]:
        return self._live_tokens

    @tokens.setter
    def tokens(self, value: Optional[mx.array]) -> None:
        self._live_tokens = value

    def _make_cache(self) -> List[Any]:
        cache = make_prompt_cache(self.model, self._max_kv_size)
        if self._draft_model is not None:
            cache += make_prompt_cache(self._draft_model)
        return cache

    def _make_history(self) -> LRUPromptCache:
        # Store up to N checkpoints. This number can be tuned (or made configurable) if
        # it's too high or low
        return LRUPromptCache(max_size=self._history_capacity)

    def _num_tokens_in_cache(self, cache: Optional[List[Any]] = None) -> int | None:
        cache = self._live_cache if cache is None else cache
        for entry in cache:
            if hasattr(entry, "offset"):
                return entry.offset
        return None

    def _store_snapshot(
        self,
        tokens: mx.array,
        cache: List[Any],
        *,
        cache_type: Literal["user", "assistant"],
    ) -> None:
        if tokens.size == 0:
            return
        self._history.insert_cache(
            self._history_key,
            tokens.tolist(),
            copy.deepcopy(cache),
            cache_type=cache_type,
        )

    def _normalize_think_tokens(self, tokens: list) -> tuple[list, list[int]]:
        """Strip complete <think>...</think> blocks for think-invariant LRU keys.

        Clients often remove CoT blocks from conversation history. Normalizing
        before key comparison ensures the LRU still hits after such stripping.
        Incomplete trailing blocks are kept and handled by _checkpoint_offset.

        Pure-whitespace tokens immediately after THINK_END are also stripped:
        chat templates typically insert a separator between </think> and content,
        and it must be removed so the normalized key matches the no-think
        rendering. Whitespace membership is determined lazily via decode() and
        cached in ``self._think_ws_cache``.

        Returns (normalized_tokens, orig_indices) where orig_indices[i] is the
        index of normalized_tokens[i] in the original list.
        """
        if self._tokenizer is None or not getattr(
            self._tokenizer, "has_thinking", False
        ):
            return tokens, list(range(len(tokens)))
        think_start = getattr(self._tokenizer, "think_start_id", None)
        think_end = getattr(self._tokenizer, "think_end_id", None)
        if think_start is None or think_end is None:
            return tokens, list(range(len(tokens)))

        # Lazy whitespace-token cache: maps token_id → bool (True = pure ws).
        _ws = getattr(self, "_think_ws_cache", None)
        if _ws is None:
            _ws = {}
            self._think_ws_cache = _ws
        decode = getattr(self._tokenizer, "decode", None)

        result: list = []
        orig_indices: list = []
        i = 0
        while i < len(tokens):
            if tokens[i] == think_start:
                # Search for the matching end token.
                j = i + 1
                while j < len(tokens) and tokens[j] != think_end:
                    j += 1
                if j < len(tokens):
                    # Complete block: skip i..j inclusive, then strip any
                    # pure-whitespace separator tokens that immediately follow.
                    next_i = j + 1
                    while next_i < len(tokens):
                        tid = tokens[next_i]
                        if tid not in _ws:
                            _ws[tid] = decode is not None and not decode([tid]).strip()
                        if _ws[tid]:
                            next_i += 1
                        else:
                            break
                    i = next_i
                else:
                    # Trailing open block: keep as-is.
                    result.append(tokens[i])
                    orig_indices.append(i)
                    i += 1
            else:
                result.append(tokens[i])
                orig_indices.append(i)
                i += 1
        return result, orig_indices

    def _checkpoint_offset(self, tokens: list) -> int:
        """Number of trailing tokens excluded from the LRU checkpoint key.

        For thinking models the key ends before the generation-prompt prefix
        (role header + <think>) so that the stored key is a valid prefix of
        the next turn's prompt regardless of role.

        The number of tokens between the role header and <think> (inclusive)
        is read from ``tokenizer.thinking_prefix_offset`` (default 3 for
        ChatML-based templates).

        Falls back to _checkpoint_tail_tokens for non-thinking models.
        """
        if self._tokenizer is not None and getattr(
            self._tokenizer, "has_thinking", False
        ):
            think_id = getattr(self._tokenizer, "think_start_id", None)
            if think_id is not None:
                prefix_len = getattr(self._tokenizer, "thinking_prefix_offset", 3)
                for i in range(1, min(11, len(tokens))):
                    if tokens[-i] == think_id:
                        return i + prefix_len
        return self._checkpoint_tail_tokens

    def _record_lru_hit(self, token_list: list, cached_token_count: int) -> None:
        if self._disk_store:
            self._disk_store.record_lru_hit(token_list, cached_token_count)

    def _restore_cache(
        self,
        prompt_tokens: mx.array,
    ) -> tuple[Optional[List[Any]], mx.array]:
        if len(prompt_tokens) == 0:
            return None, prompt_tokens

        token_list = prompt_tokens.tolist()
        norm_tokens, norm_orig = self._normalize_think_tokens(token_list)

        # Think-normalized LRU lookup.
        cache, remaining_norm = self._history.fetch_nearest_cache(
            self._history_key,
            norm_tokens,
        )
        if cache is not None:
            cached_norm = len(norm_tokens) - len(remaining_norm)
            start = (
                norm_orig[cached_norm]
                if cached_norm < len(norm_orig)
                else len(token_list)
            )
            if start < len(token_list):
                self._record_lru_hit(token_list, start)
                return cache, prompt_tokens[start:]

            # Exact hit: try to trim 1 token to seed decode.
            if can_trim_prompt_cache(cache) and trim_prompt_cache(cache, 1) == 1:
                self._record_lru_hit(token_list, len(token_list) - 1)
                return cache, prompt_tokens[-1:]

        if len(prompt_tokens) <= 1:
            return None, prompt_tokens

        # Exact hits need one token outside the cache to seed decode. If the
        # exact-hit cache cannot be trimmed, retry with one less prompt token
        # so a stored checkpoint can win.
        truncated_norm, truncated_orig = self._normalize_think_tokens(token_list[:-1])
        cache, rest = self._history.fetch_nearest_cache(
            self._history_key,
            truncated_norm,
        )
        if cache is not None:
            cached_norm = len(truncated_norm) - len(rest)
            start = (
                truncated_orig[cached_norm]
                if cached_norm < len(truncated_orig)
                else len(token_list) - 1
            )
            self._record_lru_hit(token_list, start)
            return cache, prompt_tokens[start:]

        # Prev-checkpoint fallback: if LRU missed but the current prompt
        # shares the same normalized prefix as the last finalized turn,
        # reuse the live cache directly (no deepcopy) and prefill the delta.
        if (
            self._prev_checkpoint is not None
            and self._live_cache is not None
            and self._prev_checkpoint.kv_len <= len(token_list)
            and norm_tokens[: len(self._prev_checkpoint.lru_key)]
            == self._prev_checkpoint.lru_key
        ):
            # Always restore GDN: finalize may be a no-op (no _prefill_checkpoint),
            # leaving GDN post-generation even when KV offset appears clean.
            kv_offset = self._num_tokens_in_cache()
            n_to_trim = (
                (kv_offset - self._prev_checkpoint.kv_len)
                if kv_offset is not None
                else 0
            )
            self._apply_gdn_snapshot(self._prev_checkpoint.gdn_snapshot, n_to_trim)
            self._record_lru_hit(token_list, self._prev_checkpoint.kv_len)
            return self._live_cache, prompt_tokens[self._prev_checkpoint.kv_len :]

        if self._disk_store:
            result = self._disk_store.find_and_load(
                token_list,
                self._model_path,
                kv_bits=self._kv_cache_qtn_params.get("kv_bits"),
            )
            if result is not None:
                disk_cache, cached_count = result
                return disk_cache, prompt_tokens[cached_count:]

        return None, prompt_tokens

    def _prefill_cache(
        self,
        model: nn.Module,
        cache: List[Any],
        cache_start: int,
        tokens: mx.array,
        reporter: PromptProgressReporter,
        is_draft: bool,
        checkpoint_prefix_len: Optional[int] = None,
    ) -> None:
        remaining_tokens = tokens
        num_processed = 0
        stored_checkpoint = False

        while remaining_tokens.size > 0:
            current_chunk_size = min(self._chunk_size, remaining_tokens.size)
            current_cache_size = self._num_tokens_in_cache(cache)
            if (
                checkpoint_prefix_len is not None
                and current_cache_size is not None
                and current_cache_size < checkpoint_prefix_len
                and current_cache_size + current_chunk_size > checkpoint_prefix_len
            ):
                current_chunk_size = checkpoint_prefix_len - current_cache_size

            current_chunk = remaining_tokens[:current_chunk_size]
            model(current_chunk[None], cache=cache)
            maybe_quantize_kv_cache(prompt_cache=cache, **self._kv_cache_qtn_params)
            self._live_cache[cache_start : cache_start + len(cache)] = cache
            mx.eval([entry.state for entry in cache])

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size
            mx.clear_cache()

            current_cache_size = self._num_tokens_in_cache(cache)

            if not is_draft and self._disk_save_queue and self._disk_store:
                kv_bits = self._kv_cache_qtn_params.get("kv_bits")
                cache_fully_quantized = kv_bits is None or not any(
                    hasattr(c, "to_quantized") for c in cache
                )
                remaining = []
                for h, start, end in self._disk_save_queue:
                    if current_cache_size is not None and current_cache_size >= end:
                        if cache_fully_quantized:
                            self._disk_store.save_block(
                                h, cache, start, end, self._model_path
                            )
                        else:
                            remaining.append((h, start, end))
                    else:
                        remaining.append((h, start, end))
                self._disk_save_queue = remaining
            if (
                checkpoint_prefix_len is not None
                and not stored_checkpoint
                and current_cache_size == checkpoint_prefix_len
            ):
                # GDN snapshot at checkpoint boundary (before <think>).
                # KV layers are trimmed back to this point in finalize_generation();
                # ArraysCache (GDN) layers are restored from this snapshot in-place.
                gdn_snapshot = [
                    copy.deepcopy(c) if isinstance(c, ArraysCache) else None
                    for c in self._live_cache
                ]
                token_list = self._live_tokens.tolist()
                norm_tokens, norm_orig = self._normalize_think_tokens(token_list)
                if len(norm_tokens) < len(token_list):
                    norm_cp = bisect.bisect_left(norm_orig, checkpoint_prefix_len)
                    lru_key = norm_tokens[:norm_cp]
                else:
                    lru_key = token_list[:checkpoint_prefix_len]
                self._prefill_checkpoint = _PrefillCheckpoint(
                    gdn_snapshot=gdn_snapshot,
                    lru_key=lru_key,
                    kv_len=checkpoint_prefix_len,
                )
                stored_checkpoint = True

            if not reporter.update(is_draft, num_processed):
                logger.info("Prompt processing was cancelled by the user.")
                live_cache_size = self._num_tokens_in_cache()
                if live_cache_size is None:
                    self._live_tokens = None
                    self._live_cache = self._make_cache()
                else:
                    self._live_tokens = self._live_tokens[:live_cache_size]
                raise StopPromptProcessing

    def update_cache(
        self,
        prompt_tokens: mx.array,
        reporter: PromptProgressReporter,
        *,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        num_tokens_to_exclude = max(num_tokens_to_exclude, 1)
        total_prompt_tokens = len(prompt_tokens)
        token_list = prompt_tokens.tolist()

        # If the generator from the previous turn was not exhausted by the caller
        # (early stop after receiving stop_condition), finalize_generation was never
        # called. Do it now before starting the new prefill.
        if self._prefill_checkpoint is not None:
            self.finalize_generation()

        restored_cache, uncached_tokens = self._restore_cache(prompt_tokens)
        self._live_cache = (
            restored_cache if restored_cache is not None else self._make_cache()
        )
        self._live_tokens = prompt_tokens

        cached_tokens = total_prompt_tokens - len(uncached_tokens)
        logger.info(
            "Prompt cache: using %d/%d tokens from cache",
            cached_tokens,
            total_prompt_tokens,
        )

        if self._disk_store:
            bs = self._disk_store._block_size
            hashes = compute_block_hashes(token_list, bs)
            self._disk_save_queue = [
                (h, i * bs, (i + 1) * bs)
                for i, h in enumerate(hashes)
                if self._disk_store.should_save_block(h, self._model_path)
            ]
        else:
            self._disk_save_queue = []

        reporter.begin(
            is_draft=False,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            prefill_tokens_processed=0,
        )

        # Leave num_tokens_to_exclude tokens outside the cache to seed decode.
        num_tokens_to_exclude = min(num_tokens_to_exclude, len(uncached_tokens))
        prefill_tokens = uncached_tokens[:-num_tokens_to_exclude]

        # Checkpoint position: for thinking models, exclude trailing <think>
        # tokens so the stored key is a stable prefix of the next turn's query.
        # For non-thinking models, fall back to _checkpoint_tail_tokens (11).
        checkpoint_offset = self._checkpoint_offset(token_list)
        checkpoint_prefix_len = total_prompt_tokens - checkpoint_offset
        # Cannot go before what's already cached, and skip if non-positive.
        if checkpoint_prefix_len <= cached_tokens or checkpoint_prefix_len <= 0:
            checkpoint_prefix_len = None
        # Only checkpoint the main-model path (not draft model).
        if self._draft_model is not None:
            checkpoint_prefix_len = None

        with mx.stream(generation_stream):
            try:
                if self._draft_model is not None:
                    draft_cache = self._live_cache[len(self.model.layers) :]
                    self._prefill_cache(
                        model=self._draft_model,
                        cache=draft_cache,
                        cache_start=len(self.model.layers),
                        tokens=prefill_tokens,
                        reporter=reporter,
                        is_draft=True,
                        checkpoint_prefix_len=None,
                    )

                main_cache = self._live_cache[: len(self.model.layers)]
                self._prefill_cache(
                    model=self.model,
                    cache=main_cache,
                    cache_start=0,
                    tokens=prefill_tokens,
                    reporter=reporter,
                    is_draft=False,
                    checkpoint_prefix_len=checkpoint_prefix_len,
                )
            except StopPromptProcessing:
                if (
                    self._prefill_checkpoint is None
                    and self._prev_checkpoint is not None
                ):
                    # Cancelled before checkpoint: roll back to the previous
                    # turn's state and re-insert into the history.
                    kv_offset = self._num_tokens_in_cache()
                    n_to_trim = (
                        (kv_offset - self._prev_checkpoint.kv_len)
                        if kv_offset is not None
                        else 0
                    )
                    self._restore_and_insert(
                        self._prev_checkpoint.gdn_snapshot,
                        self._prev_checkpoint.lru_key,
                        n_to_trim,
                    )
                    logger.info(
                        "[kv] prefill cancelled, rolled back to previous checkpoint"
                    )
                raise

        reporter.finish(
            is_draft=False,
            prefill_tokens_processed=total_prompt_tokens - cached_tokens,
        )
        logger.info(
            f"[kv] prefill done tokens={total_prompt_tokens} {_vram_str(self._live_cache)}"
        )
        return uncached_tokens[-num_tokens_to_exclude:]

    def _apply_gdn_snapshot(self, gdn_snapshot: list, n_to_trim: int) -> None:
        """Restore GDN layers from snapshot and trim KV layers in _live_cache."""
        for c, snap in zip(self._live_cache, gdn_snapshot):
            if snap is not None:
                c.cache = snap.cache.copy()
            else:
                c.trim(n_to_trim)

    def _restore_and_insert(
        self, gdn_snapshot: list, lru_key: list, n_to_trim: int
    ) -> None:
        """Restore GDN layers from snapshot, trim KV layers, and insert into history."""
        self._apply_gdn_snapshot(gdn_snapshot, n_to_trim)
        lru_key_arr = mx.array(lru_key)
        self._store_snapshot(lru_key_arr, self._live_cache, cache_type="user")
        self._prefill_checkpoint = None

    def finalize_generation(self) -> None:
        """Restore cache to post-prefill state and insert into history.

        Called after stream_generate completes (or is cancelled). Trims KV layers
        back to the prefill boundary and restores ArraysCache (GDN) layers from the
        snapshot taken in _prefill_cache. The resulting clean cache is stored in the
        history for next-turn prefix reuse.
        """
        if self._prefill_checkpoint is None or self._live_tokens is None:
            return
        cp = self._prefill_checkpoint
        n_to_trim = len(self._live_tokens) - cp.kv_len
        # Preserve prev-checkpoint before _restore_and_insert clears it.
        self._prev_checkpoint = _PrefillCheckpoint(
            gdn_snapshot=cp.gdn_snapshot,
            lru_key=cp.lru_key,
            kv_len=cp.kv_len,
        )
        self._restore_and_insert(cp.gdn_snapshot, cp.lru_key, n_to_trim)
        kv_offset = next(
            (c.offset for c in self._live_cache if hasattr(c, "offset")), -1
        )
        logger.info(
            f"[kv] finalize done key_len={len(cp.lru_key)} kv_offset={kv_offset}"
            f" history_size={len(self._history)} {_vram_str(self._live_cache)}"
        )

    def set_image_turn_checkpoint(
        self, cache: list, token_list: list, cached_tokens: int
    ) -> None:
        """Set _prefill_checkpoint after an image-path prefill.

        Mirrors the checkpoint logic in _prefill_cache so that finalize_generation
        can trim KV layers, restore GDN layers, and insert into the history after an
        image-turn generation completes.

        Args:
            cache:         The live KV cache after the image prefill.
            token_list:    Full VLM-expanded prompt as a flat Python list.
            cached_tokens: Number of tokens already in cache before the image
                           prefill started (used as a lower bound for kv_len).
        """
        checkpoint_offset = self._checkpoint_offset(token_list)
        checkpoint_idx = len(token_list) - checkpoint_offset
        effective_checkpoint = max(checkpoint_idx, cached_tokens)
        norm_tokens, norm_orig = self._normalize_think_tokens(token_list)
        if len(norm_tokens) < len(token_list):
            norm_cp = bisect.bisect_left(norm_orig, effective_checkpoint)
            lru_key = norm_tokens[:norm_cp]
        else:
            lru_key = token_list[:effective_checkpoint]
        gdn_snapshot = [
            copy.deepcopy(c) if isinstance(c, ArraysCache) else None for c in cache
        ]
        self._prefill_checkpoint = _PrefillCheckpoint(
            gdn_snapshot=gdn_snapshot, lru_key=lru_key, kv_len=effective_checkpoint
        )

    def _find_starting_cache(
        self, token_list: list, *, prefer_prev_checkpoint: bool = False
    ) -> tuple[Any, int]:
        """Return (cache, start_idx) for the best available prefix of token_list.

        Used by the image path in generate.py, which needs (cache, int) rather
        than (cache, mx.array). Delegates to _restore_cache internally.

        Args:
            token_list:             Full prompt as a flat Python list.
            prefer_prev_checkpoint: When True, try prev-checkpoint before the LRU.
        """
        if (
            prefer_prev_checkpoint
            and self._prev_checkpoint is not None
            and self._live_cache is not None
        ):
            return self._live_cache, self._prev_checkpoint.kv_len

        prompt_tokens = mx.array(token_list)
        cache, rest = self._restore_cache(prompt_tokens)
        if cache is not None:
            start = len(token_list) - len(rest)
            return cache, start
        return None, 0

    def record_generated_token(self, token: int) -> None:
        if self._live_tokens is None:
            self._live_tokens = mx.array([token])
            return
        self._live_tokens = mx.concat([self._live_tokens, mx.array([token])])

    def set_draft_model(self, draft_model: nn.Module) -> None:
        if self.model is None:
            raise ValueError("Cannot add a draft model to cache without a main model")
        if self._draft_model is draft_model:
            return
        if self._max_kv_size is not None:
            logger.info("Disabling max_kv_size when setting a draft model for cache")
            self._max_kv_size = None

        self._history = self._make_history()
        self._draft_model = draft_model
        self._live_tokens = None
        self._live_cache = self._make_cache()
        self._prefill_checkpoint = None
        self._prev_checkpoint = None

    def unset_draft_model(self) -> None:
        if self._draft_model is None:
            return
        main_cache = self._live_cache[: len(self.model.layers)]
        self._history = self._make_history()
        self._draft_model = None
        self._prefill_checkpoint = None
        self._prev_checkpoint = None
        if len(main_cache) == len(self.model.layers):
            self._live_cache = main_cache
            return
        self._live_tokens = None
        self._live_cache = self._make_cache()


def image_block_boundaries(
    ids_flat: list, img_tok: int, vid_tok: Optional[int]
) -> list:
    """Return (start, end) index pairs for each contiguous image token block."""
    boundaries = []
    in_block = False
    start = 0
    for i, tok in enumerate(ids_flat):
        is_img = tok == img_tok or tok == vid_tok
        if is_img and not in_block:
            start, in_block = i, True
        elif not is_img and in_block:
            boundaries.append((start, i))
            in_block = False
    if in_block:
        boundaries.append((start, len(ids_flat)))
    return boundaries


def image_block_lengths(ids_flat: list, img_tok: int, vid_tok: Optional[int]) -> tuple:
    """Return the number of image tokens in each contiguous image block, in order."""
    return tuple(e - s for s, e in image_block_boundaries(ids_flat, img_tok, vid_tok))


class ImageCheckpointStore:
    """
    Standalone store for per-image-block KV cache checkpoints.

    Keyed by a tuple of per-image SHA-256 hex digests, e.g. (hash1,) for the
    first image block, (hash1, hash2) for the second, etc. On the next turn
    the deepest matching prefix is restored and only the new text tokens are
    prefilled, skipping the vision tower entirely.

    Designed to be attached to a CacheWrapper (ModelKit path) as an optional
    component via its ``_image_store`` field.
    """

    def __init__(self) -> None:
        # key: tuple[str, ...] (per-image SHA-256 hash chain)
        # value: (image_end_index: int, prefix_hash: int, block_lengths: tuple)
        self._image_checkpoints: dict[tuple, tuple] = {}

    def clear(self) -> None:
        """Drop all image KV checkpoints and release their Metal buffers."""
        if self._image_checkpoints:
            self._image_checkpoints = {}
            mx.clear_cache()
            logger.info("[kv-image] all image checkpoints freed")

    def save_image_checkpoint(
        self,
        key: tuple,
        image_end_index: int,
        prefix_hash: int,
        block_lengths: tuple = (),
    ) -> None:
        """
        Persist image checkpoint metadata right after an image block.

        No KV tensors are stored: the text LRU (via CacheWrapper) is used at
        restore time, which keeps VRAM at 1x KV instead of 2x.

        Args:
            key:             Tuple of per-image SHA-256 hex digests up to and
                             including this block, e.g. (hash1,) or (hash1, hash2).
            image_end_index: First text-token index after this image block.
            prefix_hash:     Python hash of the VLM token ids up to image_end_index
                             at save time. Used to detect stale checkpoints from a
                             different conversation that happens to share the same images.
            block_lengths:   Number of image tokens in each block up to and
                             including this checkpoint, e.g. (782,) or (782, 874).
                             Used for a fast structural staleness check: if block
                             lengths differ on the next turn, image tokenization
                             changed (e.g. dynamic resolution padding) and the
                             checkpoint must be invalidated before computing hashes.
        """
        self._image_checkpoints[key] = (
            image_end_index,
            prefix_hash,
            block_lengths,
        )
        logger.info(
            f"[kv-image] checkpoint saved depth={len(key)} index={image_end_index}"
        )

    def get_image_checkpoint(self, key: tuple):
        """
        Return ``(image_end_index, prefix_hash, block_lengths)`` for *key*, or None.
        """
        return self._image_checkpoints.get(key)

    def validate_image_checkpoint(
        self,
        check_key: tuple,
        input_ids_flat: list,
        img_tok: Optional[int],
        vid_tok: Optional[int],
        current_block_lengths: tuple,
    ) -> bool:
        """Validate a stored image checkpoint against the current token sequence.

        Performs three checks in order:

        1. **Block-length gap check**: when block token counts differ (e.g. due to
           dynamic resolution re-padding), verifies that all extra tokens in the
           gap are image-pad tokens. If so, the checkpoint is still valid because
           the partial-hit prefill will cover the gap naturally. Non-pad tokens
           indicate a real content change and must invalidate the cache.
        2. **Bounds check**: ``stored_end_idx`` must not exceed the current
           sequence length.
        3. **Prefix hash**: the hash of tokens up to ``stored_end_idx`` must match.

        Args:
            check_key:             Checkpoint key to validate (image hash tuple).
            input_ids_flat:        Full VLM token sequence as a flat list.
            img_tok:               Image pad token ID (``None`` if unavailable).
            vid_tok:               Video pad token ID (``None`` if unavailable).
            current_block_lengths: Image block token counts for the current turn.

        Returns:
            ``True`` if the checkpoint is stale and must be invalidated.
        """
        entry = self.get_image_checkpoint(check_key)
        if entry is None:
            return False

        stored_end_idx, stored_prefix_hash, stored_block_lengths = entry
        depth = len(check_key)

        # 1. Block-length gap check.
        if (
            stored_block_lengths
            and current_block_lengths
            and current_block_lengths[:depth] != stored_block_lengths
        ):
            block_bounds = (
                image_block_boundaries(input_ids_flat, img_tok, vid_tok)
                if img_tok is not None
                else []
            )
            if depth > len(block_bounds):
                logger.info(
                    f"[kv-image] stale checkpoint (depth={depth}): "
                    f"sequence has fewer image blocks than checkpoint "
                    f"({len(block_bounds)} < {depth})"
                )
                return True
            blk_end = block_bounds[depth - 1][1]
            gap = (
                input_ids_flat[stored_end_idx:blk_end]
                if stored_end_idx < blk_end
                else []
            )
            if all(t == img_tok or t == vid_tok for t in gap):
                logger.info(
                    f"[kv-image] depth={depth}: image block re-padded "
                    f"({stored_block_lengths[depth - 1]}"
                    f"→{current_block_lengths[depth - 1]} tokens), "
                    "checkpoint accepted"
                )
            else:
                logger.info(
                    f"[kv-image] stale checkpoint (depth={depth}): "
                    "image block content changed "
                    f"(stored={stored_block_lengths}, "
                    f"current={current_block_lengths[:depth]})"
                )
                return True

        # 2. Bounds check.
        if stored_end_idx > len(input_ids_flat):
            logger.info(
                f"[kv-image] stale checkpoint (depth={depth}): "
                f"stored_end_idx={stored_end_idx} exceeds "
                f"current sequence length {len(input_ids_flat)}"
            )
            return True

        # 3. Prefix hash.
        current_hash = hash(tuple(input_ids_flat[:stored_end_idx]))
        if current_hash != stored_prefix_hash:
            logger.info(
                f"[kv-image] stale checkpoint (depth={depth}): "
                f"prefix hash mismatch at position {stored_end_idx}"
            )
            return True

        return False

    def save_block_checkpoints(
        self,
        hash_chain: tuple,
        offset: int,
        block_checkpoints: list,
        input_ids_flat: list,
        block_lengths: tuple,
    ) -> None:
        """Compute per-block prefix hashes and persist image checkpoint metadata.

        Args:
            hash_chain:        Full image hash chain for this turn.
            offset:            Number of already-cached image blocks.
            block_checkpoints: List of ``image_end_index`` values (int), one per
                               newly processed image block.
            input_ids_flat:    Full VLM token sequence as a flat list.
            block_lengths:     Image block token counts for the full turn.
        """
        for i, end_idx in enumerate(block_checkpoints):
            pfx_hash = hash(tuple(input_ids_flat[:end_idx]))
            self.save_image_checkpoint(
                hash_chain[: offset + i + 1],
                end_idx,
                pfx_hash,
                block_lengths[: offset + i + 1],
            )

    def invalidate_image_checkpoint(self, key: tuple) -> None:
        """Remove a stale image checkpoint."""
        self._image_checkpoints.pop(key, None)
        logger.info(f"[kv-image] stale checkpoint invalidated depth={len(key)}")

    def find_deepest_image_checkpoint(self, hash_chain: tuple):
        """
        Return (key, image_end_index) for the longest prefix of *hash_chain*
        that has a stored checkpoint, or None if no prefix matches.
        """
        for depth in range(len(hash_chain), 0, -1):
            key = hash_chain[:depth]
            entry = self._image_checkpoints.get(key)
            if entry is not None:
                return key, entry[0]
        return None

    # TODO: remove once lmstudio-ai/lmstudio-bug-tracker#1663 is fixed upstream.
    def reorder_images_chronologically(
        self,
        images_b64: list,
        image_hashes: list,
    ) -> tuple:
        """Reorder images into chronological conversation order using stored checkpoints.

        Uses the checkpoint history, built turn-by-turn in the correct order,
        to reconstruct the canonical image sequence. Any images not yet seen
        in a checkpoint are appended in the order received.

        Args:
            images_b64:   Images as received, possibly out of order.
            image_hashes: SHA-256 hex digests corresponding to images_b64.

        Returns:
            (reordered_images_b64, reordered_hashes) in chronological order.
            Returns the inputs unchanged when no checkpoint overlap is found
            (e.g. the very first turn).
        """
        hash_set = set(image_hashes)

        # Find the deepest checkpoint whose entire key is a subset of received hashes.
        best_key: tuple = ()
        for key in self._image_checkpoints:
            if len(key) > len(best_key) and all(h in hash_set for h in key):
                best_key = key

        if not best_key:
            return images_b64, image_hashes

        hash_to_img = dict(zip(image_hashes, images_b64))
        ordered_imgs = [hash_to_img[h] for h in best_key]
        ordered_hashes = list(best_key)

        # New images: entries not covered by the checkpoint, including duplicates.
        checkpoint_counts = Counter(best_key)
        remaining = {
            h: max(0, cnt - checkpoint_counts.get(h, 0))
            for h, cnt in Counter(image_hashes).items()
        }
        new_imgs, new_hashes = [], []
        for img, h in zip(images_b64, image_hashes):
            if remaining.get(h, 0) > 0:
                new_imgs.append(img)
                new_hashes.append(h)
                remaining[h] -= 1

        reordered_imgs = ordered_imgs + new_imgs
        reordered_hashes = ordered_hashes + new_hashes

        if reordered_hashes != image_hashes:
            logger.info(
                f"[kv-image] reordered {len(images_b64)} images to chronological order"
            )

        return reordered_imgs, reordered_hashes
