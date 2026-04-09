import bisect
import copy
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    ArraysCache,
    KVCache,
    LRUPromptCache,
    make_prompt_cache,
)

from mlx_engine.utils.prompt_progress_reporter import (
    PromptProgressReporter,
    StopPromptProcessing,
)


PROMPT_PROCESSING_CHUNK_SIZE = 2048


@dataclass
class _PrefillCheckpoint:
    """State captured at the LRU checkpoint boundary during update_cache.

    Consumed by finalize_generation to restore the cache and insert into LRU.
    """

    gdn_snapshot: List  # deepcopy of ArraysCache layers at the checkpoint
    lru_key: list  # think-normalized token key for LRU insertion
    kv_len: int  # original-space token count at the checkpoint boundary


def validate_prefill_step_size(prefill_step_size: Optional[int] = None) -> int:
    """
    Resolve and validate the configured prefill chunk size.

    Args:
        prefill_step_size: Optional override for tokens processed per prefill chunk.

    Returns:
        int: The provided chunk size, or PROMPT_PROCESSING_CHUNK_SIZE when unset.

    Raises:
        ValueError: If prefill_step_size is not a positive integer.
    """
    if prefill_step_size is None:
        return PROMPT_PROCESSING_CHUNK_SIZE
    if (
        isinstance(prefill_step_size, bool)
        or not isinstance(prefill_step_size, int)
        or prefill_step_size < 1
    ):
        raise ValueError("prefill_step_size must be a positive integer")
    return prefill_step_size


logger = logging.getLogger(__name__)


def _vram_str(kv_cache: list | None = None) -> str:
    active = mx.get_active_memory() // 1024**2
    alloc_cache = mx.get_cache_memory() // 1024**2
    total = active + alloc_cache
    if kv_cache is not None:
        kv_mb = sum(getattr(c, "nbytes", 0) for c in kv_cache) // 1024**2
        return f"VRAM {total} MB (active={active} cache={alloc_cache} kv={kv_mb})"
    return f"VRAM {total} MB (active={active} cache={alloc_cache})"


class CacheWrapper:
    """
    Wrapper class for the MLX LM cache to maintain an in-memory cache
    """

    def __init__(
        self,
        model: nn.Module,
        max_kv_size: Optional[int],
        *,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        chunk_size: int,
        tokenizer=None,
    ):
        """
        Initialize the CacheWrapper.

        Args:
            model (nn.Module): The model to be cached.
            max_kv_size (Optional[int]): Maximum size of the key-value cache.
            chunk_size (int): Number of tokens per prefill chunk.
        """
        self.tokens: Optional[mx.array] = None
        self.cache: List[Any] = make_prompt_cache(model, max_kv_size)
        self._lru: LRUPromptCache = LRUPromptCache(max_size=1)
        # Set by update_cache, consumed by finalize_generation.
        # Set by update_cache, consumed and cleared by finalize_generation.
        self._prefill_checkpoint: Optional[_PrefillCheckpoint] = None
        # Checkpoint from the last finalized turn, used as fallback when both
        # LRU and disk miss: restore and prefill only the delta.
        self._prev_gdn_snapshot: Optional[List] = None
        self._prev_kv_len: Optional[int] = None
        self._prev_lru_key: Optional[list] = None
        self.model = model
        self.draft_model: Optional[nn.Module] = None
        self.max_kv_size = max_kv_size
        self.kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self.chunk_size = chunk_size
        self._image_store: Optional[ImageCheckpointStore] = None
        self._tokenizer = tokenizer

    def _prefill(
        self,
        model,
        cache,
        tokens,
        reporter: PromptProgressReporter,
        is_draft: bool,
        *,
        progress_offset: int = 0,
    ):
        """
        Fill a KV cache for a specific model

        Args:
            model: The model to use for cache filling
            cache: The cache to fill
            tokens: Tokens to process
            reporter: Reporter for reporting progress
            is_draft: Whether this is draft model prefill (True) or main model (False)
        """
        remaining_tokens = tokens
        num_processed = progress_offset

        while remaining_tokens.size > 0:
            current_chunk_size = min(self.chunk_size, remaining_tokens.size)
            current_chunk = remaining_tokens[:current_chunk_size]

            model(current_chunk[None], cache=cache)
            maybe_quantize_kv_cache(prompt_cache=cache, **self.kv_cache_qtn_params)
            mx.eval([c.state for c in cache])

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size

            mx.clear_cache()

            # Report progress
            should_continue = reporter.update(is_draft, num_processed)
            if not should_continue:
                logger.info("Prompt processing was cancelled by the user.")
                raise StopPromptProcessing

    def set_draft_model(self, draft_model: nn.Module):
        """
        Sets or updates the draft model to use in the cache.

        If the provided draft_model is already set, returns without changes.
        Otherwise, clears existing cache and rebuilds it by combining caches
        from the main model and draft model. Requires a main model to be set first.
        Args:
            draft_model: The draft model to cache. Pass None to remove draft model.

        Raises:
            ValueError: If main model hasn't been set yet.
        """
        if self.model is None:
            raise ValueError("Cannot add a draft model to cache without a main model")
        if self.max_kv_size is not None:
            logger.info("Disabling max_kv_size when setting a draft model for cache")
            self.max_kv_size = None

        if self.draft_model is draft_model:
            # Skip if the exact same draft model instance is already in cache
            return

        # clear the current cache, append draft model cache to the end of the main model cache as per
        # https://github.com/ml-explore/mlx-examples/blob/514502da22f0dc4c1ac439bdf78c07d5ec41acf7/llms/mlx_lm/utils.py#L381-L382
        logger.info("Clearing current prompt cache and adding draft model to the cache")
        self.tokens = None
        self._lru = LRUPromptCache(max_size=1)
        self._prefill_checkpoint = None
        self._prev_gdn_snapshot = None
        self._prev_kv_len = None
        self._prev_lru_key = None
        self.cache: List[Any] = make_prompt_cache(self.model)
        if draft_model is not None:
            self.cache += make_prompt_cache(draft_model)
        self.draft_model = draft_model

    def unset_draft_model(self):
        """Removes the draft model from the cache if one exists."""
        if self.draft_model is None:
            return
        self.draft_model = None
        self.tokens = None
        self._lru = LRUPromptCache(max_size=1)
        self._prefill_checkpoint = None
        self._prev_gdn_snapshot = None
        self._prev_kv_len = None
        self._prev_lru_key = None
        self.cache = self.cache[: len(self.model.layers)]

    def _checkpoint_offset(self, tokens: list) -> int:
        """Number of trailing tokens excluded from the LRU checkpoint key.

        For thinking models the key ends before the generation-prompt prefix
        (role header + <think>) so that the stored key is a valid prefix of
        the next turn's prompt regardless of role.

        The number of tokens between the role header and <think> (inclusive)
        is read from ``tokenizer.thinking_prefix_offset`` (default 3 for
        ChatML-based templates).

        Falls back to 1 for non-thinking models.
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
        return 1

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

    def _find_starting_cache(
        self, token_list: list, *, prefer_prev_checkpoint: bool = False
    ) -> tuple[Any, int]:
        """Return (cache, start_idx) for the best available prefix of token_list.

        Checks, in order: prev-checkpoint (if preferred), LRU, prev-checkpoint
        (fallback). Evicts the LRU when a cache is returned so VRAM stays at 1x.
        Returns (None, 0) when no cached prefix is available.

        Args:
            token_list:             Full prompt as a flat Python list.
            prefer_prev_checkpoint: When True, try prev-checkpoint before the LRU.
                                    Use this in the image path to avoid the deepcopy
                                    that fetch_nearest_cache performs.
        """
        if (
            prefer_prev_checkpoint
            and self._prev_kv_len is not None
            and self.cache is not None
        ):
            self._lru = LRUPromptCache(max_size=1)
            return self.cache, self._prev_kv_len

        norm_tokens, norm_orig = self._normalize_think_tokens(token_list)
        cache, remaining_norm = self._lru.fetch_nearest_cache("main", norm_tokens)
        if cache is not None:
            self._lru = LRUPromptCache(max_size=1)
            cached_norm = len(norm_tokens) - len(remaining_norm)
            start = (
                norm_orig[cached_norm]
                if cached_norm < len(norm_orig)
                else len(token_list)
            )
            return cache, start

        if (
            self._prev_kv_len is not None
            and self.cache is not None
            and self._prev_lru_key is not None
            and self._prev_kv_len <= len(token_list)
            and norm_tokens[: len(self._prev_lru_key)] == self._prev_lru_key
        ):
            self._lru = LRUPromptCache(max_size=1)
            return self.cache, self._prev_kv_len

        return None, 0

    def set_image_turn_checkpoint(
        self, cache: list, token_list: list, cached_tokens: int
    ) -> None:
        """Set _prefill_checkpoint after an image-path prefill.

        Mirrors the checkpoint logic in update_cache so that finalize_generation
        can trim KV layers, restore GDN layers, and insert into the LRU after an
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

    def update_cache(
        self,
        prompt_tokens: mx.array,
        reporter: PromptProgressReporter,
        *,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        """
        Set up the KV cache for the next generation, reusing as much of the
        previous cache as possible.

        The LRU key is built from think-normalized tokens so that it matches
        the next turn's query even when clients strip CoT blocks from history.
        The key also ends just before any trailing open <think> token. The GDN
        snapshot is taken at the same boundary so finalize_generation can
        restore to that state.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            reporter: Reporter for reporting prompt processing progress.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The prompt tokens to be used for the next generation.
        """
        num_tokens_to_exclude = max(num_tokens_to_exclude, 1)
        total_prompt_tokens = len(prompt_tokens)
        token_list = prompt_tokens.tolist()

        # If the generator from the previous turn was not exhausted by the caller
        # (early stop after receiving stop_condition), finalize_generation was never
        # called. Do it now before starting the new prefill.
        if self._prefill_checkpoint is not None:
            self.finalize_generation()

        cache, cached_by = self._find_starting_cache(token_list)
        if cache is not None:
            remaining = token_list[cached_by:]
            logger.info(f"[kv] cache hit cached={cached_by}/{total_prompt_tokens}")
        else:
            cache = make_prompt_cache(self.model, self.max_kv_size)
            remaining = token_list

        cached_tokens = total_prompt_tokens - len(remaining)

        reporter.begin(
            is_draft=False,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            prefill_tokens_processed=0,
        )

        # Tokens to prefill: all remaining except the last num_tokens_to_exclude.
        num_tokens_to_exclude = min(num_tokens_to_exclude, len(remaining))
        remaining_arr = mx.array(remaining)
        prefill_tokens = remaining_arr[:-num_tokens_to_exclude]

        # LRU checkpoint: exclude trailing <think> tokens for thinking models
        # so the stored key is a stable prefix of the next turn's query.
        checkpoint_offset = self._checkpoint_offset(token_list)
        checkpoint_idx = total_prompt_tokens - checkpoint_offset
        # Cannot go before what's already cached.
        effective_checkpoint = max(checkpoint_idx, cached_tokens)
        # Position within prefill_tokens (0-based relative to cached_tokens).
        checkpoint_end = min(effective_checkpoint - cached_tokens, len(prefill_tokens))

        with mx.stream(generation_stream):
            try:
                if self.draft_model is not None:
                    draft_cache = cache[len(self.model.layers) :]
                    self._prefill(
                        model=self.draft_model,
                        cache=draft_cache,
                        tokens=prefill_tokens,
                        reporter=reporter,
                        is_draft=True,
                    )

                main_cache = cache[: len(self.model.layers)]
                phase1 = prefill_tokens[:checkpoint_end]
                phase2 = prefill_tokens[checkpoint_end:]

                # Phase 1: prefill up to checkpoint.
                if phase1.size > 0:
                    self._prefill(
                        model=self.model,
                        cache=main_cache,
                        tokens=phase1,
                        reporter=reporter,
                        is_draft=False,
                    )

                # GDN snapshot at checkpoint boundary (before <think>).
                # KV layers are trimmed back to this point in finalize_generation();
                # ArraysCache (GDN) layers are restored from this snapshot in-place.
                gdn_snapshot = [
                    copy.deepcopy(c) if isinstance(c, ArraysCache) else None
                    for c in cache
                ]
                # LRU key in normalized space; bisect maps the checkpoint boundary.
                norm_tokens, norm_orig = self._normalize_think_tokens(token_list)
                if len(norm_tokens) < len(token_list):
                    norm_cp = bisect.bisect_left(norm_orig, effective_checkpoint)
                    lru_key = norm_tokens[:norm_cp]
                else:
                    lru_key = token_list[:effective_checkpoint]
                self._prefill_checkpoint = _PrefillCheckpoint(
                    gdn_snapshot=gdn_snapshot,
                    lru_key=lru_key,
                    kv_len=effective_checkpoint,
                )

                # Phase 2: prefill from checkpoint to end (the <think>\n tokens).
                if phase2.size > 0:
                    self._prefill(
                        model=self.model,
                        cache=main_cache,
                        tokens=phase2,
                        reporter=reporter,
                        is_draft=False,
                        progress_offset=checkpoint_end,
                    )
            except StopPromptProcessing:
                if (
                    self._prefill_checkpoint is None
                    and self._prev_gdn_snapshot is not None
                ):
                    # Cancelled before checkpoint: roll back self.cache to the
                    # previous turn's state and re-insert into LRU.
                    kv_layer = next(
                        (c for c in self.cache if isinstance(c, KVCache)), None
                    )
                    n_to_trim = (kv_layer.offset - self._prev_kv_len) if kv_layer else 0
                    self._restore_and_insert(
                        self._prev_gdn_snapshot, self._prev_lru_key, n_to_trim
                    )
                    logger.info(
                        "[kv] prefill cancelled, rolled back to previous checkpoint"
                    )
                raise

        reporter.finish(is_draft=False)

        # Give the cache directly to generation (no KV deepcopy).
        # finalize_generation() will restore the clean state afterwards.
        self.cache = cache
        self.tokens = prompt_tokens
        logger.info(
            f"[kv] prefill done tokens={total_prompt_tokens} {_vram_str(cache)}"
        )

        return prompt_tokens[-num_tokens_to_exclude:]

    def _restore_and_insert(
        self, gdn_snapshot: list, lru_key: list, n_to_trim: int
    ) -> None:
        """Restore GDN layers from snapshot, trim KV layers, and insert into LRU."""
        for c, snap in zip(self.cache, gdn_snapshot):
            if snap is not None:
                c.cache = snap.cache
            else:
                c.trim(n_to_trim)
        self._lru.insert_cache("main", lru_key, self.cache)
        self._prefill_checkpoint = None

    def finalize_generation(self) -> None:
        """Restore cache to post-prefill state and insert into LRU.

        Called after stream_generate completes (or is cancelled). Trims KV layers
        back to the prefill boundary and restores ArraysCache (GDN) layers from the
        snapshot taken in update_cache. The resulting clean cache is stored in the
        LRU for next-turn prefix reuse.
        """
        if self._prefill_checkpoint is None or self.tokens is None:
            return
        cp = self._prefill_checkpoint
        n_to_trim = len(self.tokens) - cp.kv_len
        # Preserve prev-checkpoint before _restore_and_insert clears it.
        self._prev_gdn_snapshot = cp.gdn_snapshot
        self._prev_kv_len = cp.kv_len
        self._prev_lru_key = cp.lru_key
        self._restore_and_insert(cp.gdn_snapshot, cp.lru_key, n_to_trim)
        kv_offset = next((c.offset for c in self.cache if isinstance(c, KVCache)), -1)
        lru_size_after = len(self._lru)
        logger.info(
            f"[kv] finalize done key_len={len(cp.lru_key)} kv_offset={kv_offset}"
            f" lru_size={lru_size_after} {_vram_str(self.cache)}"
        )

    def record_generated_token(self, token):
        """
        Add the generated token to the token list, so that we can map the token to the KV cache.
        """
        self.tokens = mx.concat([self.tokens, mx.array([token])])


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
        # value: (cache_snapshot, image_end_index: int, prefix_hash: int, block_lengths: tuple)
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
        # Block lengths can change when dynamic resolution re-pads all images in
        # a batch to the largest dimensions (e.g. 300→672 tokens when a larger
        # second image is added). Extra padding tokens are semantically neutral
        # and will be processed by the partial-hit prefill from stored_end_idx
        # onwards. Only invalidate when the gap contains non-pad tokens,
        # indicating a real content change.
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

        Abstracts the hash computation and key construction so callers do not
        need to repeat this logic for both the full-miss and partial-hit paths.

        Args:
            hash_chain:        Full image hash chain for this turn.
            offset:            Number of already-cached image blocks. Zero for
                               a full miss; ``partial_depth`` for a partial hit.
                               Checkpoint at index ``i`` is stored under
                               ``hash_chain[:offset + i + 1]``.
            block_checkpoints: List of ``image_end_index`` values (int), one per
                               newly processed image block.
            input_ids_flat:    Full VLM token sequence as a flat list.
                               Used to compute prefix hashes at each boundary.
            block_lengths:     Image block token counts for the full turn, used
                               as a structural staleness check on the next turn.
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
        """
        Remove a stale image checkpoint.

        Called when the prefix hash of an existing checkpoint no longer matches
        the current conversation, indicating that the KV positions stored in the
        snapshot correspond to a different conversation and must not be reused.
        """
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
        # Use Counter so that an image sent N times with the same hash keeps
        # (N - checkpoint_count) extra copies instead of being dropped entirely.
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
