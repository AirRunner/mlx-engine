import logging
import sys
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

from mlx_engine.utils.prompt_progress_reporter import (
    PromptProgressReporter,
    StopPromptProcessing,
)


PROMPT_PROCESSING_CHUNK_SIZE = 2048


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


class CacheWrapper:
    """
    Wrapper class for the MLX LM cache to maintain an in-memory cache
    """

    def __init__(
        self,
        model: nn.Module,
        max_kv_size: Optional[int],
        *,
        verbose: bool = False,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        chunk_size: int,
    ):
        """
        Initialize the CacheWrapper.

        Args:
            model (nn.Module): The model to be cached.
            max_kv_size (Optional[int]): Maximum size of the key-value cache.
            chunk_size (int): Number of tokens per prefill chunk.
        """
        # utilize a simple ordered list of tokens processed so far for cache invalidation checking
        self.tokens: Optional[mx.array] = None
        self.cache: List[Any] = make_prompt_cache(model, max_kv_size)
        self.model = model
        self.draft_model: Optional[nn.Module] = None
        self.max_kv_size = max_kv_size
        self.verbose = verbose
        self.kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self.chunk_size = chunk_size
        self._image_store: Optional[ImageCheckpointStore] = None

    def _get_num_tokens_in_cache(self) -> int | None:
        """
        Get the number of tokens in the cache.

        Returns:
            int | None: The number of tokens in the cache, or None if the size cannot be determined.
        """
        for c in self.cache:
            if hasattr(c, "offset"):
                return c.offset
        return None

    @staticmethod
    def _find_common_prefix(
        current_tokens: mx.array, prompt_tokens: mx.array, num_tokens_to_exclude: int
    ) -> int:
        """
        Determine the common prefix length between the current tokens and the prompt tokens.

        Args:
            current_tokens (mx.array): The cached tokens (self.tokens).
            prompt_tokens (mx.array): The prompt tokens.
            num_tokens_to_exclude (int): The minimum length of the remaining prompt tokens array.

        Returns:
            int: The length of the common prefix.
        """
        prompt_tokens = prompt_tokens
        current_tokens = current_tokens
        # Find the minimum length between the two arrays
        min_length = min(len(current_tokens), len(prompt_tokens))

        # Compare elements up to the minimum length
        mask = prompt_tokens[:min_length] == current_tokens[:min_length]

        # Find the index where the first mismatch occurs
        if mx.any(mask == False):  # noqa E712
            common_length = int(mx.argmax(mask == False))  # noqa E712
        else:
            common_length = int(min_length)

        # Ensure that the prompt is at least num_tokens_to_exclude long
        uncached_prompt_tokens_length = len(prompt_tokens[common_length:])
        length_adjustment = max(
            0, num_tokens_to_exclude - uncached_prompt_tokens_length
        )
        common_length = max(common_length - length_adjustment, 0)
        return common_length

    def _get_unprocessed_tokens(
        self, prompt_tokens: mx.array, num_tokens_to_exclude: int
    ):
        """
        Get the unprocessed tokens from the prompt.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The unprocessed tokens.
        """
        if self.tokens is None:
            self.tokens = prompt_tokens
            return self.tokens

        # Find common KV between the last generation and the current prompt
        common_prefix = self._find_common_prefix(
            self.tokens, prompt_tokens, num_tokens_to_exclude
        )

        # Trim the cache if the common prefix is shorter than the current cache
        num_tokens_in_cache = self._get_num_tokens_in_cache()
        if num_tokens_in_cache is None:
            logger.warning(
                "Could not determine the number of tokens in the cache, clearing the cache."
            )
            self.cache = make_prompt_cache(self.model, self.max_kv_size)
            self.tokens = prompt_tokens
            return self.tokens
        num_tokens_to_trim = num_tokens_in_cache - common_prefix
        if num_tokens_to_trim > 0:
            if not can_trim_prompt_cache(self.cache):
                logger.warning(
                    f"Tried to trim '{num_tokens_to_trim}' tokens from the prompt cache, but could not: Cache is not trimmable. Clearing the cache instead."
                )
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                return self.tokens
            tokens_trimmed = trim_prompt_cache(self.cache, num_tokens_to_trim)
            if tokens_trimmed != num_tokens_to_trim:
                # If we trimmed fewer tokens than expected, the cache is invalid
                logger.error(
                    f"Tokens trimmed from cache ({tokens_trimmed}) is less than expected ({num_tokens_to_trim}). Clearing the cache."
                )
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                return self.tokens
            logger.info(f"Trimmed {num_tokens_to_trim} tokens from the prompt cache")

        # Keep track of the prompt tokens
        self.tokens = prompt_tokens

        if self.verbose:
            print(f"Common prefix length: {common_prefix}", file=sys.stderr)
            print(f"Trimmed tokens: {num_tokens_to_trim}", file=sys.stderr)

        # All of the common tokens are now in the cache, so we can return the remaining tokens that still need to be processed
        return prompt_tokens[common_prefix:]

    def _prefill(
        self,
        model,
        cache,
        tokens,
        reporter: PromptProgressReporter,
        is_draft: bool,
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
        num_processed = 0

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
                num_tokens_in_cache = self._get_num_tokens_in_cache()
                if num_tokens_in_cache is not None and num_tokens_in_cache > len(
                    self.tokens
                ):
                    logger.warning(
                        "The number of tokens in the cache is greater than the number of prompt tokens. This is unexpected. Clearing the cache."
                    )
                    num_tokens_in_cache = None
                if num_tokens_in_cache is None:
                    self.cache = make_prompt_cache(self.model, self.max_kv_size)
                    self.tokens = None
                else:
                    # Remember which tokens were processed so far, so that we can continue processing at a later point
                    self.tokens = self.tokens[:num_tokens_in_cache]
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
        self.cache: List[Any] = make_prompt_cache(self.model)
        if draft_model is not None:
            self.cache += make_prompt_cache(draft_model)
        self.draft_model = draft_model

    def unset_draft_model(self):
        """Removes the draft model from the cache if one exists."""
        if self.draft_model is None:
            return
        self.draft_model = None
        self.cache = self.cache[: len(self.model.layers)]

    def update_cache(
        self,
        prompt_tokens: mx.array,
        reporter: PromptProgressReporter,
        *,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        """
        Set up the KV cache for the next generation.
        Re-use as much of the KV cache from the previous generation as possible.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            reporter: Reporter for reporting prompt processing progress.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The prompt tokens to be used for the next generation.
        """
        num_tokens_to_exclude = max(num_tokens_to_exclude, 1)
        total_prompt_tokens = len(prompt_tokens)
        prompt_tokens = self._get_unprocessed_tokens(
            prompt_tokens, num_tokens_to_exclude
        )
        cached_tokens = total_prompt_tokens - len(prompt_tokens)

        # Report begin
        reporter.begin(
            is_draft=False,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            prefill_tokens_processed=0,
        )

        # Prefill the cache with the non-excluded prompt tokens
        num_tokens_to_exclude = min(num_tokens_to_exclude, len(prompt_tokens))
        prefill_tokens = prompt_tokens[:-num_tokens_to_exclude]

        with mx.stream(generation_stream):
            if self.draft_model is not None:
                # Fill draft model cache
                draft_cache = self.cache[len(self.model.layers) :]
                self._prefill(
                    model=self.draft_model,
                    cache=draft_cache,
                    tokens=prefill_tokens,
                    reporter=reporter,
                    is_draft=True,
                )
            # Fill main model cache
            main_cache = self.cache[: len(self.model.layers)]
            self._prefill(
                model=self.model,
                cache=main_cache,
                tokens=prefill_tokens,
                reporter=reporter,
                is_draft=False,
            )

        # Report finish
        reporter.finish(is_draft=False)

        # Return the tokens that must still be processed outside of the cache
        non_prefill_tokens = prompt_tokens[-num_tokens_to_exclude:]
        return non_prefill_tokens

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
    # TODO: ViT-based models (e.g. Qwen3-VL) use uniform placeholder IDs for image tokens,
    # so block_lengths is insufficient as a cache key. Incorporate image content (e.g. sha256
    # of raw bytes) to avoid false cache hits across different images of the same resolution.
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
        cache_snapshot,
        image_end_index: int,
        prefix_hash: int,
        block_lengths: tuple = (),
    ) -> None:
        """
        Persist a KV snapshot taken right after an image block.

        Args:
            key:             Tuple of per-image SHA-256 hex digests up to and
                             including this block, e.g. (hash1,) or (hash1, hash2).
            cache_snapshot:  Deep-copied KV cache list.
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
            cache_snapshot,
            image_end_index,
            prefix_hash,
            block_lengths,
        )
        logger.info(
            f"[kv-image] checkpoint saved depth={len(key)} index={image_end_index}"
        )

    def get_image_checkpoint(self, key: tuple):
        """
        Return ``(cache_snapshot, image_end_index, prefix_hash, block_lengths)``
        for *key*, or None.
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

        _, stored_end_idx, stored_prefix_hash, stored_block_lengths = entry
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
        """Compute per-block prefix hashes and persist KV checkpoints.

        Abstracts the hash computation and key construction so callers do not
        need to repeat this logic for both the full-miss and partial-hit paths.

        Args:
            hash_chain:        Full image hash chain for this turn.
            offset:            Number of already-cached image blocks. Zero for
                               a full miss; ``partial_depth`` for a partial hit.
                               Checkpoint at index ``i`` is stored under
                               ``hash_chain[:offset + i + 1]``.
            block_checkpoints: List of ``(image_end_index, cache_snapshot)``
                               pairs, one per newly processed image block.
            input_ids_flat:    Full VLM token sequence as a flat list.
                               Used to compute prefix hashes at each boundary.
            block_lengths:     Image block token counts for the full turn, used
                               as a structural staleness check on the next turn.
        """
        for i, (end_idx, snap) in enumerate(block_checkpoints):
            pfx_hash = hash(tuple(input_ids_flat[:end_idx]))
            self.save_image_checkpoint(
                hash_chain[: offset + i + 1],
                snap,
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
        Return (key, cache_snapshot, image_end_index) for the longest prefix of
        *hash_chain* that has a stored checkpoint, or None if no prefix matches.
        """
        for depth in range(len(hash_chain), 0, -1):
            key = hash_chain[:depth]
            entry = self._image_checkpoints.get(key)
            if entry is not None:
                return key, entry[0], entry[1]
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
        known = set(best_key)
        ordered_imgs = [hash_to_img[h] for h in best_key]
        ordered_hashes = list(best_key)

        # New images: in the received set but absent from the checkpoint, bridge order.
        new_imgs = [img for img, h in zip(images_b64, image_hashes) if h not in known]
        new_hashes = [h for h in image_hashes if h not in known]

        reordered_imgs = ordered_imgs + new_imgs
        reordered_hashes = ordered_hashes + new_hashes

        if reordered_hashes != image_hashes:
            logger.info(
                f"[kv-image] reordered {len(images_b64)} images to chronological order"
            )

        return reordered_imgs, reordered_hashes
