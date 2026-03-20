import copy
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
from mlx_lm.server import LRUPromptCache

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


# TODO: extract a shared ABC for CacheWrapper and VisionCacheWrapper.
class VisionCacheWrapper:
    """
    LRU-backed KV cache for VisionModelKit requests.

    Text layer (self._lru)
    ----------------------
    Parallels CacheWrapper but uses a multi-slot LRU instead of a single-slot
    cache.  VisionModelWrapper's KV cache is non-trimmable (ArraysCache), so
    the trim-based prefix-reuse approach of CacheWrapper is unavailable.
    Instead, a checkpoint is saved at the end of every prefill and fetched on
    the next request, allowing arbitrarily long prefix matches across turns.

    Image layer (self._image_checkpoints)
    --------------------------------------
    Stores KV snapshots taken right after each image block during prefill,
    keyed by a tuple of per-image SHA-256 hashes: (hash(img1),),
    (hash(img1), hash(img2)), etc.  On the next turn, the deepest matching
    prefix is restored and only the new text tokens are prefilled, skipping
    the vision tower entirely.

    Requires the mlx-vlm model to expose image_end_index via
    InputEmbeddingsFeatures.

    Lifecycle
    ---------
    - Instantiated once in VisionModelKit._full_model_init(), preserved across
      subsequent _reset_for_prediction() calls so that LRU snapshots survive.
    - clear_text() is called before image requests to drop stale text-only
      snapshots.  Image checkpoints are intentionally preserved across turns.
    """

    def __init__(self, model, tokenizer) -> None:
        """
        Args:
            model:     VisionModelWrapper, used for prefill forward passes.
            tokenizer: Model tokenizer, inspected for has_thinking /
                       think_start_id to compute the checkpoint offset.
        """
        self._lru: LRUPromptCache = LRUPromptCache()
        # tuple[str, ...] → (cache_snapshot, image_end_index: int)
        self._image_checkpoints: dict = {}
        self._model = model
        self._tokenizer = tokenizer

    def clear_text(self) -> None:
        """
        Drop text-only KV snapshots and release their Metal buffers.

        Image checkpoints are intentionally left intact so that a cached image
        can still be reused on future turns.
        """
        self._lru = LRUPromptCache()
        mx.clear_cache()

    def clear(self) -> None:
        """Drop all KV snapshots (text-only and image). Callers that only need
        to flush text-only state should prefer clear_text()."""
        self._lru = LRUPromptCache()
        self._image_checkpoints = {}
        mx.clear_cache()

    def save_image_checkpoint(
        self, key: tuple, cache_snapshot, image_end_index: int, prefix_hash: int
    ) -> None:
        """
        Persist a KV snapshot taken right after an image block.

        Args:
            key:             Tuple of per-image SHA-256 hex digests up to and
                             including this block, e.g. (hash1,) or (hash1, hash2).
            cache_snapshot:  Deep-copied KV cache list from VisionModelWrapper.
            image_end_index: First text-token index after this image block.
            prefix_hash:     Python hash of the VLM token ids
                             ``input_ids[0, :image_end_index]`` at save time.
                             Used to detect stale checkpoints from a different
                             conversation that happens to share the same images.
        """
        self._image_checkpoints[key] = (cache_snapshot, image_end_index, prefix_hash)
        logger.info(
            f"[kv-image] checkpoint saved depth={len(key)} index={image_end_index}"
        )

    def get_image_checkpoint(self, key: tuple):
        """
        Return ``(cache_snapshot, image_end_index, prefix_hash)`` for *key*, or None.
        """
        return self._image_checkpoints.get(key)

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

    def prefill_text_after_image(
        self,
        base_cache,
        vlm_text_tokens: mx.array,
        reporter: PromptProgressReporter,
    ) -> tuple:
        """
        Prefill text tokens that follow the image block, using the restored
        image KV snapshot as the starting cache state.

        Runs in PROMPT_PROCESSING_CHUNK_SIZE-token blocks with eager eval and
        mx.clear_cache() between chunks, following the same pattern as update_cache().

        Args:
            base_cache:       Deep-copied image KV snapshot (will be mutated).
            vlm_text_tokens:  input_ids[image_end_index:], shape (1, T).
            reporter:         Progress reporter for prefill UI feedback.

        Returns:
            (base_cache, last_token) where base_cache has been filled with
            vlm_text_tokens[:-1] and last_token (shape (1,)) is the single
            token to pass as prompt to stream_generate.

        Raises:
            StopPromptProcessing: if the user cancels during prefill.
        """
        total = vlm_text_tokens.shape[1]

        reporter.begin(
            is_draft=False,
            cached_tokens=0,
            total_prompt_tokens=total,
            prefill_tokens_processed=0,
        )

        processed = 0
        for start in range(0, total - 1, PROMPT_PROCESSING_CHUNK_SIZE):
            end = min(start + PROMPT_PROCESSING_CHUNK_SIZE, total - 1)
            self._model.language_model(vlm_text_tokens[:, start:end], cache=base_cache)
            mx.eval([c.state for c in base_cache])
            mx.clear_cache()
            processed += end - start
            if not reporter.update(is_draft=False, prefill_tokens_processed=processed):
                raise StopPromptProcessing

        reporter.finish(is_draft=False, prefill_tokens_processed=processed)

        # Return the last token as a 1-D array for stream_generate
        last_token = vlm_text_tokens[0, -1:]
        return base_cache, last_token

    def _checkpoint_offset(self, tokens: list) -> int:
        """
        Number of trailing tokens excluded from the checkpoint.

        For thinking models the checkpoint is saved just *before* the
        <think> token so that stream_generate always regenerates it —
        required to keep the thinking-mode logit bias active.
        For all other models the offset is 1 (standard last-token exclusion).
        """
        if getattr(self._tokenizer, "has_thinking", False):
            think_id = self._tokenizer.think_start_id
            for i in range(1, min(11, len(tokens))):
                if tokens[-i] == think_id:
                    return i + 1
        return 1

    def update_cache(
        self,
        prompt_tokens: list,
        reporter: PromptProgressReporter,
    ) -> tuple:
        """
        Set up the KV cache for the next text-only generation step.

        The caller must pass *prompt_tokens* from LM Studio's stable
        tokenisation (not the VLM processor's re-encoded ids, which vary
        across turns and cause spurious cache misses).

        Phase 1: prefill all tokens except the last *checkpoint_offset*
                 ones, in PROMPT_PROCESSING_CHUNK_SIZE-token blocks with
                 eager eval + mx.clear_cache() between chunks.
        Checkpoint: deep-copy the populated cache into the LRU so the next
                    request can start from this prefix.
        Phase 2: the remaining *checkpoint_offset* tokens are returned to
                 the caller for processing by stream_generate.

        Args:
            prompt_tokens: Stable list[int] token sequence from LM Studio.
            reporter:      Progress reporter for prefill UI feedback.

        Returns:
            (cache, rest_tokens): a pre-populated KV cache list and an
            mx.array of the tokens still to be processed by stream_generate.

        Raises:
            StopPromptProcessing: if the user cancels during Phase 1.
        """
        offset = self._checkpoint_offset(prompt_tokens)

        base_cache, rest = self._lru.fetch_nearest_cache("model", prompt_tokens)
        if base_cache is None:
            base_cache = make_prompt_cache(self._model.language_model)
            rest = prompt_tokens
        else:
            cached = len(prompt_tokens) - len(rest)
            logger.info(
                f"[kv-seq] cache hit: {cached}/{len(prompt_tokens)} tokens cached"
            )

        # Prefill rest[:-offset] in chunks
        phase1_len = len(rest) - offset
        cached_tokens = len(prompt_tokens) - len(rest)

        reporter.begin(
            is_draft=False,
            cached_tokens=cached_tokens,
            total_prompt_tokens=len(prompt_tokens),
            prefill_tokens_processed=0,
        )
        processed = 0
        if phase1_len > 0:
            phase1_array = mx.array(rest[:phase1_len])
            for i in range(0, phase1_len, PROMPT_PROCESSING_CHUNK_SIZE):
                chunk = phase1_array[i : i + PROMPT_PROCESSING_CHUNK_SIZE][None]
                self._model(chunk, cache=base_cache)
                mx.eval([c.state for c in base_cache])
                mx.clear_cache()
                processed += chunk.shape[1]
                if not reporter.update(
                    is_draft=False,
                    prefill_tokens_processed=cached_tokens + processed,
                ):
                    raise StopPromptProcessing
        reporter.finish(
            is_draft=False,
            prefill_tokens_processed=cached_tokens + phase1_len,
        )

        checkpoint_key = prompt_tokens[:-offset]
        self._lru.insert_cache(
            "model", checkpoint_key, copy.deepcopy(base_cache), checkpoint=True
        )
        logger.info(f"[kv-seq] checkpoint saved at len={len(checkpoint_key)}")

        # Phase 2 tokens: processed by stream_generate
        return base_cache, mx.array(rest[-offset:])

    def save_post_prefill_snapshot(self, prompt_tokens: list, cache) -> None:
        """
        Save a KV snapshot to the text LRU after an image-hit or partial-hit prefill.

        Called by generate.py after prefill_text_after_image() or
        prefill_with_partial_cache() so that subsequent text turns can
        restore the full conversation state without re-prefilling from scratch.

        Uses the same checkpoint offset as update_cache() to exclude trailing
        think tokens from the key.

        Args:
            prompt_tokens: LM Studio stable token list for the current request.
            cache:         KV cache after all text has been prefilled (will be
                           deep-copied before insertion into the LRU).
        """
        offset = self._checkpoint_offset(prompt_tokens)
        checkpoint_key = prompt_tokens[:-offset]
        self._lru.insert_cache(
            "model", checkpoint_key, copy.deepcopy(cache), checkpoint=True
        )
        logger.info(f"[kv-seq] post-image snapshot saved len={len(checkpoint_key)}")

    def fetch_continuation_cache(self, prompt_tokens: list) -> tuple:
        """
        Look up the text LRU for a continuation snapshot saved by a previous
        image-hit or partial-hit prefill (save_post_prefill_snapshot).

        Returns:
            (cache | None, remaining_tokens: list)
            cache: deep-copied KV snapshot on hit, None on miss.
            remaining_tokens: tokens not yet in the cache (empty on full hit).
        """
        return self._lru.fetch_nearest_cache("model", prompt_tokens)

    def fetch_pre_image_cache(
        self, vlm_input_ids_flat: list, img_tok: int, vid_tok: int
    ) -> tuple:
        """
        Look up the text LRU using the VLM token prefix that precedes the
        first image block.  On a hit the returned cache can be injected into
        VisionModelWrapper as _pre_populated_cache so the vision tower runs on
        top of already-prefilled text KV instead of a fresh cache.

        Works for any model that exposes image_token_index on its config.
        Falls back gracefully (returns None cache) when the LRU has no match.

        Args:
            vlm_input_ids_flat: flat list of VLM processor token ids for the
                                 current request (input_ids[0].tolist()).
            img_tok:             model config image_token_index.
            vid_tok:             model config video_token_index (may equal img_tok).

        Returns:
            (cache | None, remaining_prefix: list, first_image_start: int)
            cache:               deep-copied KV snapshot or None on miss.
            remaining_prefix:    VLM token ids between the LRU boundary and
                                 first_image_start; must be prefilled before use.
            first_image_start:   index of the first image token in the flat id
                                 list, or len(vlm_input_ids_flat) if none found.
        """
        # Locate the first image token
        first_image_start = len(vlm_input_ids_flat)
        for i, tok in enumerate(vlm_input_ids_flat):
            if tok == img_tok or tok == vid_tok:
                first_image_start = i
                break

        if first_image_start == 0:
            # Image starts at position 0: no pre-image text to cache.
            return None, [], 0

        vlm_prefix = vlm_input_ids_flat[:first_image_start]
        cache, rest = self._lru.fetch_nearest_cache("model", vlm_prefix)
        if cache is None:
            return None, vlm_prefix, first_image_start

        cached_len = len(vlm_prefix) - len(rest)
        logger.info(
            f"[kv-seq] pre-image cache hit {cached_len}/{len(vlm_prefix)} tokens"
        )
        return cache, list(rest), first_image_start
