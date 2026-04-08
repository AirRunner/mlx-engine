import hashlib
from contextlib import contextmanager
import mlx.core as mx
import uuid
from mlx_engine.model_kit.batched_model_kit import (
    BatchedGenerationResponse,
    BatchedModelKit,
)
from mlx_engine.model_kit.batched_model_kit_types import RequestCancelled
from typing import Iterator, List, Optional
import json
import logging
from pathlib import Path
import sys
import threading

from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params
from mlx_lm.generate import stream_generate
from mlx_lm.utils import load as mlx_lm_load
from mlx_lm.models.cache import (
    KVCache,
    LRUPromptCache,
    make_prompt_cache,
)

from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit
from mlx_engine.utils.token import Token
from mlx_engine.utils.eot_tokens import sanitize_eos_tokens
from mlx_engine.utils.top_logprobs import summarize_top_logprobs
from mlx_engine.stop_string_processor import (
    StopStringProcessorResult,
)
from mlx_engine.utils.generation_result import (
    GenerationStopCondition,
    GenerationResult,
    construct_user_cancelled_result,
)
from mlx_engine.utils.set_seed import set_seed
from mlx_engine.utils.speculative_decoding import (
    determine_draft_model_for_generation,
    configure_num_draft_tokens_in_generate_args,
    is_speculative_decoding_supported,
    SpeculativeDecodingNotSupportedError,
)
from outlines.processors.structured import JSONLogitsProcessor
from mlx_engine.utils.outlines_transformer_tokenizer import OutlinesTransformerTokenizer
from mlx_engine.cache_wrapper import (
    validate_prefill_step_size,
    image_block_boundaries,
    image_block_lengths,
    ImageCheckpointStore,
)
from mlx_engine.utils.prompt_progress_reporter import (
    BatchedMlxLmReporterAdapter,
    LoggerReporter,
    PromptProgressReporter,
    DefaultPromptProgressReporter,
    MlxLmReporterAdapter,
    StopPromptProcessing,
)
from mlx_engine.utils.generation_helpers import (
    setup_repetition_penalty,
    setup_logits_processors,
    create_sampler,
    validate_top_logprobs,
    create_stop_string_processor,
    process_stop_string_check,
    should_yield_token,
)

MAX_TOP_LOGPROBS = 10


logger = logging.getLogger(__name__)


def _handle_stop_string_detected(
    tokenizer,
    stop_string_processor_result: StopStringProcessorResult,
    text: str,
    token_buffer: List[Token],
    top_logprobs_buffer: List[List[Token]],
) -> GenerationResult:
    """
    Helper method to Handle completion of text generation when a stop string is
    encountered.

    Args:
        tokenizer: The tokenizer instance
        stop_string_processor_result: Result from stop string processor
        text: Current generated text
        token_buffer: Buffer of generated tokens
        top_logprobs_buffer: Buffer of token probabilities

    Returns:
        GenerationResult: Final generation result including stop condition
    """
    # Finalize detokenizer to get remaining text
    detokenizer = tokenizer.detokenizer
    detokenizer.finalize()
    text += detokenizer.last_segment

    # Process stop string by trimming text segment where it begins
    stop_string = stop_string_processor_result.stop_string
    stop_string_start_pos = text.find(stop_string)

    if stop_string_start_pos != -1:
        text = text[:stop_string_start_pos]
    else:
        # this is known to happen when the eos token is a stop string
        sys.stderr.write(
            f"[mlx-engine] Stop string '{stop_string}' not found in final text segment, "
            "even though a full stop was detected. Not trimming final segment."
        )

    stop_condition = GenerationStopCondition(
        stop_reason="stop_string",
        stop_string=stop_string,
        stop_tokens=stop_string_processor_result.stop_tokens,
    )

    return GenerationResult(
        text=text,
        tokens=token_buffer,
        stop_condition=stop_condition,
        top_logprobs=top_logprobs_buffer,
    )


def load_model(
    model_path: str | Path,
    *,
    vocab_only: bool = False,
    max_kv_size: int | None = 4096,
    max_seq_nums: int | None = 4,
    seed: int | None = None,
    trust_remote_code: bool = False,
    kv_bits: Optional[int] = None,
    kv_group_size: Optional[int] = None,
    quantized_kv_start: Optional[int] = None,
    prefill_step_size: Optional[int] = None,
) -> ModelKit | VisionModelKit:
    """
    Load a language model or vision-language model from the specified path.

    This function determines the model type based on the config.json file in the model directory
    and initializes either a standard language model or a vision-language model accordingly.

    Args:
        model_path (str | Path): Path to the model directory containing model files and config.json.
        vocab_only (bool): Only load vocabulary/tokenizer, not the full model.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        max_seq_nums (int): The maximum number of parallel generation requests that can be worked on
        seed (Optional[int]): Random seed for reproducible generation. If provided, sets the
            random seed for all subsequent generation operations with this model.
        trust_remote_code (bool): Whether to allow loading of remote code during model initialization.
        kv_bits (Optional[int]): Number of bits for KV cache quantization.
        kv_group_size (Optional[int]): Group size for KV cache quantization.
        quantized_kv_start (Optional[int]): Step to begin KV cache quantization when enabled.
        prefill_step_size (Optional[int]): Number of tokens to process per prefill chunk.
            Defaults to PROMPT_PROCESSING_CHUNK_SIZE when None.

    Returns:
        ModelKit | VisionModelKit: An initialized model instance:
            - ModelKit: for text-only models and vision models with vision add-on support
            - VisionModelKit: for vision models that are not yet supported by ModelKit

    Raises:
        FileNotFoundError: If config.json is not found in the specified model path
        json.JSONDecodeError: If config.json exists but contains invalid JSON
        ValueError: If the model configuration is invalid or unsupported
    """
    set_seed(seed)
    prefill_step_size = validate_prefill_step_size(prefill_step_size)
    model_path = Path(model_path)
    config_json = json.loads((model_path / "config.json").read_text())
    model_type = config_json.get("model_type", None)
    parallel_requested = max_seq_nums is not None and max_seq_nums > 1

    def warn_if_parallel(reason: str) -> None:
        """Helper to warn about batching not being supported, only if parallel was requested."""
        if parallel_requested:
            logger.warning(
                f"max_concurrent_predictions={max_seq_nums} was specified, but {reason}. "
                f"The model will process requests sequentially."
            )

    # Determine which model kit to use based on model capabilities and configuration.
    # The decision tree is:
    # 1. VisionModelKit: for vision models not yet supported by ModelKit's vision implementation
    # 2. BatchedModelKit: for models that support continuous batching (can process multiple requests concurrently)
    # 3. ModelKit: fallback for all other cases (sequential processing)
    if "vision_config" in config_json and not ModelKit.is_supported_vision_arch(
        model_type
    ):
        if any([kv_bits, kv_group_size, quantized_kv_start]):
            raise ValueError(
                "MLX vision models do not currently support KV cache quantization"
            )
        if parallel_requested:
            raise ValueError(
                "numParallelSessions must be 1 for vision models as they do not currently support continuous batching"
            )
        model_kit = VisionModelKit(
            model_path,
            vocab_only,
            trust_remote_code,
            prefill_step_size=prefill_step_size,
        )
    else:
        # For non-vision models or ModelKit-supported vision models, choose between
        # BatchedModelKit (continuous batching) and ModelKit (sequential)
        kv_bits, kv_group_size, quantized_kv_start = get_kv_cache_quantization_params(
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )

        def is_batchable() -> bool:
            # 0. Ensure the load isn't vocab only
            if vocab_only:
                return False
            # 1. All cache layers must support merge
            model, _ = mlx_lm_load(model_path, lazy=True)
            cache_has_merge_attr = all(
                hasattr(c, "merge") for c in make_prompt_cache(model)
            )
            del model
            if not cache_has_merge_attr:
                warn_if_parallel(
                    "this model architecture does not support continuous batching"
                )
                return False
            # 2. KV cache quantization is not compatible with batching yet
            if kv_bits is not None:
                warn_if_parallel(
                    "concurrency is not supported with KV Cache Quantization"
                )
                return False
            # 3. Vision models are not compatible with batching yet
            if "vision_config" in config_json:
                if parallel_requested:
                    raise ValueError(
                        "numParallelSessions must be 1 for vision models as they do not currently support continuous batching"
                    )
                return False
            return True

        batchable = is_batchable()
        # If max_seq_nums is set to 1, use ModelKit instead of BatchedModelKit. This gives users an escape hatch,
        # which they could use to enable spec decoding. We can remove this additional restriction once we add
        # spec decoding support to the batched backend
        use_batched_kit = batchable and max_seq_nums != 1

        if use_batched_kit:
            model_kit = BatchedModelKit(
                model_path,
                max_kv_size=max_kv_size,
                max_seq_nums=max_seq_nums,
                prefill_step_size=prefill_step_size,
            )
        else:
            model_kit = ModelKit(
                model_path,
                prefill_step_size=prefill_step_size,
                vocab_only=vocab_only,
                max_kv_size=max_kv_size,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            )
    sanitize_eos_tokens(model_kit)
    model_kit.start()
    return model_kit


def load_draft_model(
    model_kit: ModelKit | VisionModelKit | BatchedModelKit, path: str | Path
) -> None:
    if not is_speculative_decoding_supported(model_kit):
        raise SpeculativeDecodingNotSupportedError(
            "Speculative decoding is not supported for batched MLX models."
        )
    model_kit.load_draft_model(path)


def is_draft_model_compatible(
    model_kit: ModelKit | VisionModelKit | BatchedModelKit, path: str | Path
) -> bool:
    if not is_speculative_decoding_supported(model_kit):
        return False
    return model_kit.is_draft_model_compatible(path)


def unload_draft_model(
    model_kit: ModelKit | VisionModelKit | BatchedModelKit,
) -> None:
    if not is_speculative_decoding_supported(model_kit):
        return
    model_kit.unload_draft_model()


def create_generator(
    model_kit: ModelKit | VisionModelKit | BatchedModelKit,
    prompt_tokens: List[int],
    **kwargs,
) -> Iterator[GenerationResult]:
    """
    Create a generator that streams text generation results from the model.

    This function sets up and manages the text generation process, handling various generation
    parameters, processing callbacks, and managing generation constraints. It supports both
    standard language models and vision-language models.

    Args:
        model_kit (ModelKit | VisionModelKit): The initialized model to use for generation
        prompt_tokens (List[int]): List of token IDs representing the input prompt
        prompt_progress_reporter (Optional[PromptProgressReporter]): Reporter for receiving prompt
            processing progress updates. Reporter methods should return True to continue processing,
            or False to stop generation
        images_b64 (Optional[List[str]]): List of base64-encoded images for vision-language models
        max_image_size (Optional[tuple[int, int]]): Maximum dimensions (width, height) for images.
            Images will be resized to fit within these dimensions while maintaining aspect ratio if
            they exceed this size. If None, no resizing.
        stop_strings (Optional[List[str]]): List of strings that will trigger generation to stop
            when encountered
        top_logprobs (Optional[int]): Number of top token probabilities to return per token
            Must be <= MAX_TOP_LOGPROBS
        repetition_penalty (Optional[float]): Penalty factor for repeated tokens. Higher values
            discourage repetition
        repetition_context_size (Optional[int]): Number of previous tokens to consider for
            repetition penalty. Defaults to 20
        temp (Optional[float]): Temperature for sampling. Higher values increase randomness
        top_p (Optional[float]): Top-p (nucleus) sampling parameter
        top_k (Optional[int]): Top-k sampling parameter
        min_p (Optional[float]): Minimum probability threshold for token sampling
        min_tokens_to_keep (Optional[int]): Minimum number of tokens to keep during sampling
        seed (Optional[int]): Random seed for reproducible generation
        json_schema (Optional[str]): JSON schema for structured output generation
        max_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to 10000000
        speculative_decoding_toggle (Optional[bool]): If not set, use speculative decoding
            if a draft model is loaded. If set to true, draft model must be loaded or else error.
            If set to false, speculative decoding is disabled even if a draft model is loaded.
        num_draft_tokens (Optional[int]): Number of tokens to draft when using speculative decoding
        request_id (Optional[int]): Id associated with the request

    Yields:
        GenerationResult: A named tuple containing:
            - text (str): Generated text segment
            - tokens (List[TokenLogprob]): List of generated tokens with their probabilities
            - top_logprobs (List[List[TokenLogprob]]): Token probability information if requested
            - stop_condition (Optional[GenerationStopCondition]): Information about why
              generation stopped, if applicable

    Raises:
        ValueError: If top_logprobs exceeds MAX_TOP_LOGPROBS or if any parameters are invalid
    """
    if isinstance(model_kit, BatchedModelKit):
        return _batched_generation(model_kit, prompt_tokens, **kwargs)
    return _sequential_generation(model_kit, prompt_tokens, **kwargs)


@contextmanager
def _sequential_gen_abort_handler(
    model_kit: ModelKit | VisionModelKit, request_id: Optional[str]
):
    """
    Acquires the generation lock for sequential generation, with support for cancellation.

    Creates a per-request cancellation event that can be signaled while waiting for the lock
    or during generation.
    """

    cancel_event = threading.Event()
    should_track_request = True
    if request_id is None or request_id == "":
        logger.warning(
            "request_id missing for sequential generation; cancellation by id is disabled"
        )
        should_track_request = False
    else:
        model_kit.pending_requests[request_id] = cancel_event

    try:
        # Try to acquire lock, checking for cancellation while waiting
        while True:
            if cancel_event.is_set() or model_kit.is_shutdown():
                # The request is cancelled. Bypass acquiring the lock and let the generator yield a "user cancelled" result
                yield cancel_event
                return

            if model_kit.generation_lock.acquire(timeout=0.1):
                break

        try:
            yield cancel_event
        finally:
            model_kit.generation_lock.release()
    finally:
        if should_track_request:
            model_kit.pending_requests.pop(request_id, None)


def _get_image_store(model_kit) -> Optional[ImageCheckpointStore]:
    cw = getattr(model_kit, "cache_wrapper", None)
    return getattr(cw, "_image_store", None) if cw else None


def _hash_images(images_b64: list) -> tuple:
    return tuple(hashlib.sha256(img.encode()).hexdigest() for img in images_b64)


def _try_inject_pre_image_cache(
    model_kit: ModelKit,
    input_ids_list: list,
    img_tok: int,
    vid_tok: Optional[int],
) -> tuple:
    """Try to reuse pre-image KV state from CacheWrapper to seed the miss prefill.

    Delegates prefix matching to the LRU in CacheWrapper. Returns a deepcopy of
    the best matching cached prefix so the prefill can start from there instead
    of position 0. Works for both trimmable and non-trimmable caches via the
    LRU shorter-match fallback.

    Returns:
        (cache, n_cached_tokens) or (None, 0) when no reuse is possible.
    """
    cw = getattr(model_kit, "cache_wrapper", None)
    if cw is None:
        return None, 0

    first_img = next(
        (i for i, t in enumerate(input_ids_list) if t == img_tok or t == vid_tok),
        None,
    )
    if first_img is None or first_img == 0:
        return None, 0

    pre_image = input_ids_list[:first_img]
    cache, remaining = cw._lru.fetch_nearest_cache("main", pre_image)

    if cache is not None:
        n_cached = first_img - len(remaining)
        if n_cached > 0:
            logger.info(
                f"[kv-image] pre-image LRU hit: {n_cached}/{first_img} tokens reused"
            )
            return cache, n_cached

    # LRU returned None: no entry, or stored key longer than common prefix with
    # a non-trimmable cache (mixed KV + ArraysCache). Fallback: find common prefix
    # manually and trim KV layers in-place on cw.cache. ArraysCache layers are left
    # at their current position; the error is bounded by the number of diverged
    # tokens (typically a few tens due to context truncation) and is negligible.
    #
    # No deepcopy: cw.cache and the LRU entry are the same Python object. We reset
    # the LRU to release its reference, dropping the ref count back to 1, so VRAM
    # stays at 1x throughout the image prefill.
    cw_tokens = cw.tokens
    if cw_tokens is None or cw.cache is None:
        return None, 0

    cw_list = cw_tokens.tolist()
    common = sum(1 for a, b in zip(cw_list, pre_image) if a == b)
    if common == 0:
        return None, 0

    n_kv = next((c.offset for c in cw.cache if isinstance(c, KVCache)), None)
    if n_kv is None or n_kv == 0:
        return None, 0

    if n_kv > common:
        to_trim = n_kv - common
        for c in cw.cache:
            if isinstance(c, KVCache):
                c.trim(to_trim)

    # Release the LRU's reference (same object as cw.cache) so the trie overhead
    # is freed and ref count stays at 1.
    cw._lru = LRUPromptCache(max_size=1)

    logger.info(
        f"[kv-image] pre-image fallback: {common}/{first_img} tokens reused"
        f" (KV {n_kv}->{common}, GDN approx at {n_kv})"
    )
    return cw.cache, common


def _prefill_modelkit_with_image_checkpoints(
    model_kit: ModelKit,
    input_ids: mx.array,
    embeddings: Optional[mx.array],
    image_boundaries: list,
    reporter: PromptProgressReporter,
    *,
    initial_cache: Optional[list] = None,
    cached_tokens: int = 0,
    base_offset: int = 0,
) -> tuple:
    """Chunked prefill saving a KV snapshot after each image block.

    Args:
        model_kit:        Loaded ModelKit instance.
        input_ids:        Token ids to prefill, shape (seq_len,). May be a suffix
                          when initial_cache is provided (partial hit / pre-image injection).
        embeddings:       Merged embeddings, shape (seq_len, dim). Pass None for
                          pure text suffixes (full hit path): model is called
                          with token ids only.
        image_boundaries: (start, end_exclusive) pairs relative to input_ids.
        reporter:         Progress reporter.
        initial_cache:    Pre-populated KV cache to start from (partial hit /
                          pre-image injection). If None, a fresh cache is created.
        cached_tokens:    Tokens already covered by initial_cache. Used to report
                          accurate progress and as base for total_prompt_tokens.
        base_offset:      Added to each snapshot's image_end_index so stored
                          indices are absolute (relative to the full VLM sequence).

    Returns:
        (cache, block_checkpoints) where block_checkpoints is a list of
        image_end_index values (int) with absolute indices, one per image block.
    """
    total = input_ids.shape[0]
    prefill_len = total - 1  # stream_generate handles the last token

    snap_set = {end for _, end in image_boundaries if end <= prefill_len}
    snap_list = sorted(snap_set)

    cache = (
        initial_cache
        if initial_cache is not None
        else make_prompt_cache(model_kit.model)
    )
    block_checkpoints = []
    chunk_size = model_kit.prefill_step_size

    reporter.begin(
        is_draft=False,
        cached_tokens=cached_tokens,
        total_prompt_tokens=total + cached_tokens,
        prefill_tokens_processed=0,
    )

    processed = 0
    i = 0
    while i < prefill_len:
        end = min(i + chunk_size, prefill_len)
        for snap in snap_list:
            if snap > i and snap <= end:
                end = snap
                break

        chunk_ids = input_ids[i:end][None]
        if embeddings is not None:
            model_kit.model(
                chunk_ids, cache=cache, input_embeddings=embeddings[i:end][None]
            )
        else:
            model_kit.model(chunk_ids, cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        processed += end - i

        if end in snap_set:
            block_checkpoints.append(end + base_offset)

        if not reporter.update(is_draft=False, prefill_tokens_processed=processed):
            raise StopPromptProcessing

        i = end

    reporter.finish(is_draft=False, prefill_tokens_processed=total)
    return cache, block_checkpoints


def _process_modelkit_image_cache(
    model_kit: ModelKit,
    image_store: ImageCheckpointStore,
    prompt_tokens,
    images_b64: list,
    max_image_size,
    generate_args: dict,
    reporter: PromptProgressReporter,
) -> tuple:
    """Handle image prompts with KV cache for VisionAddOn-based ModelKit instances.

    Full hit:    skips the vision tower entirely, restores the KV snapshot,
                 prefills only the text suffix after the last image block.
    Partial hit: restores the deepest checkpoint, runs the ViT only on new
                 images, prefills the suffix from that point, saves new snapshots.
    Miss:        runs the full vision tower with optional pre-image KV reuse,
                 prefills with per-image checkpoints, persists them.

    Sets generate_args["prompt_cache"] in all paths.
    Returns (input_tokens, input_embeddings) for stream_generate.
    """
    vision_add_on = model_kit.vision_add_on
    vision_add_on.clear_prediction_state(model_kit.model)
    model_kit._cross_prompt_cache_active = False

    cw = getattr(model_kit, "cache_wrapper", None)

    def _activate_lru(cache, token_list, start_idx, input_ids) -> None:
        """Register the post-prefill cache with CacheWrapper and enable LRU recording."""
        if cw is None:
            return
        cw.set_image_turn_checkpoint(cache, token_list, start_idx)
        cw.tokens = input_ids
        cw.cache = cache
        model_kit._cross_prompt_cache_active = True

    # Reorder images to chronological conversation order.
    image_hashes = list(_hash_images(images_b64))
    images_b64, image_hashes = image_store.reorder_images_chronologically(
        images_b64, image_hashes
    )
    hash_chain = tuple(image_hashes)

    # Cheap CPU step: tokenize + resize, no ViT. Sets rope_deltas on text_model.
    # Keep grid_thw and pixel_values for partial hit path.
    input_ids, grid_thw, pixel_values, position_ids = vision_add_on._prepare_inputs(
        model_kit.model, prompt_tokens, images_b64, max_image_size
    )
    input_ids_list = input_ids.tolist()
    img_tok = vision_add_on.config.image_token_id
    vid_tok = getattr(vision_add_on.config, "video_token_id", None)
    block_lengths = image_block_lengths(input_ids_list, img_tok, vid_tok)

    # Invalidate stale checkpoints (prefix hash mismatch or block length change).
    for depth in range(len(hash_chain), 0, -1):
        key = hash_chain[:depth]
        if image_store.validate_image_checkpoint(
            key, input_ids_list, img_tok, vid_tok, block_lengths
        ):
            image_store.invalidate_image_checkpoint(key)

    hit = image_store.find_deepest_image_checkpoint(hash_chain)
    is_full_hit = hit is not None and hit[0] == hash_chain

    # --- Full hit path ---
    if is_full_hit:
        _, image_end_index = hit

        # Try LRU/prev-checkpoint first: if it covers at least image_end_index
        # tokens we can start prefill from there without any deepcopy.
        lru_cache, lru_start = (
            cw._find_starting_cache(input_ids_list, prefer_prev_checkpoint=True)
            if cw
            else (None, 0)
        )
        if lru_cache is not None and lru_start >= image_end_index:
            cache = lru_cache
            start_idx = lru_start
        else:
            # Impossible in practice: prev-checkpoint is always available and
            # covers image_end_index after any completed image turn.
            logger.warning(
                "[kv-image] full hit: no LRU available, rebuilding from scratch"
            )
            cache = make_prompt_cache(model_kit.model)
            start_idx = 0

        # Restore full MRoPE state so suffix tokens use the original stored positions.
        # rope_deltas was already set by _prepare_inputs; position_ids completes it.
        if position_ids is not None:
            model_kit.model.language_model.model.position_ids = position_ids

        cache, _ = _prefill_modelkit_with_image_checkpoints(
            model_kit,
            input_ids[start_idx:],
            None,
            [],
            reporter,
            initial_cache=cache,
            cached_tokens=start_idx,
        )

        _activate_lru(cache, input_ids_list, start_idx, input_ids)
        generate_args["prompt_cache"] = cache
        logger.info(
            f"[kv-image] cache hit depth={len(hash_chain)}"
            f" cached={start_idx}/{len(input_ids_list)} tokens"
        )
        return input_ids[-1:], None

    # --- Partial hit path ---
    is_partial_hit = hit is not None
    if is_partial_hit:
        hit_key, hit_image_end_index = hit
        partial_depth = len(hit_key)

        # Compute pixel patch boundaries per image to slice pixel_values.
        grid_list = grid_thw.tolist()
        if len(grid_list) == 3 and isinstance(grid_list[0], int):
            grid_list = [grid_list]  # single image: normalise to [[T, H, W]]
        patches_per_image = [t * h * w for t, h, w in grid_list]
        pixel_start = sum(patches_per_image[:partial_depth])
        new_pixel_values = pixel_values[pixel_start:]
        new_grid_thw = grid_thw[partial_depth:]

        # Embed text tokens of the suffix (image token positions will be replaced).
        suffix_input_ids = input_ids[hit_image_end_index:]
        suffix_embeds = model_kit.model.language_model.model.embed_tokens(
            suffix_input_ids[None]
        )
        if new_pixel_values.dtype != suffix_embeds.dtype:
            new_pixel_values = new_pixel_values.astype(suffix_embeds.dtype)

        # Run ViT only on the new images.
        hidden_states, _ = vision_add_on.vision_tower(
            new_pixel_values, new_grid_thw, output_hidden_states=False
        )

        # Merge new image features into the suffix embeddings.
        final_embeds, _ = vision_add_on.model_cls.merge_input_ids_with_image_features(
            hidden_states,
            suffix_embeds,
            suffix_input_ids[None],
            img_tok,
            vid_tok,
        )
        final_embeds = final_embeds.squeeze(0)

        # Restore MRoPE state.
        if position_ids is not None:
            model_kit.model.language_model.model.position_ids = position_ids

        # Try LRU/prev-checkpoint: if it covers past hit_image_end_index we can
        # skip any deepcopy and shorten the prefill.
        lru_cache, lru_start = (
            cw._find_starting_cache(input_ids_list, prefer_prev_checkpoint=True)
            if cw
            else (None, 0)
        )
        if lru_cache is not None and lru_start >= hit_image_end_index:
            cache = lru_cache
            actual_start = lru_start
        else:
            # Impossible in practice: prev-checkpoint is always available and
            # covers hit_image_end_index after any completed image turn.
            # Rebuild from scratch and skip the partial hit optimisation entirely.
            logger.warning(
                "[kv-image] partial hit: no LRU available, rebuilding from scratch"
            )
            generate_args["prompt_cache"] = make_prompt_cache(model_kit.model)
            return

        # Boundaries relative to suffix_input_ids, then adjusted for actual_start.
        all_boundaries = image_block_boundaries(input_ids_list, img_tok, vid_tok)
        suffix_boundaries = [
            (s - hit_image_end_index, e - hit_image_end_index)
            for s, e in all_boundaries
            if e > hit_image_end_index
        ]
        embed_slice = actual_start - hit_image_end_index
        adjusted_boundaries = [
            (s - embed_slice, e - embed_slice)
            for s, e in suffix_boundaries
            if e > embed_slice
        ]

        cache, new_checkpoints = _prefill_modelkit_with_image_checkpoints(
            model_kit,
            suffix_input_ids[embed_slice:],
            final_embeds[embed_slice:],
            adjusted_boundaries,
            reporter,
            initial_cache=cache,
            cached_tokens=actual_start,
            base_offset=actual_start,
        )
        image_store.save_block_checkpoints(
            hash_chain, partial_depth, new_checkpoints, input_ids_list, block_lengths
        )

        _activate_lru(cache, input_ids_list, actual_start, input_ids)
        generate_args["prompt_cache"] = cache
        logger.info(
            f"[kv-image] partial hit depth={partial_depth}/{len(hash_chain)}"
            f" cached={actual_start}/{len(input_ids_list)} tokens"
        )
        return input_ids[-1:], final_embeds[-1:]

    # --- Miss path ---
    # compute_embeddings calls _prepare_inputs again (cheap) and sets position_ids.
    input_ids, embeddings = vision_add_on.compute_embeddings(
        model_kit.model, prompt_tokens, images_b64, max_image_size
    )
    ids_list = input_ids.tolist()
    image_boundaries = image_block_boundaries(ids_list, img_tok, vid_tok)

    # Try to reuse pre-image KV state from CacheWrapper (e.g. system-prompt tokens).
    pre_cache, pre_cached = _try_inject_pre_image_cache(
        model_kit, ids_list, img_tok, vid_tok
    )

    if pre_cache is not None:
        suffix_ids = input_ids[pre_cached:]
        suffix_embs = embeddings[pre_cached:]
        suffix_bounds = [
            (s - pre_cached, e - pre_cached)
            for s, e in image_boundaries
            if e > pre_cached
        ]
        cache, block_checkpoints = _prefill_modelkit_with_image_checkpoints(
            model_kit,
            suffix_ids,
            suffix_embs,
            suffix_bounds,
            reporter,
            initial_cache=pre_cache,
            cached_tokens=pre_cached,
            base_offset=pre_cached,
        )
    else:
        cache, block_checkpoints = _prefill_modelkit_with_image_checkpoints(
            model_kit, input_ids, embeddings, image_boundaries, reporter
        )

    image_store.save_block_checkpoints(
        hash_chain, 0, block_checkpoints, ids_list, block_lengths
    )

    start_idx = pre_cached if pre_cache is not None else 0
    _activate_lru(cache, ids_list, start_idx, input_ids)
    generate_args["prompt_cache"] = cache
    return input_ids[-1:], embeddings[-1:]


def _sequential_generation(
    model_kit: ModelKit | VisionModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_reporter: Optional[PromptProgressReporter] = None,
    images_b64: Optional[List[str]] = None,
    max_image_size: Optional[tuple[int, int]] = None,
    stop_strings: Optional[List[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
    speculative_decoding_toggle: Optional[bool] = None,
    num_draft_tokens: Optional[int] = None,
    request_id: Optional[str] = None,
) -> Iterator[GenerationResult]:
    with _sequential_gen_abort_handler(model_kit, request_id) as cancel_event:
        if cancel_event.is_set() or model_kit.is_shutdown():
            yield construct_user_cancelled_result()
            return

        set_seed(seed)

        generate_args = {}
        if prompt_progress_reporter is None:
            prompt_progress_reporter = LoggerReporter()

        # Set up kv cache
        if type(model_kit) is not VisionModelKit:
            for attr in [
                "max_kv_size",
                "kv_bits",
                "kv_group_size",
                "quantized_kv_start",
            ]:
                value = getattr(model_kit, attr, None)
                if value is not None:
                    generate_args[attr] = value

        # Set up repetition penalty
        repetition_penalty_kwargs = setup_repetition_penalty(
            repetition_penalty, repetition_context_size
        )

        # Set up speculative decoding
        draft_model = determine_draft_model_for_generation(
            model_kit, speculative_decoding_toggle
        )
        configure_num_draft_tokens_in_generate_args(
            model_kit, draft_model, num_draft_tokens, generate_args
        )

        # Process prompt
        image_store = _get_image_store(model_kit)
        image_cache_prefill_done = False
        try:
            if images_b64 and image_store is not None:
                input_tokens, input_embeddings = _process_modelkit_image_cache(
                    model_kit,
                    image_store,
                    prompt_tokens,
                    images_b64,
                    max_image_size,
                    generate_args,
                    prompt_progress_reporter,
                )
                image_cache_prefill_done = True
            else:
                input_tokens, input_embeddings = model_kit.process_prompt(
                    prompt_tokens,
                    images_b64,
                    prompt_progress_reporter,
                    generate_args,
                    max_image_size,
                    speculative_decoding_toggle,
                )
        except StopPromptProcessing:
            yield construct_user_cancelled_result()
            return
        if draft_model is None:
            # input embeddings not yet supported for speculative decoding in mlx-lm
            generate_args["input_embeddings"] = input_embeddings

        # Setup logits processors
        logits_processors = setup_logits_processors(
            repetition_penalty,
            repetition_penalty_kwargs,
            prompt_tokens,
            input_tokens,
            None,
            model_kit.tokenizer,
        )

        # Set up sampler
        generate_args["sampler"] = create_sampler(
            temp, top_p, min_p, min_tokens_to_keep, top_k
        )

        # If using VisionModelKit, immediately record the token once it's sampled
        if type(model_kit) is VisionModelKit:
            sampler_func = generate_args["sampler"]

            def sampler_func_wrapper(*args, **kwargs):
                token = sampler_func(*args, **kwargs)
                model_kit.record_sampled_token(token)
                return token

            generate_args["sampler"] = sampler_func_wrapper

        # Validate top_logprobs
        top_logprobs = validate_top_logprobs(top_logprobs)

        # Keep track of tokens buffered by detokenizer to yield accurate generation results
        token_buffer: List[Token] = []
        top_logprobs_buffer: List[List[Token]] = []

        tokenizer = model_kit.tokenizer

        # Add outlines logits processor if json_schema is provided
        if json_schema is not None:
            logits_processors.append(
                JSONLogitsProcessor(
                    json_schema,
                    OutlinesTransformerTokenizer(model_kit.tokenizer._tokenizer),
                    tensor_library_name="mlx",
                )
            )

        # Set up stop string processor if non-empty stop_strings are provided
        stop_string_processor = create_stop_string_processor(stop_strings, tokenizer)
        text = ""

        # Determine callback for mlx-lm based on processing mode.
        # image_cache_prefill_done: prefill already handled + reporter.finish called.
        # is_cross_prompt_cache_active: CacheWrapper LRU handled reporting.
        if image_cache_prefill_done or model_kit.is_cross_prompt_cache_active():
            mlx_lm_callback = None
        else:
            mlx_lm_callback = MlxLmReporterAdapter(
                prompt_progress_reporter, emit_begin=True
            )

        stream = stream_generate(
            model=model_kit.model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt=input_tokens,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            prompt_progress_callback=mlx_lm_callback,
            prefill_step_size=model_kit.prefill_step_size,
            **generate_args,
        )

        try:
            while not model_kit.is_shutdown() and not cancel_event.is_set():
                try:
                    generation_result = next(stream)
                except StopIteration:
                    break
                except StopPromptProcessing:
                    yield construct_user_cancelled_result()
                    return

                # Token processor
                token = generation_result.token
                text += generation_result.text

                # record generated token to cache, if cache is active
                if model_kit.is_cross_prompt_cache_active():
                    model_kit.record_token_to_cache(token)

                logprobs = generation_result.logprobs
                token_buffer.append(
                    Token(
                        token,
                        tokenizer.decode(token),
                        float(logprobs[token]),
                        from_draft=generation_result.from_draft,
                    )
                )
                if top_logprobs:
                    top_logprobs_buffer.append(
                        summarize_top_logprobs(tokenizer, logprobs, top_logprobs)
                    )

                # Stop processor
                should_stop, should_buffer, stop_result = process_stop_string_check(
                    stop_string_processor, token
                )
                if should_stop:
                    yield _handle_stop_string_detected(
                        tokenizer,
                        stop_result,
                        text,
                        token_buffer,
                        top_logprobs_buffer,
                    )
                    break  # stop generation

                # If we currently have generated a partial match with a stop sequence, or detected an
                # in-progress multi-byte string, generate new tokens until we know if the stop sequence
                # is hit or not (i.e., make sure not to yield yet)
                if should_buffer:
                    continue

                # Standard yield - yield when a non-empty text segment is available or eos token is hit
                should_yield, stop_condition = should_yield_token(
                    text, token, tokenizer
                )
                if should_yield:
                    yield GenerationResult(
                        text=text,
                        tokens=token_buffer,
                        stop_condition=stop_condition,
                        top_logprobs=top_logprobs_buffer,
                    )
                    token_buffer = []
                    top_logprobs_buffer = []
                    text = ""
            if cancel_event.is_set() or model_kit.is_shutdown():
                yield construct_user_cancelled_result()
                return
        finally:
            # Callers that stop iterating before StopIteration would otherwise
            # skip finalize_generation. It is a no-op if _prefill_checkpoint is None.
            cw = getattr(model_kit, "cache_wrapper", None)
            if cw is not None:
                cw.finalize_generation()


def _batched_generation(
    model_kit: BatchedModelKit,
    prompt_tokens: List[int],
    *,
    prompt_progress_reporter: Optional[PromptProgressReporter] = None,
    images_b64: Optional[List[str]] = None,
    max_image_size: Optional[tuple[int, int]] = None,
    stop_strings: Optional[List[str]] = None,
    top_logprobs: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    min_tokens_to_keep: Optional[int] = None,
    seed: Optional[int] = None,  # Seed arg is ignored for batched gen
    json_schema: Optional[str] = None,
    max_tokens: Optional[int] = 10000000,
    speculative_decoding_toggle: Optional[bool] = None,
    num_draft_tokens: Optional[int] = None,
    request_id: str | None = None,
) -> Iterator[GenerationResult]:
    # We need a request_id so that we can communicate with the batched backend
    if request_id is None or request_id == "":
        logger.warning(
            "Received a generation request without a request_id! Please send a request_id"
        )
        request_id = uuid.uuid4()

    input_tokens = prompt_tokens
    if prompt_progress_reporter is None:
        prompt_progress_reporter = DefaultPromptProgressReporter()

    # Set up repetition penalty
    repetition_penalty_kwargs = setup_repetition_penalty(
        repetition_penalty, repetition_context_size
    )

    # Setup logits processors
    tokenizer = model_kit.tokenizer
    logits_processors = setup_logits_processors(
        repetition_penalty,
        repetition_penalty_kwargs,
        prompt_tokens,
        input_tokens,
        None,
        tokenizer,
    )

    # Set up sampler
    sampler = create_sampler(temp, top_p, min_p, min_tokens_to_keep, top_k)

    # Validate top_logprobs
    top_logprobs = validate_top_logprobs(top_logprobs)

    # Keep track of tokens buffered by detokenizer to yield accurate generation results
    token_buffer: List[Token] = []
    top_logprobs_buffer: List[List[Token]] = []

    # Add outlines logits processor if json_schema is provided
    if json_schema is not None:
        logits_processors.append(
            JSONLogitsProcessor(
                json_schema,
                OutlinesTransformerTokenizer(model_kit.tokenizer._tokenizer),
                tensor_library_name="mlx",
            )
        )

    # Set up stop string processor if non-empty stop_strings are provided
    stop_string_processor = create_stop_string_processor(stop_strings, tokenizer)
    text = ""

    mlx_lm_callback = BatchedMlxLmReporterAdapter(
        prompt_progress_reporter, emit_begin=True
    )

    stream = model_kit.generate(
        prompt_tokens=input_tokens,
        request_id=request_id,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_progress_callback=mlx_lm_callback,
        top_logprobs=top_logprobs,
    )

    while True:
        try:
            generation_result: BatchedGenerationResponse = next(stream)
        except StopIteration:
            break
        except RequestCancelled:
            yield construct_user_cancelled_result()
            return
        # TODO: implement this - MLX doesn't yet support cancelling during prompt processing
        # for batched generation
        # except StopPromptProcessing:
        #     yield construct_user_cancelled_result()
        #     return

        # Token processor
        token = generation_result.token
        text += generation_result.text

        token_buffer.append(
            Token(
                token,
                tokenizer.decode(token),
                generation_result.token_logprob,
                from_draft=generation_result.from_draft,
            )
        )
        if top_logprobs and generation_result.top_logprobs is not None:
            top_logprobs_buffer.append(generation_result.top_logprobs)

        # Stop processor
        should_stop, should_buffer, stop_result = process_stop_string_check(
            stop_string_processor, token
        )
        if should_stop:
            yield _handle_stop_string_detected(
                tokenizer,
                stop_result,
                text,
                token_buffer,
                top_logprobs_buffer,
            )
            model_kit.remove(request_id)
            break  # stop generation

        # If we currently have generated a partial match with a stop sequence, or detected an
        # in-progress multi-byte string, generate new tokens until we know if the stop sequence
        # is hit or not (i.e., make sure not to yield yet)
        if should_buffer:
            continue

        # Standard yield - yield when a non-empty text segment is available or eos token is hit
        should_yield, stop_condition = should_yield_token(text, token, tokenizer)
        if should_yield:
            yield GenerationResult(
                text=text,
                tokens=token_buffer,
                stop_condition=stop_condition,
                top_logprobs=top_logprobs_buffer,
            )
            token_buffer = []
            top_logprobs_buffer = []
            text = ""

        # The batched generator has hit max_tokens, so we can't iterate further
        if generation_result.finish_reason == "length":
            yield GenerationResult(
                text="",
                tokens=[],
                stop_condition=GenerationStopCondition(
                    stop_reason="token_limit",
                    stop_string="",
                    stop_tokens=[],
                ),
                top_logprobs=[],
            )
            return


def stop_generation(
    model_kit: ModelKit | VisionModelKit | BatchedModelKit, request_id: str
):
    """
    Register stop request based off of request_id
    """
    if request_id is None or request_id == "":
        logger.error("request_id cannot be empty in stop request")
        return

    if isinstance(model_kit, BatchedModelKit):
        model_kit.remove(request_id)
        return

    if not model_kit.cancel_request(request_id):
        logger.warning(f"Could not cancel {request_id=} (request not found)")


def unload(model_kit: ModelKit | VisionModelKit | BatchedModelKit):
    model_kit.shutdown()


def tokenize(model_kit: ModelKit | VisionModelKit, prompt: str) -> List[int]:
    """
    Convert a text prompt into a list of token IDs using the model's tokenizer.

    Args:
        model_kit (ModelKit | VisionModelKit): The model kit instance containing the tokenizer
            to use for tokenization
        prompt (str): The raw text prompt to be tokenized

    Returns:
        List[int]: A list of integer token IDs representing the tokenized prompt,
            ready for model input
    """
    return model_kit.tokenize(prompt)
