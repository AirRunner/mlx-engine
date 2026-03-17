import copy
import logging
from typing import List, Optional

import mlx.core as mx
from mlx_vlm.models.base import InputEmbeddingsFeatures
from mlx_vlm.models.cache import make_prompt_cache

from mlx_engine.cache_wrapper import PROMPT_PROCESSING_CHUNK_SIZE
from mlx_engine.model_kit.vision_add_ons.process_prompt_with_images import (
    common_process_prompt_with_images,
)
from mlx_engine.utils.prompt_progress_reporter import (
    PromptProgressReporter,
    StopPromptProcessing,
)

logger = logging.getLogger(__name__)


def _find_image_blocks(flat_ids, img_tok, vid_tok):
    """Return list of (start, end) for each contiguous image-token block."""
    blocks = []
    in_block = False
    start = 0
    for i, tok in enumerate(flat_ids):
        is_vis = tok == img_tok or tok == vid_tok
        if is_vis and not in_block:
            in_block = True
            start = i
        elif not is_vis and in_block:
            blocks.append((start, i))
            in_block = False
    if in_block:
        blocks.append((start, len(flat_ids)))
    return blocks


def _find_image_block_ends(flat_ids, img_tok, vid_tok):
    """Return the end position (exclusive) of each contiguous image-token block."""
    return [end for _, end in _find_image_blocks(flat_ids, img_tok, vid_tok)]


class VisionModelWrapper:
    """
    Wrapper class for Vision Models support
    This wrapper class adapts mlx-vlm models so that they can be slotted into the mlx_lm generation engine
    This wrapper defines `__getattr__` and `__setattr__` to allow the model properties to be set/get as if it were a text model

    Models are evaluated in `mlx_lm` with the `__call__` method, so define a custom `__call__` method to forward calls to the vision model
    """

    def __init__(self, model):
        """
        Set the class members in this unusual way, so that we can define `__getattr__` and `__setattr__`
        """
        self.__dict__["_model_attrs"] = {
            "vision_model": model,
            "input_ids": None,
            "pixel_values": None,
            "mask": None,
            "first_call": False,
            "decoder_input_ids": None,
            "language_model_kwargs": {},
            # vision model kwargs
            "model_inputs": {},
            # set during image prefill; read by VisionCacheWrapper for cross-turn KV reuse
            "image_end_index": None,
            "image_kv_checkpoint": None,
            # one (end_idx, cache_snapshot) entry per image block, in order
            "image_block_checkpoints": [],
        }

    def __getattr__(self, name):
        """
        First, check if the name is a member of this class
        Then, check if the name is a member of the language model
        Finally, check if the name is a member of the vision model
        """
        if name in self._model_attrs:
            return self._model_attrs[name]
        try:
            return getattr(self.vision_model.language_model, name)
        except AttributeError:
            pass
        return getattr(self.vision_model, name)

    def __setattr__(self, name, value):
        """
        Set attribute of this class if it's not a member of the vision model
        """
        if name in self._model_attrs or not hasattr(self.vision_model, name):
            self._model_attrs[name] = value
        else:
            setattr(self.vision_model, name, value)

    def __call__(self, *args, input_embeddings=None, **kwargs):
        """
        This mirrors mlx-vlm's native generation loop (`mlx_vlm.generate.generate_step`):
        do one multimodal prompt/prefill step (via `get_input_embeddings`) and then
        forward all subsequent single-token decoding calls directly to the language model.
        ref: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L229
        """
        if self.pixel_values is not None and not self.first_call:
            self.first_call = True

            # Replace the mlx-lm specific prompt cache with the mlx-vlm prompt cache
            cache = make_prompt_cache(self.language_model)
            kwargs["cache"] = cache

            embedding_output = self.vision_model.get_input_embeddings(
                input_ids=self.input_ids,
                pixel_values=self.pixel_values,
                mask=self.mask,
                **self.model_inputs,
            )

            # Expect InputEmbeddingsFeatures here to match mlx-vlm's `generate_step` flow
            # https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L383-L396
            if not isinstance(embedding_output, InputEmbeddingsFeatures):
                raise TypeError(
                    "vision_model.get_input_embeddings(...) must return InputEmbeddingsFeatures. "
                    f"Got {type(embedding_output)}."
                )

            inputs_embeds = embedding_output.inputs_embeds
            if inputs_embeds is None:
                raise ValueError(
                    "vision_model.get_input_embeddings(...) returned InputEmbeddingsFeatures "
                    "without `inputs_embeds`."
                )

            lm_call_kwargs = {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }

            # Mirror model.__call__ behavior for models that produce a 4D attention mask.
            # ref: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/models/gemma3/gemma3.py#L186-L192
            attention_mask_4d = lm_call_kwargs.pop("attention_mask_4d", None)
            if attention_mask_4d is not None:
                lm_call_kwargs["mask"] = attention_mask_4d

            image_end_index = embedding_output.image_end_index
            total = inputs_embeds.shape[1]
            input_ids = self.input_ids

            if len(lm_call_kwargs) == 0 and image_end_index is not None:
                # Early-fusion model with known image boundary.
                # Split prefill at image_end_index so we can checkpoint the KV cache
                # right after the image tokens — enabling cross-turn image KV reuse.

                # Find per-image block boundaries for incremental multi-image caching.
                try:
                    img_tok = self.vision_model.config.image_token_index
                    vid_tok = getattr(
                        self.vision_model.config, "video_token_index", img_tok
                    )
                    block_ends = _find_image_block_ends(
                        input_ids[0].tolist(), img_tok, vid_tok
                    )
                except AttributeError:
                    block_ends = []
                if not block_ends or block_ends[-1] != image_end_index:
                    block_ends = [image_end_index]

                # Phase 1: process each image block and checkpoint after each one.
                # This enables VisionCacheWrapper to restore KV state at any image
                # boundary, so only new images need re-processing on the next turn.
                block_checkpoints = []
                prev = 0
                for block_end in block_ends:
                    for start in range(prev, block_end, PROMPT_PROCESSING_CHUNK_SIZE):
                        end = min(start + PROMPT_PROCESSING_CHUNK_SIZE, block_end)
                        self.language_model(
                            input_ids[:, start:end],
                            inputs_embeds=inputs_embeds[:, start:end],
                            cache=cache,
                        )
                        mx.eval([c.state for c in cache])
                        mx.clear_cache()
                    block_checkpoints.append((block_end, copy.deepcopy(cache)))
                    prev = block_end

                self.image_block_checkpoints = block_checkpoints
                self.image_end_index = image_end_index
                self.image_kv_checkpoint = block_checkpoints[-1][1]

                # Phase 2: text tokens [image_end_index, total-1) in chunks
                for start in range(
                    image_end_index, total - 1, PROMPT_PROCESSING_CHUNK_SIZE
                ):
                    end = min(start + PROMPT_PROCESSING_CHUNK_SIZE, total - 1)
                    self.language_model(
                        input_ids[:, start:end],
                        inputs_embeds=inputs_embeds[:, start:end],
                        cache=cache,
                    )
                    mx.eval([c.state for c in cache])
                    mx.clear_cache()

                # Last token is returned as the first sampled output token
                outputs = self.language_model(
                    input_ids[:, -1:],
                    inputs_embeds=inputs_embeds[:, -1:],
                    cache=cache,
                )
            elif len(lm_call_kwargs) == 0 and total > PROMPT_PROCESSING_CHUNK_SIZE + 1:
                # Early-fusion model without a known image boundary
                # (mlx-vlm model does not expose image_end_index yet).
                # Chunk the prefill to bound peak VRAM.
                self.image_end_index = None
                self.image_kv_checkpoint = None
                self.image_block_checkpoints = []
                for start in range(0, total - 1, PROMPT_PROCESSING_CHUNK_SIZE):
                    end = min(start + PROMPT_PROCESSING_CHUNK_SIZE, total - 1)
                    self.language_model(
                        input_ids[:, start:end],
                        inputs_embeds=inputs_embeds[:, start:end],
                        cache=cache,
                    )
                    mx.eval([c.state for c in cache])
                    mx.clear_cache()
                outputs = self.language_model(
                    input_ids[:, -1:],
                    inputs_embeds=inputs_embeds[:, -1:],
                    cache=cache,
                )
            else:
                # Models with per-chunk kwargs use a single pass; chunking is not applicable.
                self.image_end_index = None
                self.image_kv_checkpoint = None
                self.image_block_checkpoints = []
                outputs = self.language_model(
                    self.input_ids,
                    inputs_embeds=inputs_embeds,
                    cache=cache,
                    **lm_call_kwargs,
                )

            # Persist only decode type kwargs + cache to mirror mlx-vlm's native generation loop
            # ref: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L369-L377
            persisted_kwargs = {"cache": cache}
            if outputs.cross_attention_states is not None:
                persisted_kwargs["cross_attention_states"] = (
                    outputs.cross_attention_states
                )
            elif outputs.encoder_outputs is not None:
                # `decoder_input_ids` is updated each step via `record_sampled_token`.
                self.decoder_input_ids = self.input_ids
                persisted_kwargs["decoder_input_ids"] = self.decoder_input_ids
                persisted_kwargs["encoder_outputs"] = outputs.encoder_outputs

            self.language_model_kwargs = persisted_kwargs
        else:
            try:
                # This is only missing if self.pixel_values is None
                if "cache" in self.language_model_kwargs:
                    # Use the cache from self.language_model_kwargs
                    kwargs.pop("cache", None)

                lm_call_kwargs = dict(self.language_model_kwargs)

                # Mirrors mlx-vlm's `generate_step` continuation path for encoder-decoder models:
                # https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L332-L336
                if "decoder_input_ids" in lm_call_kwargs:
                    # Avoid passing decoder_inputs_embeds alongside decoder_input_ids.
                    lm_call_kwargs.pop("decoder_inputs_embeds", None)
                    lm_call_kwargs["decoder_input_ids"] = self.decoder_input_ids
                    outputs = self.language_model(
                        **kwargs,
                        **lm_call_kwargs,
                    )
                else:
                    outputs = self.language_model(
                        *args,
                        **kwargs,
                        **lm_call_kwargs,
                    )

            except ValueError as e:
                # Create a friendly error message if a user tries to use mllama without images
                if "Cross attention states must be provided for layer" in str(e):
                    raise ValueError(
                        "Using this model without any images attached is not supported yet."
                    )
                raise e

        return outputs.logits

    def record_sampled_token(self, token: mx.array) -> None:
        """
        Record the most recently sampled token for the next decode step.

        In mlx-lm, samplers return an `mx.array` (typically shape (1,) for batch=1).
        For encoder-decoder models, the decoder expects `decoder_input_ids` shaped
        as (batch, seq), so normalize to mirror mlx-vlm's `y[None]` flow.

        ref: https://github.com/Blaizzy/mlx-vlm/blob/1028599/mlx_vlm/generate.py#L372-L375
        """
        # defensive safety checks
        if not isinstance(token, mx.array):
            raise TypeError(f"Expected token to be an mx.array, got {type(token)}.")
        if token.shape != (1,):
            raise ValueError(
                f"Expected a single sampled token, got shape {token.shape}."
            )
        self.decoder_input_ids = token[None]

    def process_prompt_with_images(
        self,
        images_b64: Optional[List[str]],
        prompt_tokens: mx.array,
        processor,
        detokenizer,
        max_image_size: tuple[int, int] | None,
    ):
        """
        This method generates the input_ids, pixel_values, and mask for the vision model
        Call this before starting evaluation
        """
        if images_b64 is None:
            images_b64 = []

        # Handle the case with no images
        if len(images_b64) == 0:
            detokenizer.reset()
            [detokenizer.add_token(token) for token in prompt_tokens]
            detokenizer.finalize()
            prompt = detokenizer.text

            logger.debug(f"Prompt dump: {prompt}\n")

            try:
                if hasattr(processor, "process"):
                    # Needed for Molmo
                    self.input_ids = mx.array(
                        processor.process(text=prompt)["input_ids"]
                    )
                else:
                    self.input_ids = mx.array(processor(text=prompt).input_ids)
            except ValueError as e:
                if "`images` are expected as arguments" in str(e):
                    raise ValueError(
                        "Using this model without any images attached is not supported yet."
                    )
                raise e
        else:
            # Use the common function for image processing
            processed = common_process_prompt_with_images(
                prompt_tokens=prompt_tokens,
                images_b64=images_b64,
                processor=processor,
                config=self.vision_model.config,
                max_size=max_image_size,
            )

            # Set class attributes from the processed result
            self.input_ids = processed.input_ids
            self.pixel_values = processed.pixel_values
            self.mask = processed.attention_mask
            self.model_inputs = processed.other_inputs

    def prefill_with_partial_cache(
        self,
        partial_depth: int,
        base_cache,
        reporter: PromptProgressReporter,
    ) -> tuple:
        """
        Run the vision tower only for images[partial_depth:] and prefill the suffix.

        Args:
            partial_depth: Number of images whose KV is already in base_cache.
            base_cache:    Deep-copied KV snapshot at partial_depth (will be mutated).
            reporter:      Progress reporter for prefill UI feedback.

        Returns:
            (base_cache, last_token, new_block_checkpoints) where base_cache has been
            filled with all tokens after the partial checkpoint, last_token (shape (1,))
            is the final input token for stream_generate, and new_block_checkpoints is
            a list of (end_idx, cache_snapshot) for each newly processed image block.

        Raises:
            ValueError: if prerequisites are not available.
            StopPromptProcessing: if the user cancels during prefill.
        """
        if not hasattr(self.vision_model, "get_partial_input_embeddings"):
            raise NotImplementedError(
                f"{type(self.vision_model).__name__} does not support partial image KV "
                "caching. Implement get_partial_input_embeddings() to enable this feature."
            )

        input_ids = self.input_ids  # (1, T)
        total = input_ids.shape[1]
        img_tok = self.vision_model.config.image_token_index
        vid_tok = getattr(self.vision_model.config, "video_token_index", img_tok)

        blocks = _find_image_blocks(input_ids[0].tolist(), img_tok, vid_tok)
        if not blocks or partial_depth >= len(blocks):
            raise ValueError(
                f"partial_depth {partial_depth} is out of range for {len(blocks)} image blocks"
            )

        inputs_embeds = self.vision_model.get_partial_input_embeddings(
            input_ids=input_ids,
            pixel_values=self.pixel_values,
            mask=self.mask,
            model_inputs=self.model_inputs,
            partial_depth=partial_depth,
        )
        mx.eval(inputs_embeds)
        mx.clear_cache()

        # Prefill suffix: from end of last cached image block to end of sequence
        start_pos = blocks[partial_depth - 1][1]
        image_end = blocks[-1][1]
        processed = 0

        reporter.begin(
            is_draft=False,
            cached_tokens=start_pos,
            total_prompt_tokens=total,
            prefill_tokens_processed=0,
        )

        new_block_checkpoints = []
        prev = start_pos

        # Process each new image block (and preceding text) with per-block checkpoints
        for block_start, block_end in blocks[partial_depth:]:
            # Text between prev and this image block
            for s in range(prev, block_start, PROMPT_PROCESSING_CHUNK_SIZE):
                e = min(s + PROMPT_PROCESSING_CHUNK_SIZE, block_start)
                self.language_model(
                    input_ids[:, s:e],
                    inputs_embeds=inputs_embeds[:, s:e],
                    cache=base_cache,
                )
                mx.eval([c.state for c in base_cache])
                mx.clear_cache()
                processed += e - s
                if not reporter.update(
                    is_draft=False, prefill_tokens_processed=processed
                ):
                    raise StopPromptProcessing
            # Image block
            for s in range(block_start, block_end, PROMPT_PROCESSING_CHUNK_SIZE):
                e = min(s + PROMPT_PROCESSING_CHUNK_SIZE, block_end)
                self.language_model(
                    input_ids[:, s:e],
                    inputs_embeds=inputs_embeds[:, s:e],
                    cache=base_cache,
                )
                mx.eval([c.state for c in base_cache])
                mx.clear_cache()
                processed += e - s
                if not reporter.update(
                    is_draft=False, prefill_tokens_processed=processed
                ):
                    raise StopPromptProcessing
            new_block_checkpoints.append((block_end, copy.deepcopy(base_cache)))
            prev = block_end

        # Text after all images, excluding the last token
        for s in range(image_end, total - 1, PROMPT_PROCESSING_CHUNK_SIZE):
            e = min(s + PROMPT_PROCESSING_CHUNK_SIZE, total - 1)
            self.language_model(
                input_ids[:, s:e],
                inputs_embeds=inputs_embeds[:, s:e],
                cache=base_cache,
            )
            mx.eval([c.state for c in base_cache])
            mx.clear_cache()
            processed += e - s
            if not reporter.update(is_draft=False, prefill_tokens_processed=processed):
                raise StopPromptProcessing

        reporter.finish(is_draft=False, prefill_tokens_processed=processed)

        last_token = input_ids[0, -1:]  # shape (1,)
        return base_cache, last_token, new_block_checkpoints

    def reset(self) -> None:
        """
        Reset per-request generation state without reloading model weights.

        Zeroes all mutable state fields while preserving the vision_model
        reference.  Use this instead of recreating the VisionModelWrapper to
        avoid the 2-3 s overhead of mlx_vlm.utils.load() on every request.
        """
        self._model_attrs.update(
            {
                "input_ids": None,
                "pixel_values": None,
                "mask": None,
                "first_call": False,
                "decoder_input_ids": None,
                "language_model_kwargs": {},
                "model_inputs": {},
                "image_end_index": None,
                "image_kv_checkpoint": None,
                "image_block_checkpoints": [],
            }
        )

    @property
    def vision_model(self):
        return self._model_attrs["vision_model"]

    @property
    def language_model(self):
        return self.vision_model.language_model
