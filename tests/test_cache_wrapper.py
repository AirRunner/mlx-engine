import types
import unittest
import mlx.core as mx
from mlx_engine.cache_wrapper import (
    CacheWrapper,
    VisionCacheWrapper,
    StopPromptProcessing,
)
from tests.shared import model_getter, RecordingReporter, CancellingReporter
from mlx_engine.generate import load_model, tokenize


# ---------------------------------------------------------------------------
# Shared mock infrastructure for TestVisionCacheWrapper
# ---------------------------------------------------------------------------


class _MockCacheEntry:
    """Minimal KV cache entry compatible with LRUPromptCache (needs .nbytes)."""

    def __init__(self):
        self.nbytes = 256
        self._state = mx.zeros((1,))

    @property
    def state(self):
        return self._state

    def is_trimmable(self):
        return False


def _make_cache(n: int = 2):
    """Return a list of _MockCacheEntry objects that acts as a KV cache list."""
    return [_MockCacheEntry() for _ in range(n)]


class _MockLanguageModel:
    def __call__(self, tokens, cache=None, **kwargs):
        return types.SimpleNamespace(logits=mx.zeros((1, 1, 100)))


class _MockVisionModel:
    def __init__(self):
        self.language_model = _MockLanguageModel()
        self.config = types.SimpleNamespace(
            image_token_index=9999,
            video_token_index=9999,
        )


class _MockWrapper:
    """Minimal VisionModelWrapper-like object for VisionCacheWrapper tests."""

    def __init__(self):
        self.vision_model = _MockVisionModel()
        self.language_model = self.vision_model.language_model

    def __call__(self, tokens, cache=None, **kwargs):
        return self.language_model(tokens, cache=cache)


class _MockTokenizer:
    has_thinking = False


class _MockThinkingTokenizer:
    has_thinking = True
    think_start_id = 8888


class TestCacheWrapper(unittest.TestCase):
    def test_find_common_prefix_with_mismatch(self):
        """Test when there's a mismatch in the tokens"""
        # Create two arrays with a known common prefix [1, 2, 3]
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 6, 7])  # Mismatch at index 3
        num_tokens_to_exclude = 1

        print("\nTest with mismatch:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_common_prefix(
            current_tokens, prompt_tokens, num_tokens_to_exclude
        )
        self.assertEqual(result, 3)  # Should find 3 matching tokens

    def test_find_common_prefix_all_match(self):
        """Test when all tokens match"""
        # Create two identical arrays
        current_tokens = mx.array([1, 2, 3, 4, 5])
        prompt_tokens = mx.array([1, 2, 3, 4, 5])  # All tokens match
        num_tokens_to_exclude = 1

        print("\nTest with all matching:")
        print(f"current_tokens: {current_tokens}")
        print(f"prompt_tokens: {prompt_tokens}")

        result = CacheWrapper._find_common_prefix(
            current_tokens, prompt_tokens, num_tokens_to_exclude
        )
        self.assertEqual(
            result, 4
        )  # Should find 4 matching tokens (5-1 due to num_tokens_to_exclude)

    def test_prompt_processing_cancellation(self):
        """Test that progress is saved when processing is cancelled and cache is reused on retry"""

        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096)

        chunk_size = 20  # Small chunk size to ensure multiple progress callbacks
        num_tokens_to_exclude = 1
        model_kit.cache_wrapper = CacheWrapper(
            model_kit.model,
            max_kv_size=4096,
            chunk_size=chunk_size,
        )

        long_prompt = (
            "This is a test prompt that needs to be long enough to require multiple chunks for processing. "
            * 50
        )
        prompt_tokens = mx.array(tokenize(model_kit, long_prompt))
        tokens_to_process = len(prompt_tokens) - num_tokens_to_exclude
        # ceiling division +1 for finish
        expected_chunks = (tokens_to_process + chunk_size - 1) // chunk_size + 1

        # First attempt: Reporter that cancels after 3 events
        cancelling_reporter = CancellingReporter(cancel_after=3)

        with self.assertRaises(StopPromptProcessing):
            model_kit.cache_wrapper.update_cache(
                prompt_tokens=prompt_tokens,
                reporter=cancelling_reporter,
                num_tokens_to_exclude=1,
            )
        first_attempt_event_count = len(cancelling_reporter.events)

        # Second attempt: Reporter that doesn't cancel
        recording_reporter = RecordingReporter()

        result_tokens = model_kit.cache_wrapper.update_cache(
            prompt_tokens=prompt_tokens,
            reporter=recording_reporter,
            num_tokens_to_exclude=1,
        )
        second_attempt_event_count = len(recording_reporter.events)

        self.assertEqual(
            second_attempt_event_count,
            # +1 for finish, +1 for the begin event on retry
            expected_chunks - first_attempt_event_count + 2,
        )

        # Verify that the second attempt completed successfully
        self.assertIsNotNone(result_tokens)


class TestVisionCacheWrapper(unittest.TestCase):
    """Unit tests for VisionCacheWrapper's new unified cache methods."""

    def _make_wrapper(self, tokenizer=None):
        if tokenizer is None:
            tokenizer = _MockTokenizer()
        return VisionCacheWrapper(_MockWrapper(), tokenizer)

    # ------------------------------------------------------------------
    # save_post_prefill_snapshot / fetch_continuation_cache
    # ------------------------------------------------------------------

    def test_save_and_fetch_post_prefill_snapshot(self):
        """Snapshot saved after image-hit prefill is retrieved on the next turn."""
        wrapper = self._make_wrapper()
        tokens = [1, 2, 3, 4, 5]
        cache = _make_cache()
        wrapper.save_post_prefill_snapshot(tokens, cache)

        # Checkpoint key excludes the last token (offset=1 for non-thinking model)
        result, rest = wrapper.fetch_continuation_cache(tokens)
        self.assertIsNotNone(result)
        # rest should be at most the last token
        self.assertLessEqual(len(rest), 1)

    def test_fetch_continuation_cache_miss(self):
        """Empty LRU returns (None, full_tokens)."""
        wrapper = self._make_wrapper()
        tokens = [10, 20, 30]
        result, rest = wrapper.fetch_continuation_cache(tokens)
        self.assertIsNone(result)
        self.assertEqual(rest, tokens)

    def test_continuation_cache_hit_skips_reprefill(self):
        """LRU hit condition used in generate.py: len(rest) <= 1 is satisfied."""
        wrapper = self._make_wrapper()
        tokens = list(range(50))
        wrapper.save_post_prefill_snapshot(tokens, _make_cache())

        result, rest = wrapper.fetch_continuation_cache(tokens)
        self.assertIsNotNone(result)
        self.assertLessEqual(
            len(rest), 1, "Full hit should leave at most 1 token for stream_generate"
        )

    def test_save_post_prefill_snapshot_thinking_model(self):
        """Thinking models: checkpoint key excludes <think> token and everything after."""
        tokenizer = _MockThinkingTokenizer()
        wrapper = self._make_wrapper(tokenizer)

        # tokens[-2] is the think token → offset = 2 + 1 = 3 → key = tokens[:-3]
        tokens = [1, 2, 3, 8888, 4]
        wrapper.save_post_prefill_snapshot(tokens, _make_cache())

        # Fetching with the full token list should hit (key covers tokens[:2])
        result, rest = wrapper.fetch_continuation_cache(tokens)
        self.assertIsNotNone(result)
        # rest must contain the 3 tokens not in the key
        self.assertEqual(len(rest), 3)

    def test_clear_text_preserves_image_checkpoints(self):
        """clear_text() drops LRU snapshots but keeps image checkpoints intact."""
        wrapper = self._make_wrapper()
        tokens = [1, 2, 3, 4]
        wrapper.save_post_prefill_snapshot(tokens, _make_cache())
        wrapper.save_image_checkpoint(("abc",), _make_cache(), 10, 0xABC)

        wrapper.clear_text()

        lru_result, _ = wrapper.fetch_continuation_cache(tokens)
        self.assertIsNone(lru_result, "LRU should be empty after clear_text()")
        self.assertIsNotNone(
            wrapper.get_image_checkpoint(("abc",)),
            "Image checkpoint should survive clear_text()",
        )

    # ------------------------------------------------------------------
    # fetch_pre_image_cache
    # ------------------------------------------------------------------

    def test_fetch_pre_image_cache_exact_hit(self):
        """LRU prefix exactly matches text before first image block."""
        wrapper = self._make_wrapper()
        # Manually insert a cache for the pre-image prefix [1, 2, 3]
        wrapper._lru.insert_cache("model", [1, 2, 3], _make_cache(), checkpoint=True)

        flat_ids = [1, 2, 3, 9999, 100, 101]  # image token at index 3
        cache, rest, first_img = wrapper.fetch_pre_image_cache(flat_ids, 9999, 9999)

        self.assertIsNotNone(cache)
        self.assertEqual(rest, [], "Exact hit: no remaining tokens to top up")
        self.assertEqual(first_img, 3)

    def test_fetch_pre_image_cache_partial_hit(self):
        """LRU covers a prefix shorter than the pre-image text → rest is non-empty."""
        wrapper = self._make_wrapper()
        wrapper._lru.insert_cache("model", [1, 2], _make_cache(), checkpoint=True)

        flat_ids = [1, 2, 3, 9999, 100]
        cache, rest, first_img = wrapper.fetch_pre_image_cache(flat_ids, 9999, 9999)

        self.assertIsNotNone(cache)
        self.assertEqual(rest, [3], "Should top up with the unmatched token")
        self.assertEqual(first_img, 3)

    def test_fetch_pre_image_cache_miss(self):
        """Empty LRU returns (None, full_prefix, first_image_start)."""
        wrapper = self._make_wrapper()
        flat_ids = [50, 51, 9999, 100]
        cache, rest, first_img = wrapper.fetch_pre_image_cache(flat_ids, 9999, 9999)

        self.assertIsNone(cache)
        self.assertEqual(rest, [50, 51])
        self.assertEqual(first_img, 2)

    def test_fetch_pre_image_cache_image_at_start(self):
        """Image token at index 0: no pre-image text, no LRU lookup attempted."""
        wrapper = self._make_wrapper()
        # Insert something in LRU to verify it is NOT consulted
        wrapper._lru.insert_cache("model", [9999], _make_cache(), checkpoint=True)

        flat_ids = [9999, 1, 2, 3]
        cache, rest, first_img = wrapper.fetch_pre_image_cache(flat_ids, 9999, 9999)

        self.assertIsNone(cache)
        self.assertEqual(rest, [])
        self.assertEqual(first_img, 0)

    def test_fetch_pre_image_cache_no_image_tokens(self):
        """Sequence contains no image tokens: returns miss with full list."""
        wrapper = self._make_wrapper()
        flat_ids = [1, 2, 3, 4]
        cache, rest, first_img = wrapper.fetch_pre_image_cache(flat_ids, 9999, 9999)

        self.assertIsNone(cache)
        self.assertEqual(rest, [1, 2, 3, 4])
        self.assertEqual(first_img, 4)  # len(flat_ids), no image found

    def test_fetch_pre_image_cache_video_token(self):
        """Video tokens (vid_tok ≠ img_tok) are treated as image boundaries."""
        wrapper = self._make_wrapper()
        wrapper._lru.insert_cache("model", [1, 2], _make_cache(), checkpoint=True)

        img_tok, vid_tok = 9999, 8888
        flat_ids = [1, 2, 8888, 100]  # video token at index 2
        cache, rest, first_img = wrapper.fetch_pre_image_cache(
            flat_ids, img_tok, vid_tok
        )

        self.assertIsNotNone(cache)
        self.assertEqual(rest, [])
        self.assertEqual(first_img, 2)

    # ------------------------------------------------------------------
    # prefix hash validation (save / invalidate / get_image_checkpoint)
    # ------------------------------------------------------------------

    def test_save_image_checkpoint_stores_prefix_hash(self):
        """Saved checkpoint exposes (snapshot, end_idx, prefix_hash, block_lengths)."""
        wrapper = self._make_wrapper()
        cache = _make_cache()
        wrapper.save_image_checkpoint(("h1",), cache, 42, 0xDEAD, (100,))

        entry = wrapper.get_image_checkpoint(("h1",))
        self.assertIsNotNone(entry)
        self.assertEqual(len(entry), 4)
        snapshot, end_idx, pfx_hash, block_lengths = entry
        self.assertIs(snapshot, cache)
        self.assertEqual(end_idx, 42)
        self.assertEqual(pfx_hash, 0xDEAD)
        self.assertEqual(block_lengths, (100,))

    def test_invalidate_image_checkpoint_removes_entry(self):
        """invalidate_image_checkpoint() removes the entry; subsequent lookup returns None."""
        wrapper = self._make_wrapper()
        wrapper.save_image_checkpoint(("h1",), _make_cache(), 10, 0xABC)
        self.assertIsNotNone(wrapper.get_image_checkpoint(("h1",)))

        wrapper.invalidate_image_checkpoint(("h1",))
        self.assertIsNone(wrapper.get_image_checkpoint(("h1",)))

    def test_invalidate_image_checkpoint_missing_key_is_noop(self):
        """invalidate_image_checkpoint() on a non-existent key does not raise."""
        wrapper = self._make_wrapper()
        wrapper.invalidate_image_checkpoint(("nonexistent",))  # must not raise

    def test_prefix_hash_mismatch_detected(self):
        """Storing a checkpoint with one hash and querying with another is detectable."""
        wrapper = self._make_wrapper()
        ids = [1, 2, 3, 4, 5]
        end_idx = 3
        correct_hash = hash(tuple(ids[:end_idx]))  # hash of [1, 2, 3]
        wrong_hash = correct_hash ^ 0xFFFFFFFF  # guaranteed to differ

        wrapper.save_image_checkpoint(("img",), _make_cache(), end_idx, wrong_hash)
        entry = wrapper.get_image_checkpoint(("img",))
        self.assertIsNotNone(entry)
        # Simulate the validation logic in generate.py
        _, stored_end, stored_hash, _bl = entry
        current_hash = hash(tuple(ids[:stored_end]))
        self.assertNotEqual(current_hash, stored_hash, "Mismatch should be detected")

    def test_prefix_hash_match_accepted(self):
        """Checkpoint saved with the correct prefix hash passes validation."""
        wrapper = self._make_wrapper()
        ids = [10, 20, 30, 40]
        end_idx = 3
        pfx_hash = hash(tuple(ids[:end_idx]))

        wrapper.save_image_checkpoint(("img",), _make_cache(), end_idx, pfx_hash)
        entry = wrapper.get_image_checkpoint(("img",))
        self.assertIsNotNone(entry)
        _, stored_end, stored_hash, _bl = entry
        current_hash = hash(tuple(ids[:stored_end]))
        self.assertEqual(current_hash, stored_hash, "Matching hash should pass")

    # ------------------------------------------------------------------
    # reorder_images_chronologically
    # ------------------------------------------------------------------

    def _make_hashes(self, *labels: str) -> list:
        """Return stable fake hex digests for each label."""
        return [f"{label:0>64}" for label in labels]

    def test_reorder_no_checkpoints_keeps_bridge_order(self):
        """With no checkpoints, bridge order is returned unchanged."""
        wrapper = self._make_wrapper()
        imgs = ["img_c", "img_b", "img_a"]
        hashes = self._make_hashes("c", "b", "a")

        out_imgs, out_hashes = wrapper.reorder_images_chronologically(imgs, hashes)

        self.assertEqual(out_imgs, imgs)
        self.assertEqual(out_hashes, hashes)

    def test_reorder_two_images_no_checkpoint_unchanged(self):
        """N=2 with no checkpoint: order preserved (mirrors bridge correct behaviour)."""
        wrapper = self._make_wrapper()
        imgs = ["img_a", "img_b"]
        hashes = self._make_hashes("a", "b")

        out_imgs, _ = wrapper.reorder_images_chronologically(imgs, hashes)

        self.assertEqual(out_imgs, ["img_a", "img_b"])

    def test_reorder_restores_chronological_order(self):
        """Bridge sends [C, B, A]; checkpoint (a,)(a,b) exists → reordered to [A, B, C]."""
        wrapper = self._make_wrapper()
        ha, hb, hc = self._make_hashes("a", "b", "c")

        wrapper.save_image_checkpoint((ha,), _make_cache(), 10, 0)
        wrapper.save_image_checkpoint((ha, hb), _make_cache(), 20, 0)

        imgs_bridge = ["img_c", "img_b", "img_a"]
        hashes_bridge = [hc, hb, ha]

        out_imgs, out_hashes = wrapper.reorder_images_chronologically(
            imgs_bridge, hashes_bridge
        )

        self.assertEqual(out_imgs, ["img_a", "img_b", "img_c"])
        self.assertEqual(out_hashes, [ha, hb, hc])

    def test_reorder_new_image_appended_after_known(self):
        """Bridge sends [B, C, D, A]; checkpoint (a,b,c) exists → [A, B, C, D]."""
        wrapper = self._make_wrapper()
        ha, hb, hc, hd = self._make_hashes("a", "b", "c", "d")

        wrapper.save_image_checkpoint((ha,), _make_cache(), 10, 0)
        wrapper.save_image_checkpoint((ha, hb), _make_cache(), 20, 0)
        wrapper.save_image_checkpoint((ha, hb, hc), _make_cache(), 30, 0)

        imgs_bridge = ["img_b", "img_c", "img_d", "img_a"]
        hashes_bridge = [hb, hc, hd, ha]

        out_imgs, out_hashes = wrapper.reorder_images_chronologically(
            imgs_bridge, hashes_bridge
        )

        self.assertEqual(out_imgs, ["img_a", "img_b", "img_c", "img_d"])
        self.assertEqual(out_hashes, [ha, hb, hc, hd])

    def test_reorder_already_correct_order_unchanged(self):
        """Bridge sends images in correct order: output equals input."""
        wrapper = self._make_wrapper()
        ha, hb, hc = self._make_hashes("a", "b", "c")

        wrapper.save_image_checkpoint((ha,), _make_cache(), 10, 0)
        wrapper.save_image_checkpoint((ha, hb), _make_cache(), 20, 0)

        imgs = ["img_a", "img_b", "img_c"]
        hashes = [ha, hb, hc]

        out_imgs, out_hashes = wrapper.reorder_images_chronologically(imgs, hashes)

        self.assertEqual(out_imgs, imgs)
        self.assertEqual(out_hashes, hashes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
