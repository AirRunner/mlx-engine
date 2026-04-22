import types
import unittest
import mlx.core as mx
from mlx_engine.cache_wrapper import (
    CacheWrapper,
    StopPromptProcessing,
)
from tests.shared import model_getter, RecordingReporter, CancellingReporter
from mlx_engine.generate import load_model, tokenize


class TestCacheWrapper(unittest.TestCase):
    def test_prompt_processing_cancellation(self):
        """Test that progress is saved when processing is cancelled and cache is reused on retry"""

        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        model_kit = load_model(model_path=model_path, max_kv_size=4096, max_seq_nums=1)

        chunk_size = 20  # Small chunk size to ensure multiple progress callbacks
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

        # First attempt: Reporter that cancels after 3 events
        cancelling_reporter = CancellingReporter(cancel_after=3)

        with self.assertRaises(StopPromptProcessing):
            model_kit.cache_wrapper.update_cache(
                prompt_tokens=prompt_tokens,
                reporter=cancelling_reporter,
                num_tokens_to_exclude=1,
            )
        # Second attempt: Reporter that doesn't cancel
        recording_reporter = RecordingReporter()

        result_tokens = model_kit.cache_wrapper.update_cache(
            prompt_tokens=prompt_tokens,
            reporter=recording_reporter,
            num_tokens_to_exclude=1,
        )
        # Cancellation no longer saves partial progress: LRU has no entry since
        # insert_cache is only called on successful completion. The second attempt
        # does a full re-prefill from scratch.
        begin_event = next(e for e in recording_reporter.events if e["type"] == "begin")
        self.assertEqual(
            begin_event["cached_tokens"],
            0,
            "cancelled run should not save partial progress to the LRU",
        )

        # Verify that the second attempt completed successfully
        self.assertIsNotNone(result_tokens)


class TestCacheWrapperIntegration(unittest.TestCase):
    """Integration tests using the real Qwen2.5-0.5B model.

    These tests exercise the full update_cache / finalize_generation cycle,
    verifying LRU prefix reuse and think-normalization end-to-end.
    The model is loaded once for the whole class to avoid repeated I/O.
    """

    _model_kit = None

    @classmethod
    def setUpClass(cls):
        model_path = model_getter("lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit")
        cls._model_kit = load_model(
            model_path=model_path, max_kv_size=4096, max_seq_nums=1
        )

    def _make_cache_wrapper(self, tokenizer=None):
        assert self._model_kit is not None
        return CacheWrapper(
            self._model_kit.model,
            max_kv_size=4096,
            chunk_size=512,
            tokenizer=tokenizer,
        )

    @staticmethod
    def _make_think_tokenizer(think_start=100, think_end=101):
        tok = types.SimpleNamespace()
        tok.has_thinking = True
        tok.think_start_id = think_start
        tok.think_end_id = think_end
        return tok

    def test_lru_hit_on_continuation(self):
        """Turn N+1 reuses the LRU entry from turn N when it extends the same prefix."""
        assert self._model_kit is not None
        cw = self._make_cache_wrapper()
        base_text = "The quick brown fox jumps over the lazy dog. " * 5

        tokens1 = mx.array(tokenize(self._model_kit, base_text))
        cw.update_cache(tokens1, RecordingReporter(), num_tokens_to_exclude=1)
        cw.finalize_generation()

        tokens2 = mx.array(tokenize(self._model_kit, base_text + " The end."))
        reporter2 = RecordingReporter()
        cw.update_cache(tokens2, reporter2, num_tokens_to_exclude=1)

        begin = reporter2.events[0]
        self.assertEqual(begin["type"], "begin")
        self.assertGreater(
            begin["cached_tokens"],
            0,
            "LRU should hit when turn 2 extends the prefix of turn 1",
        )

    def test_lru_hit_with_think_normalization(self):
        """LRU hits even when think blocks present in turn N are absent in turn N+1."""
        assert self._model_kit is not None
        THINK_START = 100
        THINK_END = 101

        cw = self._make_cache_wrapper(tokenizer=self._make_think_tokenizer())

        prefix_ids = tokenize(self._model_kit, "The quick brown fox. " * 4)
        suffix_ids = tokenize(self._model_kit, " Answer: Yes.")

        # Turn 1: prefix + <think>inner</think> + suffix (think block present).
        turn1 = prefix_ids + [THINK_START, 150, 160, 170, THINK_END] + suffix_ids
        cw.update_cache(mx.array(turn1), RecordingReporter(), num_tokens_to_exclude=1)
        cw.finalize_generation()

        # Turn 2: same prefix + same suffix but WITHOUT the think block, plus an
        # extra token so the sequence is strictly longer (needed for LRU lookup).
        turn2 = prefix_ids + suffix_ids + [200]
        reporter2 = RecordingReporter()
        cw.update_cache(mx.array(turn2), reporter2, num_tokens_to_exclude=1)

        begin = reporter2.events[0]
        self.assertEqual(begin["type"], "begin")
        self.assertGreater(
            begin["cached_tokens"],
            0,
            "LRU should hit via think-normalization when think block absent in turn 2",
        )

    def test_lru_hit_with_think_normalization_and_sep(self):
        """LRU hits even when think blocks include a post-THINK_END whitespace separator."""
        assert self._model_kit is not None
        THINK_START = 100
        THINK_END = 101
        SEP = 102  # fake whitespace separator token

        cw = self._make_cache_wrapper(tokenizer=self._make_think_tokenizer())
        # Pre-populate the whitespace cache so SEP is recognised as pure whitespace.
        cw._think_ws_cache = {SEP: True}

        prefix_ids = tokenize(self._model_kit, "The quick brown fox. " * 4)
        suffix_ids = tokenize(self._model_kit, " Answer: Yes.")

        # Turn 1: prefix + [THINK_START inner THINK_END SEP] + suffix.
        # The SEP token after THINK_END is the residual that must be stripped.
        turn1 = prefix_ids + [THINK_START, 150, 160, THINK_END, SEP] + suffix_ids
        cw.update_cache(mx.array(turn1), RecordingReporter(), num_tokens_to_exclude=1)
        cw.finalize_generation()

        # Turn 2: prefix + suffix WITHOUT think block and WITHOUT SEP.
        # Without sep stripping, the stored key (which includes SEP residual)
        # would be longer than the new request → GDN non-trimmable miss.
        turn2 = prefix_ids + suffix_ids + [200]
        reporter2 = RecordingReporter()
        cw.update_cache(mx.array(turn2), reporter2, num_tokens_to_exclude=1)

        begin = reporter2.events[0]
        self.assertEqual(begin["type"], "begin")
        self.assertGreater(
            begin["cached_tokens"],
            0,
            "LRU should hit after stripping both think block and its post-sep token",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
