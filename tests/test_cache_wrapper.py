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
        # ceiling division +1 for begin event, +1 for finish event
        expected_events = (tokens_to_process + chunk_size - 1) // chunk_size + 2

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
        second_attempt_event_count = len(recording_reporter.events)

        # Cancellation no longer saves partial progress: LRU has no entry since
        # insert_cache is only called on successful completion. The second attempt
        # does a full re-prefill from scratch.
        self.assertEqual(second_attempt_event_count, expected_events)

        # Verify that the second attempt completed successfully
        self.assertIsNotNone(result_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
