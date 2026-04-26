import tempfile
import time
import unittest
from pathlib import Path

import mlx.core as mx

import mlx_engine.disk_kv_cache as dkvc
from mlx_engine.cache_wrapper import CacheWrapper
from mlx_engine.disk_kv_cache import PagedDiskKVCache, compute_block_hashes
from mlx_engine.generate import load_model, tokenize
from tests.shared import RecordingReporter, model_getter

MODEL_NAME = "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-8bit"
# Small block size so tests don't need 2048-token prompts.
TEST_BLOCK_SIZE = 64


def _make_long_prompt(n_tokens: int) -> str:
    """Return a repetitive string that tokenizes to roughly n_tokens."""
    word = "hello world testing the disk kv cache with a repeating sequence "
    return (word * (n_tokens // 10 + 1))[: n_tokens * 4]


class TestHashChaining(unittest.TestCase):
    """Pure unit tests, no model required."""

    def test_deterministic(self):
        tokens = list(range(TEST_BLOCK_SIZE * 3))
        self.assertEqual(
            compute_block_hashes(tokens, TEST_BLOCK_SIZE),
            compute_block_hashes(tokens, TEST_BLOCK_SIZE),
        )

    def test_chaining_propagates(self):
        """Changing block 0 must change hashes for blocks 1 and 2."""
        tokens_a = list(range(TEST_BLOCK_SIZE * 3))
        tokens_b = [999] + list(range(1, TEST_BLOCK_SIZE * 3))
        ha = compute_block_hashes(tokens_a, TEST_BLOCK_SIZE)
        hb = compute_block_hashes(tokens_b, TEST_BLOCK_SIZE)
        for i in range(3):
            self.assertNotEqual(ha[i], hb[i], f"block {i} should differ")

    def test_common_prefix_same_hash(self):
        """Blocks with identical content + identical ancestors have the same hash."""
        prefix = list(range(TEST_BLOCK_SIZE))
        tokens_c = prefix + [42] * TEST_BLOCK_SIZE * 2
        tokens_d = prefix + [99] * TEST_BLOCK_SIZE * 2
        hc = compute_block_hashes(tokens_c, TEST_BLOCK_SIZE)
        hd = compute_block_hashes(tokens_d, TEST_BLOCK_SIZE)
        self.assertEqual(hc[0], hd[0], "block 0 same content → same hash")
        self.assertNotEqual(hc[1], hd[1], "block 1 differs")

    def test_partial_prompt_no_block(self):
        """Prompt shorter than block_size produces no hashes."""
        self.assertEqual(compute_block_hashes([1, 2, 3], TEST_BLOCK_SIZE), [])


class TestPagedDiskKVCache(unittest.TestCase):
    """Integration tests using Qwen2.5-0.5B and a temporary cache directory."""

    @classmethod
    def setUpClass(cls):
        model_path = model_getter(MODEL_NAME)
        cls.model_kit = load_model(
            model_path=model_path, max_kv_size=4096, max_seq_nums=1
        )
        cls.model_path_str = str(model_path)

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _make_cache_wrapper(self) -> CacheWrapper:
        cw = CacheWrapper(
            self.model_kit.model,
            max_kv_size=4096,
            chunk_size=TEST_BLOCK_SIZE,
        )
        cw._model_path = self.model_path_str
        cw._disk_store = PagedDiskKVCache(
            cache_dir=self.cache_dir, block_size=TEST_BLOCK_SIZE
        )
        return cw

    def _prefill(self, cache_wrapper: CacheWrapper, prompt: str) -> int:
        """Prefill and return cached_tokens from the begin event."""
        tokens = mx.array(tokenize(self.model_kit, prompt))
        reporter = RecordingReporter()
        cache_wrapper.update_cache(
            prompt_tokens=tokens, reporter=reporter, num_tokens_to_exclude=1
        )
        cache_wrapper.finalize_generation()
        begin = next(e for e in reporter.events if e["type"] == "begin")
        return begin["cached_tokens"]

    def test_blocks_saved_during_prefill(self):
        """After a prefill longer than block_size, at least one block is on disk."""
        cw = self._make_cache_wrapper()
        prompt = _make_long_prompt(TEST_BLOCK_SIZE * 3)
        self._prefill(cw, prompt)

        safetensors = list(self.cache_dir.glob("*.safetensors"))
        self.assertGreater(len(safetensors), 0, "expected at least one block saved")

    def test_disk_cache_hit_on_second_prefill(self):
        """Second prefill with the same prompt should hit the disk cache."""
        cw_first = self._make_cache_wrapper()
        prompt = _make_long_prompt(TEST_BLOCK_SIZE * 3)
        self._prefill(cw_first, prompt)

        n_saved = len(list(self.cache_dir.glob("*.safetensors")))
        self.assertGreater(n_saved, 0)

        # New CacheWrapper — no in-memory LRU, but same disk store.
        cw_second = self._make_cache_wrapper()
        tokens = mx.array(tokenize(self.model_kit, prompt))
        reporter = RecordingReporter()
        cw_second.update_cache(
            prompt_tokens=tokens, reporter=reporter, num_tokens_to_exclude=1
        )
        begin = next(e for e in reporter.events if e["type"] == "begin")
        cached = begin["cached_tokens"]
        self.assertGreaterEqual(
            cached,
            TEST_BLOCK_SIZE,
            f"expected disk cache hit of at least {TEST_BLOCK_SIZE} tokens, got {cached}",
        )

    def test_sysprompt_stability(self):
        """Sysprompt block hash must be identical across two different conversations."""
        sysprompt = _make_long_prompt(TEST_BLOCK_SIZE * 2)
        turn_a = sysprompt + " user: what is 2+2?"
        turn_b = sysprompt + " user: tell me a joke"

        tokens_a = tokenize(self.model_kit, turn_a)
        tokens_b = tokenize(self.model_kit, turn_b)

        hashes_a = compute_block_hashes(tokens_a, TEST_BLOCK_SIZE)
        hashes_b = compute_block_hashes(tokens_b, TEST_BLOCK_SIZE)

        # All blocks covered by the sysprompt must have identical hashes.
        sysprompt_blocks = len(tokenize(self.model_kit, sysprompt)) // TEST_BLOCK_SIZE
        self.assertGreater(sysprompt_blocks, 0)
        for i in range(sysprompt_blocks):
            self.assertEqual(
                hashes_a[i],
                hashes_b[i],
                f"sysprompt block {i} should have the same hash across conversations",
            )

    def test_eviction_prefers_low_hit_count(self):
        """Under disk pressure, blocks with lower hit_count are evicted first."""
        store = PagedDiskKVCache(cache_dir=self.cache_dir, block_size=TEST_BLOCK_SIZE)

        # Inject two fake manifest entries: one with high hit_count, one with low.
        store._manifest["aaa"] = {
            "model_path": "m",
            "file_size": 100,
            "last_used": time.time(),
            "hit_count": 10,
        }
        store._manifest["bbb"] = {
            "model_path": "m",
            "file_size": 100,
            "last_used": time.time(),
            "hit_count": 0,
        }
        # Create dummy files so unlink doesn't fail.
        (self.cache_dir / "aaa.safetensors").write_bytes(b"x" * 100)
        (self.cache_dir / "bbb.safetensors").write_bytes(b"x" * 100)

        # Cap at 150 bytes — must evict one entry.
        original_cap = dkvc._MAX_CACHE_BYTES
        dkvc._MAX_CACHE_BYTES = 150
        try:
            store._evict_if_needed()
        finally:
            dkvc._MAX_CACHE_BYTES = original_cap

        self.assertIn("aaa", store._manifest, "high hit_count block should survive")
        self.assertNotIn(
            "bbb", store._manifest, "low hit_count block should be evicted"
        )
