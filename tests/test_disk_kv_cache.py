import math
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


class TestHitCountDecay(unittest.TestCase):
    """Pure unit tests for positional hit_count decay and eviction scoring."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = PagedDiskKVCache(
            cache_dir=Path(self._tmpdir.name), block_size=TEST_BLOCK_SIZE
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def _inject(self, key: str, hit_count: float, age: float, size: int = 100):
        (self.store._cache_dir / f"{key}.safetensors").write_bytes(b"x" * size)
        self.store._manifest[key] = {
            "model_path": "m",
            "file_size": size,
            "last_used": time.time() - age,
            "hit_count": hit_count,
        }

    # _increment_hit_counts

    def test_single_block_gets_weight_one(self):
        self.store._manifest["k0"] = {"hit_count": 0.0}
        self.store._increment_hit_counts(["k0"])
        self.assertAlmostEqual(self.store._manifest["k0"]["hit_count"], 1.0)

    def test_positional_decay_order_and_sum(self):
        for k in ["k0", "k1", "k2"]:
            self.store._manifest[k] = {"hit_count": 0.0}
        self.store._increment_hit_counts(["k0", "k1", "k2"])
        w0 = self.store._manifest["k0"]["hit_count"]
        w1 = self.store._manifest["k1"]["hit_count"]
        w2 = self.store._manifest["k2"]["hit_count"]
        self.assertGreater(w0, w1)
        self.assertGreater(w1, w2)
        self.assertAlmostEqual(w0 + w1 + w2, 1.0)
        self.assertAlmostEqual(w0, 3 / 6)
        self.assertAlmostEqual(w1, 2 / 6)
        self.assertAlmostEqual(w2, 1 / 6)

    def test_missing_keys_skipped_positional_index_preserved(self):
        self.store._manifest["k0"] = {"hit_count": 0.0}
        result = self.store._increment_hit_counts(["k0", "missing"])
        self.assertTrue(result)
        # k0 is still at index 0 in a 2-block chain: weight = 2/3
        self.assertAlmostEqual(self.store._manifest["k0"]["hit_count"], 2 / 3)

    def test_all_missing_returns_false(self):
        self.assertFalse(self.store._increment_hit_counts(["x", "y"]))

    def test_empty_keys_returns_false(self):
        self.assertFalse(self.store._increment_hit_counts([]))

    def test_accumulates_across_calls(self):
        self.store._manifest["k0"] = {"hit_count": 0.0}
        self.store._increment_hit_counts(["k0"])
        self.store._increment_hit_counts(["k0"])
        self.assertAlmostEqual(self.store._manifest["k0"]["hit_count"], 2.0)

    # record_lru_hit

    def test_lru_hit_does_not_update_last_used(self):
        tokens = list(range(TEST_BLOCK_SIZE * 2))
        old_time = time.time() - 1000
        for h in compute_block_hashes(tokens, TEST_BLOCK_SIZE):
            self.store._manifest[h.hex()] = {"hit_count": 0.0, "last_used": old_time}
        self.store.record_lru_hit(tokens, TEST_BLOCK_SIZE * 2)
        for h in compute_block_hashes(tokens, TEST_BLOCK_SIZE):
            self.assertEqual(self.store._manifest[h.hex()]["last_used"], old_time)

    def test_lru_hit_applies_positional_decay(self):
        tokens = list(range(TEST_BLOCK_SIZE * 2))
        for h in compute_block_hashes(tokens, TEST_BLOCK_SIZE):
            self.store._manifest[h.hex()] = {"hit_count": 0.0, "last_used": 0.0}
        self.store.record_lru_hit(tokens, TEST_BLOCK_SIZE * 2)
        hashes = compute_block_hashes(tokens, TEST_BLOCK_SIZE)
        h0 = self.store._manifest[hashes[0].hex()]["hit_count"]
        h1 = self.store._manifest[hashes[1].hex()]["hit_count"]
        self.assertGreater(h0, h1)
        self.assertAlmostEqual(h0 + h1, 1.0)

    def test_lru_hit_below_block_size_is_noop(self):
        """Sub-block cached_token_count must not touch any manifest entry."""
        tokens = list(range(TEST_BLOCK_SIZE * 2))
        for h in compute_block_hashes(tokens, TEST_BLOCK_SIZE):
            self.store._manifest[h.hex()] = {"hit_count": 5.0, "last_used": 0.0}
        self.store.record_lru_hit(tokens, TEST_BLOCK_SIZE - 1)
        for h in compute_block_hashes(tokens, TEST_BLOCK_SIZE):
            self.assertEqual(self.store._manifest[h.hex()]["hit_count"], 5.0)

    # _evict_score

    def test_score_zero_when_hit_count_zero(self):
        entry = {"hit_count": 0.0, "last_used": time.time() - 1}
        self.assertEqual(self.store._evict_score(entry, time.time(), 3600.0), 0.0)

    def test_score_decreases_with_age(self):
        now = time.time()
        tau = 3600.0
        recent = {"hit_count": 1.0, "last_used": now - 60}
        stale = {"hit_count": 1.0, "last_used": now - 7200}
        self.assertGreater(
            self.store._evict_score(recent, now, tau),
            self.store._evict_score(stale, now, tau),
        )

    def test_score_increases_with_hit_count(self):
        now = time.time()
        tau = 3600.0
        low = {"hit_count": 1.0, "last_used": now - 60}
        high = {"hit_count": 10.0, "last_used": now - 60}
        self.assertGreater(
            self.store._evict_score(high, now, tau),
            self.store._evict_score(low, now, tau),
        )

    def test_score_at_one_tau(self):
        tau = 3600.0
        now = time.time()
        entry = {"hit_count": 5.0, "last_used": now - tau}
        self.assertAlmostEqual(
            self.store._evict_score(entry, now, tau), 5.0 * math.exp(-1), places=6
        )

    # _evict_if_needed

    def test_evict_stale_before_recent_same_hit_count(self):
        self._inject("recent", hit_count=1.0, age=60)
        self._inject("stale", hit_count=1.0, age=86400)
        original = dkvc._MAX_CACHE_BYTES
        dkvc._MAX_CACHE_BYTES = 150
        try:
            self.store._evict_if_needed()
        finally:
            dkvc._MAX_CACHE_BYTES = original
        self.assertIn("recent", self.store._manifest)
        self.assertNotIn("stale", self.store._manifest)

    def test_evict_low_hit_count_before_high_same_age(self):
        self._inject("popular", hit_count=20.0, age=3600)
        self._inject("unpopular", hit_count=0.1, age=3600)
        original = dkvc._MAX_CACHE_BYTES
        dkvc._MAX_CACHE_BYTES = 150
        try:
            self.store._evict_if_needed()
        finally:
            dkvc._MAX_CACHE_BYTES = original
        self.assertIn("popular", self.store._manifest)
        self.assertNotIn("unpopular", self.store._manifest)

    def test_evict_stale_high_hit_count_loses_to_fresh_block(self):
        """Stale sysprompt (high hit, high age) evicted before active block."""
        self._inject("stale_sysprompt", hit_count=15.0, age=60 * 86400)
        self._inject("active", hit_count=3.0, age=3600)
        original = dkvc._MAX_CACHE_BYTES
        dkvc._MAX_CACHE_BYTES = 150
        try:
            self.store._evict_if_needed()
        finally:
            dkvc._MAX_CACHE_BYTES = original
        self.assertIn("active", self.store._manifest)
        self.assertNotIn("stale_sysprompt", self.store._manifest)
