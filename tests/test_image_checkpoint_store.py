"""Tests for ImageCheckpointStore.

Metadata-only store keyed by image hash chains. No KV tensors are stored;
the text LRU provides the starting cache at restore time.
All tests are pure Python and require no MLX or model fixtures.
"""

import unittest

from mlx_engine.cache_wrapper import ImageCheckpointStore

IMG_TOK = 42


def _store(*entries):
    """Return a pre-loaded store. entries: (key, end_idx, pfx_hash, lengths)."""
    s = ImageCheckpointStore()
    for key, end_idx, pfx_hash, lengths in entries:
        s.save_image_checkpoint(key, end_idx, pfx_hash, lengths)
    return s


def _hash(ids, end):
    return hash(tuple(ids[:end]))


class TestImageCheckpointStore(unittest.TestCase):
    def test_round_trip_stores_3_tuple(self):
        """Checkpoint is saved and retrieved as (image_end_index, prefix_hash, block_lengths)."""
        s = ImageCheckpointStore()
        s.save_image_checkpoint(("h1",), 100, 12345, (50,))
        self.assertEqual(s.get_image_checkpoint(("h1",)), (100, 12345, (50,)))

    def test_find_deepest_returns_2_tuple_and_deepest_prefix(self):
        """Returns (key, image_end_index) for the longest matching prefix of the hash chain."""
        s = _store(
            (("h1",), 100, 1, (50,)),
            (("h1", "h2"), 200, 2, (50, 60)),
        )
        result = s.find_deepest_image_checkpoint(("h1", "h2", "h3"))
        self.assertEqual(result, (("h1", "h2"), 200))

    def test_find_deepest_none_when_no_match(self):
        s = _store((("h1",), 100, 1, (50,)))
        self.assertIsNone(s.find_deepest_image_checkpoint(("other",)))

    def test_validate_not_stale_when_hash_matches(self):
        ids = list(range(200))
        s = _store((("h1",), 100, _hash(ids, 100), (50,)))
        self.assertFalse(
            s.validate_image_checkpoint(("h1",), ids, IMG_TOK, None, (50,))
        )

    def test_validate_stale_when_out_of_bounds(self):
        ids = list(range(50))  # only 50 tokens, stored end_idx=100
        s = _store((("h1",), 100, 0, (50,)))
        self.assertTrue(s.validate_image_checkpoint(("h1",), ids, IMG_TOK, None, (50,)))

    def test_validate_stale_when_hash_mismatch(self):
        ids = list(range(200))
        s = _store((("h1",), 100, 99999, (50,)))  # wrong hash
        self.assertTrue(s.validate_image_checkpoint(("h1",), ids, IMG_TOK, None, (50,)))

    def test_validate_pad_only_gap_accepted(self):
        # Stored checkpoint had 5 image tokens, re-padded to 10. Extra tokens are IMG_TOK.
        ids = list(range(10)) + [IMG_TOK] * 10 + list(range(20, 30))
        s = _store((("h1",), 15, _hash(ids, 15), (5,)))
        self.assertFalse(
            s.validate_image_checkpoint(("h1",), ids, IMG_TOK, None, (10,))
        )

    def test_validate_stale_when_fewer_blocks_than_depth(self):
        # Depth-2 checkpoint but current sequence has only one image block.
        ids = list(range(10)) + [IMG_TOK] * 5 + list(range(20, 30))
        s = _store((("h1", "h2"), 15, _hash(ids, 15), (5, 8)))
        self.assertTrue(
            s.validate_image_checkpoint(("h1", "h2"), ids, IMG_TOK, None, (5,))
        )

    def test_save_block_checkpoints_with_offset(self):
        """offset=1 skips the first block key and stores only the second."""
        ids = (
            list(range(10))
            + [IMG_TOK] * 5
            + list(range(15, 25))
            + [IMG_TOK] * 5
            + list(range(30, 40))
        )
        s = ImageCheckpointStore()
        s.save_block_checkpoints(("h1", "h2"), 1, [30], ids, (5, 5))
        self.assertIsNone(s.get_image_checkpoint(("h1",)))
        entry = s.get_image_checkpoint(("h1", "h2"))
        self.assertIsNotNone(entry)
        self.assertEqual(entry[0], 30)

    def test_reorder_uses_checkpoint_history(self):
        """Images delivered out of chronological order are reordered."""
        s = _store((("h1", "h2"), 200, 2, (50, 60)))
        imgs, hashes = s.reorder_images_chronologically(
            ["img_h2", "img_h1"], ["h2", "h1"]
        )
        self.assertEqual(hashes, ["h1", "h2"])
        self.assertEqual(imgs, ["img_h1", "img_h2"])

    def test_reorder_new_image_appended_after_known(self):
        """A new image (not in checkpoint) is appended after the known ones."""
        s = _store((("h1",), 100, 1, (50,)))
        imgs, hashes = s.reorder_images_chronologically(
            ["img_h2", "img_h1"], ["h2", "h1"]
        )
        self.assertEqual(hashes, ["h1", "h2"])
        self.assertEqual(imgs, ["img_h1", "img_h2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
