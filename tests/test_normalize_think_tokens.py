"""Tests for CacheWrapper._normalize_think_tokens.

Strips complete <think>...</think> blocks for think-invariant LRU keys.
Incomplete trailing blocks are kept (handled downstream by _checkpoint_offset).
Pure-whitespace tokens immediately after </think> are also stripped.
"""

import types
import unittest

from mlx_engine.cache_wrapper import CacheWrapper

THINK_START = 1000
THINK_END = 1001
SEP = 9999  # decodes to "\n\n" (whitespace)
SEP2 = 8888  # decodes to "\n"   (whitespace)


def _make_tokenizer(has_thinking=True):
    _WS = {SEP: "\n\n", SEP2: "\n"}
    tok = types.SimpleNamespace()
    tok.has_thinking = has_thinking
    tok.think_start_id = THINK_START
    tok.think_end_id = THINK_END
    tok.decode = lambda ids: _WS.get(ids[0], f"tok{ids[0]}") if ids else ""
    return tok


def _normalize(tokenizer, tokens):
    obj = object.__new__(CacheWrapper)
    obj._tokenizer = tokenizer
    return obj._normalize_think_tokens(tokens)


class TestNormalizeThinkTokens(unittest.TestCase):
    def test_passthrough_when_disabled(self):
        """No tokenizer or thinking disabled → tokens returned unchanged."""
        tokens = [THINK_START, 10, THINK_END]
        for tok in (None, _make_tokenizer(has_thinking=False)):
            obj = object.__new__(CacheWrapper)
            obj._tokenizer = tok
            norm, orig = obj._normalize_think_tokens(tokens)
            self.assertEqual(norm, tokens)
            self.assertEqual(orig, list(range(len(tokens))))

    def test_complete_block_stripped(self):
        """A complete <think>…</think> block is stripped; surrounding tokens kept."""
        tok = _make_tokenizer()
        tokens = [1, 2, THINK_START, 10, 11, THINK_END, 3, 4]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2, 3, 4])
        self.assertEqual(orig, [0, 1, 6, 7])

    def test_trailing_open_block_kept(self):
        """An open <think> with no closing tag is preserved unchanged."""
        tok = _make_tokenizer()
        tokens = [1, 2, THINK_START, 10, 11]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, tokens)
        self.assertEqual(orig, list(range(len(tokens))))

    def test_complete_then_open(self):
        """Complete block stripped; subsequent trailing open block kept."""
        tok = _make_tokenizer()
        tokens = [THINK_START, 10, THINK_END, 5, THINK_START, 20]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [5, THINK_START, 20])
        self.assertEqual(orig, [3, 4, 5])

    def test_orig_indices_map_back(self):
        """norm[i] == tokens[orig[i]] holds for every position after stripping."""
        tok = _make_tokenizer()
        tokens = [1, THINK_START, 99, THINK_END, 2, 3]
        norm, orig = _normalize(tok, tokens)
        for i, idx in enumerate(orig):
            self.assertEqual(norm[i], tokens[idx])


class TestNormalizeThinkTokensSep(unittest.TestCase):
    """Post-</think> whitespace stripping."""

    def test_whitespace_tokens_stripped_after_think_end(self):
        """All consecutive whitespace tokens after THINK_END are stripped."""
        tok = _make_tokenizer()
        tokens = [1, THINK_START, 10, THINK_END, SEP, SEP2, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2])
        self.assertEqual(orig, [0, 6])

    def test_non_whitespace_after_think_end_kept(self):
        """Non-whitespace token after THINK_END stops stripping and is kept."""
        tok = _make_tokenizer()
        tokens = [1, THINK_START, 10, THINK_END, 99, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 99, 2])
        self.assertEqual(orig, [0, 4, 5])

    def test_sep_inside_open_block_not_stripped(self):
        """Whitespace inside a trailing open block is kept unchanged."""
        tok = _make_tokenizer()
        tokens = [1, THINK_START, 10, SEP]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, tokens)
        self.assertEqual(orig, list(range(len(tokens))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
