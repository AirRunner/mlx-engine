"""Tests for CacheWrapper._normalize_think_tokens.

_normalize_think_tokens strips complete <think>...</think> blocks from a token
list to produce a think-invariant LRU key. The function is exercised here
without a real model by constructing a minimal CacheWrapper-like object that
only has the _tokenizer attribute set.
"""

import types
import unittest

from mlx_engine.cache_wrapper import CacheWrapper

THINK_START = 1000
THINK_END = 1001


SEP = 9999  # Fake post-think separator — decodes to "\n\n" (whitespace).
SEP2 = 8888  # Second fake separator — decodes to "\n" (whitespace).


def _make_tokenizer(has_thinking: bool = True) -> types.SimpleNamespace:
    """Return a minimal fake tokenizer.

    ``decode([id])`` maps known whitespace token IDs (SEP, SEP2) to their
    whitespace strings so that the lazy whitespace-detection path in
    ``_normalize_think_tokens`` works correctly in unit tests.
    All other token IDs decode to a non-whitespace placeholder string.
    """
    _WS = {SEP: "\n\n", SEP2: "\n"}

    tok = types.SimpleNamespace()
    tok.has_thinking = has_thinking
    tok.think_start_id = THINK_START
    tok.think_end_id = THINK_END
    tok.decode = lambda ids: _WS.get(ids[0], f"tok{ids[0]}") if ids else ""
    return tok


def _normalize(tokenizer, tokens: list) -> tuple:
    """Call _normalize_think_tokens on a bare CacheWrapper instance."""
    obj = object.__new__(CacheWrapper)
    obj._tokenizer = tokenizer
    return obj._normalize_think_tokens(tokens)


class TestNormalizeThinkTokens(unittest.TestCase):
    # ------------------------------------------------------------------
    # No-op cases
    # ------------------------------------------------------------------

    def test_no_tokenizer(self):
        """No tokenizer -> tokens returned unchanged."""
        obj = object.__new__(CacheWrapper)
        obj._tokenizer = None
        tokens = [1, 2, 3]
        norm, orig = obj._normalize_think_tokens(tokens)
        self.assertEqual(norm, tokens)
        self.assertEqual(orig, list(range(len(tokens))))

    def test_thinking_disabled(self):
        """Tokenizer without thinking support -> tokens returned unchanged."""
        tok = _make_tokenizer(has_thinking=False)
        tokens = [THINK_START, 10, THINK_END]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, tokens)
        self.assertEqual(orig, list(range(len(tokens))))

    def test_no_think_tokens(self):
        """No think tokens in input -> returned unchanged."""
        tok = _make_tokenizer()
        tokens = [1, 2, 3, 4, 5]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, tokens)
        self.assertEqual(orig, list(range(len(tokens))))

    def test_empty_input(self):
        tok = _make_tokenizer()
        norm, orig = _normalize(tok, [])
        self.assertEqual(norm, [])
        self.assertEqual(orig, [])

    # ------------------------------------------------------------------
    # Complete block stripping
    # ------------------------------------------------------------------

    def test_single_complete_block(self):
        """A single complete <think>...</think> block is stripped."""
        tok = _make_tokenizer()
        # [prefix] <think> reasoning </think> [suffix]
        tokens = [1, 2, THINK_START, 10, 11, THINK_END, 3, 4]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2, 3, 4])
        self.assertEqual(orig, [0, 1, 6, 7])

    def test_empty_complete_block(self):
        """An empty <think></think> block is also stripped."""
        tok = _make_tokenizer()
        tokens = [1, THINK_START, THINK_END, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2])
        self.assertEqual(orig, [0, 3])

    def test_multiple_complete_blocks(self):
        """Multiple complete blocks are all stripped, interleaved content preserved."""
        tok = _make_tokenizer()
        tokens = [
            THINK_START,
            10,
            THINK_END,  # block 1
            5,  # content
            THINK_START,
            20,
            THINK_END,  # block 2
            6,  # content
        ]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [5, 6])
        self.assertEqual(orig, [3, 7])

    def test_block_at_start(self):
        """Complete block at the very start is stripped."""
        tok = _make_tokenizer()
        tokens = [THINK_START, 99, THINK_END, 7, 8]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [7, 8])
        self.assertEqual(orig, [3, 4])

    def test_block_at_end(self):
        """Complete block at the very end (with closing tag) is stripped."""
        tok = _make_tokenizer()
        tokens = [1, 2, THINK_START, 99, THINK_END]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2])
        self.assertEqual(orig, [0, 1])

    # ------------------------------------------------------------------
    # Trailing open block (no closing tag)
    # ------------------------------------------------------------------

    def test_trailing_open_block_kept(self):
        """An open <think> with no closing tag is kept (handled by _checkpoint_offset)."""
        tok = _make_tokenizer()
        tokens = [1, 2, THINK_START, 10, 11]  # no THINK_END
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, tokens)
        self.assertEqual(orig, list(range(len(tokens))))

    def test_complete_then_open(self):
        """Complete block stripped, trailing open block kept."""
        tok = _make_tokenizer()
        tokens = [
            THINK_START,
            10,
            THINK_END,  # complete block
            5,  # content
            THINK_START,
            20,  # trailing open block (no end)
        ]
        norm, orig = _normalize(tok, tokens)
        # Complete block stripped; trailing open block and its content kept
        self.assertEqual(norm, [5, THINK_START, 20])
        self.assertEqual(orig, [3, 4, 5])

    # ------------------------------------------------------------------
    # orig_indices consistency
    # ------------------------------------------------------------------

    def test_orig_indices_monotone(self):
        """orig_indices must always be strictly monotonically increasing."""
        tok = _make_tokenizer()
        tokens = [
            1,
            THINK_START,
            10,
            11,
            THINK_END,
            2,
            THINK_START,
            20,
            THINK_END,
            3,
            4,
        ]
        _, orig = _normalize(tok, tokens)
        self.assertEqual(orig, sorted(set(orig)))  # strictly increasing
        for a, b in zip(orig, orig[1:]):
            self.assertLess(a, b)

    def test_orig_indices_map_back_correctly(self):
        """norm_tokens[i] == tokens[orig_indices[i]] for all i."""
        tok = _make_tokenizer()
        tokens = [1, THINK_START, 99, THINK_END, 2, 3]
        norm, orig = _normalize(tok, tokens)
        for i, idx in enumerate(orig):
            self.assertEqual(norm[i], tokens[idx])


class TestNormalizeThinkTokensSep(unittest.TestCase):
    """Tests for post-think whitespace stripping.

    The Qwen3.5 chat template emits \\n</think>\\n\\ncontent; the \\n\\n after
    THINK_END is stripped by decoding each token and checking whether it is pure
    whitespace, so the normalized key matches the no-think rendering regardless of
    separator length or token encoding.

    SEP  (9999) decodes to "\\n\\n" — whitespace, will be stripped.
    SEP2 (8888) decodes to "\\n"   — whitespace, will be stripped.
    Token 99 decodes to "tok99"    — non-whitespace, will NOT be stripped.
    """

    # ------------------------------------------------------------------
    # Single whitespace-token separator
    # ------------------------------------------------------------------

    def test_sep_stripped_after_complete_block(self):
        """Whitespace token after THINK_END is stripped."""
        tok = _make_tokenizer()
        # [1, THINK_START, 10, THINK_END, SEP, 2] → [1, 2]
        tokens = [1, THINK_START, 10, THINK_END, SEP, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2])
        self.assertEqual(orig, [0, 5])

    def test_sep_stripped_multiple_blocks(self):
        """Whitespace separator stripped after each of several complete blocks."""
        tok = _make_tokenizer()
        tokens = [
            1,
            THINK_START,
            10,
            THINK_END,
            SEP,  # block 1 + sep
            2,
            THINK_START,
            20,
            THINK_END,
            SEP,  # block 2 + sep
            3,
        ]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2, 3])
        self.assertEqual(orig, [0, 5, 10])

    def test_non_whitespace_after_think_end_kept(self):
        """Non-whitespace token after THINK_END is kept (stops stripping)."""
        tok = _make_tokenizer()
        # 99 → "tok99" (non-whitespace) → not stripped
        tokens = [1, THINK_START, 10, THINK_END, 99, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 99, 2])
        self.assertEqual(orig, [0, 4, 5])

    def test_sep_not_stripped_for_trailing_open_block(self):
        """Trailing open block (no THINK_END) keeps everything including sep."""
        tok = _make_tokenizer()
        tokens = [1, THINK_START, 10, SEP]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, tokens)
        self.assertEqual(orig, list(range(len(tokens))))

    # ------------------------------------------------------------------
    # Multi-token whitespace separator
    # ------------------------------------------------------------------

    def test_multi_token_sep_stripped_fully(self):
        """Multiple consecutive whitespace tokens are all stripped after THINK_END."""
        tok = _make_tokenizer()
        # SEP="\n\n", SEP2="\n" — both whitespace → both stripped
        tokens = [1, THINK_START, 10, THINK_END, SEP, SEP2, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2])
        self.assertEqual(orig, [0, 6])

    def test_multi_token_sep_stops_at_non_whitespace(self):
        """Stripping stops as soon as a non-whitespace token is encountered."""
        tok = _make_tokenizer()
        # SEP stripped, then 99 ("tok99") → stop; 99 is kept
        tokens = [1, THINK_START, 10, THINK_END, SEP, 99, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 99, 2])
        self.assertEqual(orig, [0, 5, 6])

    def test_three_newline_tokens_all_stripped(self):
        """Three consecutive whitespace tokens (e.g. \\n\\n\\n) are all stripped."""
        tok = _make_tokenizer()
        # SEP2+SEP2+SEP2 = three "\n" tokens = all whitespace → all stripped
        tokens = [1, THINK_START, 10, THINK_END, SEP2, SEP2, SEP2, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [1, 2])
        self.assertEqual(orig, [0, 7])

    # ------------------------------------------------------------------
    # Template patterns from the user's list
    # ------------------------------------------------------------------

    def test_template_pattern_newline_think_block_with_sep(self):
        """[\\n, THINK_START, SEP, THINK_END, SEP, content] → [\\n, content]."""
        tok = _make_tokenizer()
        NEWLINE = SEP2  # "\n" is whitespace but appears BEFORE THINK_START → kept
        tokens = [NEWLINE, THINK_START, SEP, THINK_END, SEP, 2]
        norm, orig = _normalize(tok, tokens)
        # NEWLINE before THINK_START is kept (not inside the block)
        self.assertEqual(norm, [NEWLINE, 2])
        self.assertEqual(orig, [0, 5])

    def test_template_pattern_no_sep_after_think_end(self):
        """[\\n, THINK_START, SEP, THINK_END, content] → [\\n, content]."""
        tok = _make_tokenizer()
        NEWLINE = SEP2
        tokens = [NEWLINE, THINK_START, SEP, THINK_END, 2]
        norm, orig = _normalize(tok, tokens)
        self.assertEqual(norm, [NEWLINE, 2])
        self.assertEqual(orig, [0, 4])

    # ------------------------------------------------------------------
    # orig_indices consistency with sep
    # ------------------------------------------------------------------

    def test_orig_indices_map_back_with_sep(self):
        """norm_tokens[i] == tokens[orig_indices[i]] holds with sep stripping."""
        tok = _make_tokenizer()
        tokens = [5, THINK_START, 10, 11, THINK_END, SEP, 6, 7]
        norm, orig = _normalize(tok, tokens)
        for i, idx in enumerate(orig):
            self.assertEqual(norm[i], tokens[idx])

    def test_orig_indices_monotone_with_sep(self):
        """orig_indices are strictly increasing with sep stripping."""
        tok = _make_tokenizer()
        tokens = [
            1,
            THINK_START,
            10,
            THINK_END,
            SEP,
            2,
            THINK_START,
            20,
            THINK_END,
            SEP,
            3,
        ]
        _, orig = _normalize(tok, tokens)
        for a, b in zip(orig, orig[1:]):
            self.assertLess(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
