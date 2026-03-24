"""Shared helper for marker-based boundary detectors.

Each plugin calls ``make_marker_detector(markers)`` to obtain a
``find_stable_boundary`` function that:

1. Decodes only the system message (tokens before the first ``<|im_end|>``).
2. Searches for the earliest occurrence of any marker string.
3. Binary-searches the token index matching that character position.

The returned function returns ``None`` when no marker is found, so the
generic ChatML fallback in ``cache_wrapper`` takes over without regression.
"""

from __future__ import annotations

from typing import Callable, List, Optional


def make_marker_detector(markers: List[str]) -> Callable:
    """Return a ``find_stable_boundary(prompt_tokens, tokenizer)`` function
    that cuts the system prompt just before the earliest ``markers`` match."""

    def find_stable_boundary(prompt_tokens: list, tokenizer) -> Optional[int]:
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            return None

        im_end_id = vocab.get("<|im_end|>")
        if im_end_id is None:
            return None

        try:
            sys_end = prompt_tokens.index(im_end_id)
        except ValueError:
            return None

        if sys_end == 0:
            return None

        sys_tokens = prompt_tokens[:sys_end]

        try:
            sys_text = tokenizer.decode(sys_tokens)
        except Exception:
            return None

        earliest_pos: Optional[int] = None
        for marker in markers:
            pos = sys_text.find(marker)
            if pos >= 0 and (earliest_pos is None or pos < earliest_pos):
                earliest_pos = pos

        if earliest_pos is None:
            return None

        lo, hi = 0, len(sys_tokens)
        while lo < hi:
            mid = (lo + hi) // 2
            try:
                decoded_len = len(tokenizer.decode(sys_tokens[:mid]))
            except Exception:
                return None
            if decoded_len <= earliest_pos:
                lo = mid + 1
            else:
                hi = mid

        return lo

    return find_stable_boundary
